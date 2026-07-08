"""
scanner.py — lexical layer of the SAS chunker.

Statement-level parse primitives shared by SasSemanticChunker
(chunker.py) and the metadata extractors (metadata.py):

- _Unit / _Region — the scanner's intermediate representations
- _Deadline / _ParseWatchdog / _record_parser_timeout — stuck-parser
  protection (the design rationale lives in chunker.py's docstring)
- _classify / _classify_normed and the _CLS_* statement classifiers
- _norm / _sanitise / _blank_span and the line-offset helpers

Logging
-------
Logger: ``chunker.scanner`` — WARNING/ERROR for the watchdog's
"parse appears stuck" trail and the recorded parse-deadline timeout.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property

from .models import SasChunkKind, SasDiagnostic, SasDiagnosticSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal parse primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    text: str
    is_comment: bool = False
    terminated: bool = True
    unclosed_comment: bool = False

    def __str__(self) -> str:
        # start/end are character offsets into the source; show a short
        # single-line preview rather than the full (possibly long) text.
        preview = " ".join(self.text.split())
        if len(preview) > 60:
            preview = preview[:57] + "..."
        flags = [
            name
            for name, on in (
                ("comment", self.is_comment),
                ("unterminated", not self.terminated),
                ("unclosed-comment", self.unclosed_comment),
            )
            if on
        ]
        tag = f" ({', '.join(flags)})" if flags else ""
        return f"_Unit chars {self.start}-{self.end}{tag}: {preview!r}"


@dataclass(frozen=True)
class _Region:
    kind: SasChunkKind
    start: int
    end: int
    units: list[_Unit]
    unclosed: bool = False

    def __str__(self) -> str:
        unclosed = " unclosed" if self.unclosed else ""
        return (
            f"_Region {self.kind.value} chars {self.start}-{self.end} "
            f"units={len(self.units)}{unclosed}"
        )

    @cached_property
    def text(self) -> str:
        # Cached: a region's units are fixed at construction, and `.text` is
        # read several times per region (word count, metadata, chunk build).
        # cached_property writes through the instance __dict__, which a frozen
        # (but unslotted) dataclass still allows.
        return "".join(u.text for u in self.units)


# ---------------------------------------------------------------------------
# Stuck-parser protection — deadline + watchdog
#
# See the module docstring's "Stuck-parser protection" section for the split
# of responsibilities: the deadline gives a *graceful partial exit* from the
# Python-level scan/group loops, while the watchdog guarantees a *log trail*
# for the one case the deadline cannot cover (a hang inside a single
# un-interruptible C-level regex call).
# ---------------------------------------------------------------------------


class _Deadline:
    """Monotonic wall-clock budget for a single parse.

    A ``timeout`` of ``None`` means unbounded — :meth:`expired` then always
    reports ``False`` and costs nothing.  :meth:`expired` reads the monotonic
    clock, so call sites in the hot scan/group loops gate it behind a periodic
    tick counter rather than hitting it on every character/unit.
    """

    __slots__ = ("_deadline",)

    def __init__(self, timeout: float | None) -> None:
        self._deadline = None if timeout is None else time.perf_counter() + timeout

    def expired(self) -> bool:
        return self._deadline is not None and time.perf_counter() >= self._deadline


class _ParseWatchdog:
    """Background timer that logs a trail when a parse appears stuck.

    Used as a context manager around the whole parse.  When ``timeout`` is
    ``None`` it is inert (no thread is started and :meth:`set_phase` is a
    no-op).  Otherwise a daemon thread wakes every ``timeout`` seconds and,
    while the parse is still running, logs which phase it was last in and how
    long it has taken — escalating from WARNING to ERROR after repeated
    strikes.  It never interrupts the parse (Python cannot preempt a C-level
    regex call); its sole job is to make a wedged parse diagnosable from the
    logs.  On a clean/graceful finish it is stopped and stays silent.
    """

    def __init__(self, timeout: float | None, label: str) -> None:
        self._timeout = timeout
        self._label = label
        self._phase = "starting"
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._start = 0.0
        self._thread: threading.Thread | None = None
        if timeout is not None:
            self._thread = threading.Thread(
                target=self._run, name="chunker-watchdog", daemon=True
            )

    def __enter__(self) -> _ParseWatchdog:
        if self._thread is not None:
            self._start = time.perf_counter()
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    def set_phase(self, phase: str) -> None:
        if self._thread is None:
            return
        with self._lock:
            self._phase = phase

    def _run(self) -> None:
        assert self._timeout is not None
        strikes = 0
        while not self._done.wait(self._timeout):
            strikes += 1
            elapsed = time.perf_counter() - self._start
            with self._lock:
                phase = self._phase
            level = logging.ERROR if strikes >= 3 else logging.WARNING
            logger.log(
                level,
                f"parse watchdog: '{self._label}' still running after "
                f"{elapsed:.1f}s (timeout={self._timeout:.1f}s) — appears stuck "
                f"in phase '{phase}'; graceful exit will occur at the next "
                f"statement boundary if the parser is not wedged in a regex",
            )


def _record_parser_timeout(
    diagnostics: list[SasDiagnostic],
    line_starts: list[int],
    phase: str,
    stop_char: int,
) -> None:
    """Append a single ``PARSER_TIMEOUT`` diagnostic (idempotent across phases).

    Once the deadline expires every subsequent phase sees it as expired and
    bails immediately, so this guards against emitting one diagnostic per
    phase — only the first (where the parser actually got stuck) is recorded.
    """
    if any(d.code == "PARSER_TIMEOUT" for d in diagnostics):
        return
    line = _line_for(stop_char, line_starts)
    logger.error(
        f"parse deadline exceeded during {phase} phase near line {line}; "
        f"returning partial result"
    )
    diagnostics.append(
        SasDiagnostic(
            code="PARSER_TIMEOUT",
            message=(
                f"Parsing exceeded its time budget during the {phase} phase; "
                f"output is partial (stopped near line {line})."
            ),
            severity=SasDiagnosticSeverity.ERROR,
            start_line=line,
            end_line=line,
        )
    )


# ---------------------------------------------------------------------------
# Block-boundary classifier
#
# IMPORTANT: only these three kinds open a new top-level block and therefore
# close whichever block is currently being collected.  Everything else
# (FORMAT, OPTIONS, GLOBAL_STATEMENT, etc.) is a *statement inside* the
# current block and must be collected, not treated as a boundary.
# ---------------------------------------------------------------------------

_BLOCK_OPENERS = frozenset(
    {
        SasChunkKind.DATA_STEP,
        SasChunkKind.PROC_STEP,
        SasChunkKind.MACRO_DEFINITION,
    }
)

# Precompiled statement-classifier patterns.  ``_classify`` runs once per unit
# (tens of thousands of times on a large file), so these are compiled once here
# rather than re-parsed from string literals on every call — which otherwise
# dominates via ``re``'s per-call pattern-cache lookup.  ``_norm`` lower-cases
# its result, so the classifier input is always lowercase and no IGNORECASE flag
# is needed (matching a lowercase literal without the flag is faster still).
_CLS_DATA_RE = re.compile(r"data\b")
_CLS_PROC_RE = re.compile(r"proc\b")
_CLS_MACRO_RE = re.compile(r"%\s*macro\b")
_CLS_INCLUDE_RE = re.compile(r"%\s*include\b")
_CLS_MACROVAR_RE = re.compile(r"%\s*(?:let|put|global|local)\b")
_CLS_CTRLFLOW_RE = re.compile(r"%\s*(?:if|else|do|end|return|goto|abort)\b")
_CLS_MACROCALL_RE = re.compile(r"%[A-Za-z_]\w*\b")
_CLS_STEP_RE = re.compile(r"(?:run|quit)\b")
_CLS_OPTIONS_RE = re.compile(r"options\b")
_CLS_GLOBAL_RE = re.compile(r"(?:libname|filename|title\d*|footnote\d*)\b")
_CLS_ODS_RE = re.compile(r"ods\b")
_CLS_FORMAT_RE = re.compile(r"(?:format|informat)\b")
# %MEND terminator check inside _collect_block; input is _norm'd (lowercase).
_MEND_RE = re.compile(r"%\s*mend\b")


def _classify(stripped: str) -> SasChunkKind | None:
    """
    Map a single SAS statement to its :class:`SasChunkKind`.

    Returns ``None`` for unrecognised statements (they accumulate into
    ``UNKNOWN_STATEMENT_GROUP`` / ``UNKNOWN_BLOCK``).

    This function is called both at the top level (to decide *which* block
    type to open) and inside ``_collect_block`` (only to detect the three
    block-opener kinds that close the current block).  All other statement
    types are transparently collected as block body statements.
    """
    return _classify_normed(_norm(stripped))


def _classify_normed(n: str) -> SasChunkKind | None:
    """Classify an already-``_norm``'d (stripped, lowercased) statement.

    Split out from :func:`_classify` so callers that already hold the
    normalised form — notably ``_collect_block``, which needs it for the
    %MEND / RUN / QUIT terminator checks too — can classify without
    re-normalising the same text.
    """
    if not n:
        return None
    if _CLS_DATA_RE.match(n):
        return SasChunkKind.DATA_STEP
    if _CLS_PROC_RE.match(n):
        return SasChunkKind.PROC_STEP
    if _CLS_MACRO_RE.match(n):
        return SasChunkKind.MACRO_DEFINITION
    if _CLS_INCLUDE_RE.match(n):
        return SasChunkKind.INCLUDE
    if _CLS_MACROVAR_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_CTRLFLOW_RE.match(n):
        return SasChunkKind.MACRO_CONTROL_FLOW
    if _CLS_MACROCALL_RE.match(n):
        return SasChunkKind.MACRO_CALL
    # A bare RUN;/QUIT; reached here (rather than inside _collect_block) is a
    # standalone step boundary in open code — e.g. a stray RUN; after a
    # global LIBNAME/TITLE statement.  Recognise it so it doesn't fall
    # through to UNKNOWN_STATEMENT_GROUP and raise a spurious
    # UNRECOGNIZED_SOURCE_REGION diagnostic.  RUN CANCEL; is included via the
    # trailing \b (the optional CANCEL keyword follows).  Inside a DATA/PROC
    # block this same statement still terminates the block (handled directly
    # in _collect_block) and never reaches open code.
    if _CLS_STEP_RE.match(n):
        return SasChunkKind.STEP_BOUNDARY
    if _CLS_OPTIONS_RE.match(n):
        return SasChunkKind.OPTIONS
    if _CLS_GLOBAL_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_ODS_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_FORMAT_RE.match(n):
        return SasChunkKind.FORMAT_OR_INFORMAT
    return None


# ---------------------------------------------------------------------------
# Pure helper functions  (no side-effects, no logging)
# ---------------------------------------------------------------------------


def _line_starts(source: str) -> list[int]:
    # Walk newline to newline with str.find (a C-level scan) rather than a
    # per-character Python loop; each match yields the start of the next line.
    starts = [0]
    pos = source.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = source.find("\n", pos + 1)
    return starts


def _line_for(char_index: int, line_starts: list[int]) -> int:
    return bisect_right(line_starts, char_index)


def _ws_end(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _is_stmt_comment(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("*") and not s.startswith("*/")


_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    return s[:-1].strip().lower() if s.endswith(";") else s.lower()


# A block comment (terminated ``/* … */`` or unterminated ``/* …`` to EOF) or a
# quoted string literal (single/double, with the doubled-quote ``''``/``""``
# escape and a possibly-unterminated tail).  The three alternatives start with
# distinct characters, so at any position at most one can match, and ``re``'s
# left-to-right scan reproduces the original hand-written scanner's "first
# delimiter encountered wins" precedence (a quote inside a comment, or ``/*``
# inside a string, is swallowed by whichever region opened first).
_COMMENT_OR_STRING_RE = re.compile(
    r"/\*.*?(?:\*/|\Z)"
    r"|'(?:''|[^'])*(?:'|\Z)"
    r'|"(?:""|[^"])*(?:"|\Z)',
    re.DOTALL,
)
# Comment-only variant used when blank_strings=False: string literals are left
# completely intact and a ``/*`` is a comment even inside apparent quotes,
# matching the original loop's behaviour in that mode.
_COMMENT_ONLY_RE = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)
# Every non-newline character; used to blank a span to spaces while preserving
# line breaks.  CR is normalised to LF afterwards, exactly as the original loop
# mapped both ``\r`` and ``\n`` to ``\n``.
_NON_NEWLINE_RE = re.compile(r"[^\r\n]")


def _blank_span(s: str) -> str:
    """Map every character of *s* to a space, except newlines: ``\\r`` and
    ``\\n`` both become ``\\n`` so downstream line alignment is preserved."""
    return _NON_NEWLINE_RE.sub(" ", s).replace("\r", "\n")


def _sanitise_repl(m: re.Match[str]) -> str:
    s = m.group(0)
    q = s[0]
    if q == "/":
        # block comment — delimiters included, blanked wholesale
        return _blank_span(s)
    # Quoted string — keep the delimiters, blank only the interior.  The span
    # is terminated (a real closing delimiter is present) iff it holds an even
    # number of the quote char: opener (1) + doubled-quote escapes (2 each) +
    # closer (1).  An odd count means the trailing quote is the second half of
    # a ``''`` escape and the literal actually runs to EOF unterminated — in
    # which case only the opening delimiter is kept.
    if s.count(q) % 2 == 0:
        return q + _blank_span(s[1:-1]) + q
    return q + _blank_span(s[1:])


def _sanitise(text: str, *, blank_strings: bool = True) -> str:
    """
    Blank out block comments, preserving newlines so line numbers stay
    aligned with the original source.

    When ``blank_strings`` is True (the default), quoted string literals
    are also blanked out — their delimiters are kept but their contents
    become spaces, with the doubled-quote escape (``''`` / ``""``) handled
    so it doesn't end the string early.  Pass ``blank_strings=False`` to
    keep quoted text intact, e.g. when extracting a literal filename from
    ``%include 'path.sas'``.

    Implemented as a single compiled-regex substitution rather than a
    character-by-character Python loop: the scanning stays in the regex
    engine and only the matched comment/string spans are rewritten, which is
    substantially faster on real source where most characters are neither.
    """
    if blank_strings:
        return _COMMENT_OR_STRING_RE.sub(_sanitise_repl, text)
    return _COMMENT_ONLY_RE.sub(lambda m: _blank_span(m.group(0)), text)
