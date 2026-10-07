"""Lexical layer of the SAS chunker: parse primitives, statement classifier,
sanitiser, where a macro call ends its statement, and the deadline/watchdog
machinery. See chunker/README.md.

Logger name: ``chunker.scanner``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property

from .keywords import _MACRO_LANGUAGE_WORDS
from .macro_vars import call_spans
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
        # cached_property writes through the instance __dict__, which a frozen
        # (but unslotted) dataclass still allows.
        return "".join(u.text for u in self.units)


# Stuck-parser protection — deadline + watchdog (see chunker/README.md).


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


# Block-boundary classifier. Only these three kinds open a new top-level block
# and close the block currently being collected; everything else is a statement
# inside the current block.
_BLOCK_OPENERS = frozenset(
    {
        SasChunkKind.DATA_STEP,
        SasChunkKind.PROC_STEP,
        SasChunkKind.MACRO_DEFINITION,
    }
)

# Precompiled statement-classifier patterns (``_classify`` runs once per unit).
# Input is always ``_norm``'d (lowercase), so no IGNORECASE flag is needed.
_CLS_DATA_RE = re.compile(r"data\b")
_CLS_PROC_RE = re.compile(r"proc\b")
_CLS_MACRO_RE = re.compile(r"%\s*macro\b")
_CLS_INCLUDE_RE = re.compile(r"%\s*include\b")
_CLS_MACROVAR_RE = re.compile(r"%\s*(?:let|put|global|local)\b")
# Host-command escapes -> GLOBAL_STATEMENT. `X` needs care: it is also one of
# the commonest SAS variable names, so the command form must be followed by its
# quoted argument, and the argument-less form (which opens an interactive
# shell) must be the *whole* statement — `_norm` has already stripped the
# trailing `;`, so `x;` arrives here as bare `x`. That rejects `x = 1` and
# `x + 1` while keeping both documented spellings of the statement.
_CLS_HOSTCMD_RE = re.compile(
    r"(?:x\s*(?:['\"]|$)|systask\b|waitfor\b|%\s*sysexec\b)"
)
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
    # Before the macro-call branch: %SYSEXEC would otherwise be swept up as an
    # ordinary macro invocation and lose its identity.
    if _CLS_HOSTCMD_RE.match(n):
        return SasChunkKind.GLOBAL_STATEMENT
    if _CLS_CTRLFLOW_RE.match(n):
        return SasChunkKind.MACRO_CONTROL_FLOW
    if _CLS_MACROCALL_RE.match(n):
        return SasChunkKind.MACRO_CALL
    # A bare RUN;/QUIT; here is a standalone open-code step boundary (inside a
    # DATA/PROC block it is handled by _collect_block and never reaches here).
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


# A block comment or a quoted string literal (single/double, doubled-quote
# escape, possibly-unterminated tail). The three alternatives start with
# distinct characters, so ``re``'s left-to-right scan gives "first delimiter
# encountered wins" precedence.
_COMMENT_OR_STRING_RE = re.compile(
    r"/\*.*?(?:\*/|\Z)"
    r"|'(?:''|[^'])*(?:'|\Z)"
    r'|"(?:""|[^"])*(?:"|\Z)',
    re.DOTALL,
)
# Comment-only variant (blank_strings=False): string literals stay intact.
_COMMENT_ONLY_RE = re.compile(r"/\*.*?(?:\*/|\Z)", re.DOTALL)
# Every non-newline character; blanks a span to spaces while preserving breaks.
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
    # Quoted string — keep the delimiters, blank the interior. An even count of
    # the quote char means it is terminated (opener + 2/escape + closer); an odd
    # count means it runs unterminated to EOF, so keep only the opener.
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


# ---------------------------------------------------------------------------
# Statement ends that are not semicolons
# ---------------------------------------------------------------------------

# A unit that opens with %name: the only kind a macro call can end early.
_LEADING_MACRO_RE = re.compile(r"\s*%\s*([A-Za-z_]\w*)")


def _opens_statement(mt: str) -> bool:
    """Whether sanitised *mt* — what follows a macro call — begins a statement
    of its own: one :func:`_classify` recognises, or a ``*`` comment."""
    return _is_stmt_comment(mt) or _classify(mt) is not None


def _call_and_comments(unit: _Unit, start: int, after: int, nxt: int) -> list[_Unit]:
    """The call ``unit.text[start:after]``, then each block comment between it
    and the next statement at *nxt* as a comment unit of its own — as the
    scanner makes one when a comment follows a semicolon."""
    text = unit.text
    comments: list[_Unit] = []
    at = after
    while at < nxt and text.startswith("/*", at):
        close = text.find("*/", at + 2)
        if close == -1 or close + 2 > nxt:
            break
        end = _ws_end(text, close + 2)
        comments.append(
            _Unit(
                start=unit.start + at,
                end=unit.start + end,
                text=text[at:end],
                is_comment=True,
            )
        )
        at = end
    if at != nxt:
        # Not only comments in between after all: the gap stays with the call.
        after, comments = nxt, []
    call = _Unit(start=unit.start + start, end=unit.start + after, text=text[start:after])
    return [call, *comments]


def _split_after_calls(unit: _Unit) -> list[_Unit]:
    """*unit*, cut after each macro call that ends before its statement does.

    A macro call needs no semicolon — it ends at the parenthesis closing its
    arguments, or at its name when it has none — but statements are found by
    their semicolons, so in::

        %pull(tbl=a)
        %pull(tbl=b)
        data x; set y; run;

    one unit held both calls and the DATA header, and ``set y;`` was left
    unrecognised. Worse, a call just before ``%MEND;`` or ``RUN;`` hid the
    terminator, leaving its block open to the end of the file.

    A cut follows a call only when what comes next opens a statement of its
    own: another call, a statement :func:`_classify` knows, or a ``*`` comment.
    ``%vname(x) = 1;``, where the call writes part of the statement, stays
    whole, as does a unit that opens with a macro-language word (``%mend m;``,
    ``%symdel x;``, ``%let``): those statements run to their semicolon. So does
    an argument list that never closes. Block comments between a call and the
    next statement become comment units. A call that ends the unit is
    complete, so a file ending in one is not unterminated.

    The pieces are contiguous slices of *unit*: a block collects the same text
    either way, and only where top-level regions begin and end changes.
    """
    m = _LEADING_MACRO_RE.match(unit.text)
    if m is None or m.group(1).lower() in _MACRO_LANGUAGE_WORDS:
        return [unit]
    text = unit.text
    mt = _sanitise(text)
    spans = call_spans(mt)
    pieces: list[_Unit] = []
    cut = 0
    complete = False
    for i, span in enumerate(spans):
        if span.name in _MACRO_LANGUAGE_WORDS or not span.closed:
            break
        nxt = _ws_end(mt, span.end)
        if nxt == len(mt):
            complete = True
            break
        another_call = i + 1 < len(spans) and spans[i + 1].start == nxt
        if not another_call and not _opens_statement(mt[nxt:]):
            break
        pieces += _call_and_comments(unit, cut, _ws_end(text, span.end), nxt)
        cut = nxt
    if not pieces and not complete:
        return [unit]
    rest = text[cut:]
    pieces.append(
        _Unit(
            start=unit.start + cut,
            end=unit.end,
            text=rest,
            is_comment=_is_stmt_comment(rest),
            terminated=unit.terminated or complete,
        )
    )
    return pieces
