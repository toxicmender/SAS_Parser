"""Supplementary construct detection for complexity analysis.
See complexity/README.md.

:class:`~chunker.models.SasChunkMetadata` already reports PROCs, functions,
CALL routines, component objects, macro definitions/calls, and the macro-level
control-flow hazards. It does **not** report the DATA step's own imperative
constructs — ``ARRAY``, ``DO`` loops, ``MERGE``, ``RETAIN``, BY-group
``FIRST.``/``LAST.`` flags — or the ``FILENAME`` access methods (SFTP, EMAIL,
URL, PIPE) that make a step reach outside SAS. Those are exactly the signals
the complexity brief turns on, so this module scans for them directly.

Every scan runs on text sanitised by :func:`chunker.scanner._sanitise`, which
blanks block comments and quoted-string interiors while preserving offsets — so
a construct named inside a comment or a string literal can never fire a signal.
Reusing the chunker's sanitiser (rather than re-implementing SAS comment and
quote rules, including the doubled-quote escape) is a deliberate internal
dependency on a sibling package.

Patterns follow the conventions in ``chunker/metadata.py``: precompiled, gated
behind a cheap lowercase substring test, and written so a macro-level construct
(``%DO``) can never be mistaken for its DATA step namesake (``DO``).

Logger name: ``complexity.detectors``.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

from chunker.scanner import _sanitise

logger = logging.getLogger(__name__)


class DetectedConstruct(NamedTuple):
    """One construct found by a scan.

    ``name`` is the catalogue key looked up in
    :data:`complexity.rules.DETECTOR_RULES`; ``evidence`` is a short snippet of
    the matching source, for the signal's human-readable note.
    """

    name: str
    evidence: str


def _snippet(text: str, limit: int = 60) -> str:
    """Collapse *text* to a single trimmed line of at most *limit* chars."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Pattern catalogue
#
# Each entry is (construct_name, gate_keyword, compiled_pattern). *gate_keyword*
# is a lowercase literal that must appear in the sanitised text for the pattern
# to have any chance of matching, so the regex is skipped entirely otherwise
# (the same necessary-condition trick chunker/metadata.py uses).
# ---------------------------------------------------------------------------

# DATA step ARRAY declaration. The array's own name is captured for evidence.
_ARRAY_RE = re.compile(r"\barray\s+([A-Za-z_]\w*)", re.IGNORECASE)

# Iterative DO: ``do i = 1 to 10;``. The negative lookbehind keeps the macro
# language's ``%DO`` out — that is macro-level code generation, already
# classified through the MACRO_CONTROL_FLOW chunk kind, not a row-wise loop.
_DO_LOOP_RE = re.compile(
    r"(?<!%)\bdo\s+([A-Za-z_]\w*)\s*=",
    re.IGNORECASE,
)
# Conditional DO forms. ``%DO %WHILE`` / ``%DO %UNTIL`` carry their own ``%`` on
# both tokens, so the lookbehind on each keyword is sufficient.
_DO_WHILE_RE = re.compile(r"(?<!%)\bdo\s+(?<!%)while\s*\(", re.IGNORECASE)
_DO_UNTIL_RE = re.compile(r"(?<!%)\bdo\s+(?<!%)until\s*\(", re.IGNORECASE)

# Dataset-combining statements whose semantics differ from a Spark join.
_MERGE_RE = re.compile(r"\bmerge\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_MODIFY_RE = re.compile(r"\bmodify\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)
_UPDATE_RE = re.compile(r"\bupdate\s+([A-Za-z_&][\w.&]*)", re.IGNORECASE)

# Cross-row state: RETAIN holds a value into the next iteration, and the
# BY-group FIRST./LAST. flags depend on the observation's position in its group.
_RETAIN_RE = re.compile(r"\bretain\b", re.IGNORECASE)
_FIRST_LAST_RE = re.compile(r"\b(first|last)\.([A-Za-z_]\w*)", re.IGNORECASE)

# FILENAME access methods — the device-type keyword that follows the fileref.
# One pattern captures the method; _FILENAME_METHODS maps it to a catalogue key
# so adding a method is a one-line change in that dict.
_FILENAME_METHOD_RE = re.compile(
    r"\bfilename\s+[A-Za-z_]\w*\s+(sftp|ftp|email|emailsys|url|pipe|socket)\b",
    re.IGNORECASE,
)
_FILENAME_METHODS: dict[str, str] = {
    "sftp": "filename_sftp",
    "ftp": "filename_ftp",
    "email": "filename_email",
    "emailsys": "filename_email",
    "url": "filename_url",
    "pipe": "filename_pipe",
    "socket": "filename_socket",
}

# Raw external-file I/O inside a DATA step. FILE is matched only in its
# statement forms — ``file print;``, ``file log;``, ``file "path";``,
# ``file myref;`` — because a bare ``file`` token is far too common to trust.
_INFILE_RE = re.compile(r"\binfile\s+([A-Za-z_'\"][\w.'\"/-]*)", re.IGNORECASE)
_FILE_OUT_RE = re.compile(
    r"\bfile\s+(?:(print|log|_webout)\b|(['\"])|([A-Za-z_]\w*)\s*;)",
    re.IGNORECASE,
)

# Procedural jumps inside a DATA step. ``%GOTO`` is macro-level (excluded by the
# lookbehind); LINK calls a labelled subroutine and returns.
_LINK_RE = re.compile(r"\blink\s+([A-Za-z_]\w*)", re.IGNORECASE)
_DATA_GOTO_RE = re.compile(
    r"(?<!%)\b(?:goto|go\s+to)\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)


def _detect_simple(
    name: str, pattern: re.Pattern[str], text: str, label: str
) -> list[DetectedConstruct]:
    """One construct per distinct match of *pattern*, deduplicated by evidence."""
    found: list[DetectedConstruct] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        evidence = f"{label} {_snippet(m.group(0))}".strip()
        if evidence not in seen:
            seen.add(evidence)
            found.append(DetectedConstruct(name, evidence))
    return found


def detect_constructs(text: str) -> list[DetectedConstruct]:
    """Every supplementary construct found in SAS source *text*.

    *text* is the chunk's raw source; it is sanitised here, so callers pass the
    original and never a pre-stripped form. Results are deduplicated per
    construct/evidence pair and returned in scan order.
    """
    mt = _sanitise(text)
    low = mt.lower()
    found: list[DetectedConstruct] = []

    if "array" in low:
        found += _detect_simple("array", _ARRAY_RE, mt, "ARRAY")

    if "do" in low:
        found += _detect_simple("do_loop", _DO_LOOP_RE, mt, "iterative")
        found += _detect_simple("do_while", _DO_WHILE_RE, mt, "")
        found += _detect_simple("do_until", _DO_UNTIL_RE, mt, "")

    if "merge" in low:
        found += _detect_simple("merge", _MERGE_RE, mt, "")
    if "modify" in low:
        found += _detect_simple("modify", _MODIFY_RE, mt, "")
    if "update" in low:
        found += _detect_simple("update", _UPDATE_RE, mt, "")

    if "retain" in low:
        found += _detect_simple("retain", _RETAIN_RE, mt, "")
    if "first." in low or "last." in low:
        found += _detect_simple("by_group_first_last", _FIRST_LAST_RE, mt, "")

    if "filename" in low:
        seen_methods: set[str] = set()
        for m in _FILENAME_METHOD_RE.finditer(mt):
            method = m.group(1).lower()
            name = _FILENAME_METHODS[method]
            if name not in seen_methods:
                seen_methods.add(name)
                found.append(
                    DetectedConstruct(name, f"FILENAME ... {method.upper()}")
                )

    if "infile" in low:
        found += _detect_simple("infile", _INFILE_RE, mt, "")
    if "file" in low:
        found += _detect_simple("file_output", _FILE_OUT_RE, mt, "")

    if "link" in low:
        found += _detect_simple("link_return", _LINK_RE, mt, "")
    if "goto" in low or "go to" in low:
        found += _detect_simple("data_goto", _DATA_GOTO_RE, mt, "")

    if found and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"detect_constructs: found {len(found)} construct(s): "
            f"{sorted({c.name for c in found})}"
        )
    return found
