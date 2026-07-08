"""
chunker.py — dependency-light semantic chunker for Base SAS source files.

Recognised constructs (Base SAS Programming Reference)
-------------------------------------------------------
DATA step, PROC step, %MACRO/%MEND, %INCLUDE, LIBNAME/FILENAME/TITLE/FOOTNOTE,
OPTIONS, ODS, FORMAT/INFORMAT standalone, %LET/%PUT/%GLOBAL/%LOCAL, %macro_call.

All imports are stdlib + pydantic (via local models.py).  This module is the
orchestrator; its supporting layers live beside it in the package:

  keywords.py — SAS keyword catalogues + the patterns compiled from them
  scanner.py  — _Unit/_Region parse primitives, statement classifier,
                sanitiser, and the deadline/watchdog machinery
  metadata.py — per-chunk SasChunkMetadata extraction (_metadata_for,
                _io_for, _macro_body_io, _merge_meta, ...)

Key design rule — block collection
-----------------------------------
FORMAT, LABEL, OPTIONS, LIBNAME, ODS, and other statement keywords that appear
*inside* a DATA or PROC block body are legal SAS statements within that block
and must NOT terminate the block early.  Only a new DATA, PROC, or %MACRO
header, or an explicit RUN/QUIT, closes the current block.

Stuck-parser protection
------------------------
Two independent safeguards keep a pathological input from hanging the parser
forever (see ``SasSemanticChunker(timeout=...)``; both classes live in
scanner.py):

  * A wall-clock **deadline** is checked at statement/region boundaries in the
    scan and grouping loops.  When it is exceeded the parser stops at the next
    boundary, logs where it stopped, emits a ``PARSER_TIMEOUT`` diagnostic and
    returns the *partial* result collected so far — a graceful exit rather than
    an unbounded spin on, say, a multi-megabyte generated file.
  * A background **watchdog** thread logs an escalating trail (WARNING → ERROR)
    naming the phase and elapsed time whenever a parse overruns the timeout.
    This is the only safeguard that fires when the parser is wedged inside a
    single un-interruptible C-level regex call (catastrophic backtracking on
    hostile source): pure-Python code cannot preempt that call, but the
    watchdog still guarantees the stuck phase is named in the logs, so the hang
    is diagnosable even if the process must ultimately be killed.

Logging
-------
Logger: ``chunker.chunker`` (the scanner and metadata layers log under
``chunker.scanner`` / ``chunker.metadata``)

  Level    When emitted
  -------  ---------------------------------------------------------------
  DEBUG    Per-unit / per-region decisions (very verbose; off in prod)
  INFO     File-level start/finish, oversized-split decisions, elapsed time
  WARNING  Unclosed blocks, unterminated statements, unrecognised regions
  ERROR    File-not-found (logged then re-raised); parse-deadline exceeded
           (partial result returned)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .metadata import _merge_meta, _metadata_for, _title
from .models import (
    SasChunk,
    SasChunkKind,
    SasChunkMetadata,
    SasChunkResult,
    SasDiagnostic,
    SasDiagnosticSeverity,
)
from .scanner import (
    _BLOCK_OPENERS,
    _MEND_RE,
    _Deadline,
    _ParseWatchdog,
    _Region,
    _Unit,
    _classify,
    _classify_normed,
    _is_stmt_comment,
    _line_for,
    _line_starts,
    _norm,
    _record_parser_timeout,
    _ws_end,
)

logger = logging.getLogger(__name__)


def _record_user_library(
    chunks: list[SasChunk],
    diagnostics: list[SasDiagnostic],
) -> None:
    """Emit a single ``USER_LIBRARY_ASSIGNED`` diagnostic per parse.

    Per the SAS Programmer's Guide: Essentials (pp. 236, 252-253), assigning
    a USER library — via the ``USER=`` system option or a ``libname user``
    statement — redirects one-level dataset names to that (permanent)
    library instead of the temporary WORK library.  That invalidates the
    ``work.``-canonicalisation :func:`_canon_ds` applies, so the condition
    is surfaced as a diagnostic rather than silently mis-resolved.
    Idempotent across regions, mirroring :func:`_record_parser_timeout`.
    """
    if any(d.code == "USER_LIBRARY_ASSIGNED" for d in diagnostics):
        return
    for chunk in chunks:
        meta = chunk.metadata
        assigns_user = "user" in meta.defines_librefs or (
            chunk.kind == SasChunkKind.OPTIONS
            and any(t == "user" or t.startswith("user=") for t in meta.options)
        )
        if not assigns_user:
            continue
        logger.warning(
            f"_record_user_library: USER library assigned near line "
            f"{chunk.start_line}; one-level dataset names resolve to USER, "
            f"not WORK — canonicalised names may be inaccurate"
        )
        diagnostics.append(
            SasDiagnostic(
                code="USER_LIBRARY_ASSIGNED",
                message=(
                    "A USER library is assigned (USER= system option or "
                    "LIBNAME USER), so one-level dataset names resolve to "
                    "the USER library, not WORK; the work.-canonicalised "
                    "dataset names in this result may be inaccurate."
                ),
                severity=SasDiagnosticSeverity.WARNING,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        )
        return


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class SasSemanticChunker:
    """
    Chunk Base SAS source into source-preserving semantic regions.

    Parameters
    ----------
    min_words : int
        Soft lower bound — chunks smaller than this are never split further.
    max_words : int
        Hard upper bound — regions larger than this are split at statement
        boundaries with a small overlap window.
    timeout : float | None
        Wall-clock budget, in seconds, for a single ``chunk_text`` /
        ``chunk_file`` call.  When the parser overruns it, it stops at the next
        statement/region boundary, logs where it stopped, emits a
        ``PARSER_TIMEOUT`` diagnostic and returns the partial result gathered
        so far (a background watchdog also logs the stuck phase throughout).
        Pass ``None`` to disable both safeguards and parse unbounded.  Defaults
        to 60s, comfortably above any healthy parse.
    """

    def __init__(
        self,
        *,
        min_words: int = 300,
        max_words: int = 700,
        timeout: float | None = 60.0,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.timeout = timeout
        logger.debug(
            f"SasSemanticChunker  min_words={min_words}  max_words={max_words}  "
            f"timeout={timeout}"
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def chunk_file(self, path: str) -> SasChunkResult:
        """Read *path* from disk and return a :class:`SasChunkResult`."""
        fp = Path(path)
        logger.info(f"chunk_file: reading '{fp}'")
        if not fp.exists():
            logger.error(f"chunk_file: file not found '{fp}'")
            raise FileNotFoundError(fp)
        source = fp.read_text(encoding="utf-8")
        line_count = source.count("\n") + 1
        logger.debug(f"chunk_file: {len(source)} bytes / {line_count} lines  '{fp}'")
        return self.chunk_text(source, source_id=str(fp))

    def chunk_text(
        self, source: str, *, source_id: str | None = None
    ) -> SasChunkResult:
        """Parse *source* string and return a :class:`SasChunkResult`."""
        label = source_id or "<inline>"
        line_count = source.count("\n") + 1
        logger.info(
            f"chunk_text: start  source='{label}'  chars={len(source)}  lines={line_count}"
        )
        t0 = time.perf_counter()

        line_starts = _line_starts(source)
        diagnostics: list[SasDiagnostic] = []
        deadline = _Deadline(self.timeout)

        with _ParseWatchdog(self.timeout, label) as watchdog:
            watchdog.set_phase("scan")
            units = self._scan_units(source, line_starts, diagnostics, deadline)
            logger.debug(f"chunk_text: scan → {len(units)} units")

            watchdog.set_phase("region grouping")
            regions = self._group_regions(units, line_starts, diagnostics, deadline)
            logger.debug(f"chunk_text: group → {len(regions)} regions")

            watchdog.set_phase("chunk building")
            chunks: list[SasChunk] = []
            for region in regions:
                # Deadline check between regions: a graceful exit here keeps the
                # chunks already built (each region is self-contained) and stops
                # before spending more of the budget on the remaining regions.
                if deadline.expired():
                    _record_parser_timeout(
                        diagnostics, line_starts, "chunk building", region.start
                    )
                    logger.warning(
                        f"chunk_text: deadline exceeded before region at "
                        f"line {_line_for(region.start, line_starts)}; "
                        f"stopping with {len(chunks)} chunk(s) built"
                    )
                    break
                chunks.extend(
                    self._chunks_for_region(
                        source,
                        source_id,
                        region,
                        line_starts,
                        len(chunks),
                        diagnostics,
                    )
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"chunk_text: done  source='{label}'  chunks={len(chunks)}  diagnostics={len(diagnostics)}  elapsed={elapsed:.3f}s"
        )
        return SasChunkResult(
            source_id=source_id, chunks=chunks, diagnostics=diagnostics
        )

    # ------------------------------------------------------------------
    # Phase 1 — scan source → _Unit list
    # ------------------------------------------------------------------

    def _scan_units(
        self,
        source: str,
        line_starts: list[int],
        diagnostics: list[SasDiagnostic],
        deadline: _Deadline,
    ) -> list[_Unit]:
        logger.debug(f"_scan_units: {len(source)} chars")
        units: list[_Unit] = []
        stmt_start: int | None = None
        index = 0
        quote: str | None = None
        ticks = 0

        while index < len(source):
            # Deadline check, gated behind a tick counter so perf_counter() is
            # sampled ~once per 8192 iterations, not on every character (this is
            # the module's hottest loop).  On expiry: keep the units gathered so
            # far and return them for grouping — a graceful, partial exit.
            ticks += 1
            if (ticks & 0x1FFF) == 0 and deadline.expired():
                _record_parser_timeout(diagnostics, line_starts, "scan", index)
                logger.warning(
                    f"_scan_units: deadline exceeded at char {index}/{len(source)} "
                    f"(line {_line_for(index, line_starts)}); returning "
                    f"{len(units)} partial unit(s)"
                )
                break

            if stmt_start is None:
                stmt_start = index

            char = source[index]
            nxt = source[index : index + 2]

            # ── inside quoted string ────────────────────────────────────────
            if quote:
                if char == quote:
                    if index + 1 < len(source) and source[index + 1] == quote:
                        index += 2  # doubled-quote escape
                        continue
                    quote = None
                index += 1
                continue

            if char in {"'", '"'}:
                quote = char
                index += 1
                continue

            # ── block comment /* … */ ───────────────────────────────────────
            if nxt == "/*":
                comment_start = index
                comment_end = source.find("*/", index + 2)
                before = source[stmt_start:comment_start]
                if not before.strip():
                    # standalone comment (no code before it on this statement)
                    end = (
                        len(source)
                        if comment_end == -1
                        else _ws_end(source, comment_end + 2)
                    )
                    unit = _Unit(
                        start=stmt_start,
                        end=end,
                        text=source[stmt_start:end],
                        is_comment=True,
                        terminated=(comment_end != -1),
                        unclosed_comment=(comment_end == -1),
                    )
                    units.append(unit)
                    if comment_end == -1:
                        logger.warning(
                            f"_scan_units: unclosed block comment at line {_line_for(stmt_start, line_starts)}"
                        )
                        diagnostics.append(
                            _diag(
                                "UNCLOSED_BLOCK_COMMENT",
                                "Unclosed block comment.",
                                line_starts,
                                unit,
                            )
                        )
                    elif logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_scan_units: block comment  line={_line_for(stmt_start, line_starts)}"
                        )
                    stmt_start = None
                    index = end
                    continue
                # inline comment — skip past it without ending the statement
                index = len(source) if comment_end == -1 else comment_end + 2
                continue

            # ── statement terminator ────────────────────────────────────────
            if char == ";":
                end = _ws_end(source, index + 1)
                text = source[stmt_start:end]
                is_comment = _is_stmt_comment(text)
                units.append(
                    _Unit(
                        start=stmt_start,
                        end=end,
                        text=text,
                        is_comment=is_comment,
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    text_preview = text[:60].replace("\n", "↵")
                    logger.debug(
                        f"_scan_units: stmt  line={_line_for(stmt_start, line_starts)}  "
                        f"comment={is_comment}  text={text_preview!r}"
                    )
                stmt_start = None
                index = end
                continue

            index += 1

        # trailing unterminated fragment
        if stmt_start is not None and stmt_start < len(source):
            text = source[stmt_start:]
            if text.strip():
                line = _line_for(stmt_start, line_starts)
                logger.warning(f"_scan_units: unterminated statement at line {line}")
                units.append(
                    _Unit(
                        start=stmt_start,
                        end=len(source),
                        text=text,
                        is_comment=_is_stmt_comment(text),
                        terminated=False,
                    )
                )

        non_empty = [u for u in units if u.text]
        logger.debug(
            f"_scan_units: done  total={len(units)}  non_empty={len(non_empty)}"
        )
        return non_empty

    # ------------------------------------------------------------------
    # Phase 2 — _Unit list → _Region list
    # ------------------------------------------------------------------

    def _group_regions(
        self,
        units: list[_Unit],
        line_starts: list[int],
        diagnostics: list[SasDiagnostic],
        deadline: _Deadline,
    ) -> list[_Region]:
        logger.debug(f"_group_regions: {len(units)} units")
        regions: list[_Region] = []
        unknown: list[_Unit] = []
        index = 0
        ticks = 0

        def flush_unknown() -> None:
            if not unknown:
                return
            kind = (
                SasChunkKind.UNKNOWN_BLOCK
                if any(not u.terminated for u in unknown)
                else SasChunkKind.UNKNOWN_STATEMENT_GROUP
            )
            r = _Region(
                kind,
                unknown[0].start,
                unknown[-1].end,
                list(unknown),
                unclosed=(kind == SasChunkKind.UNKNOWN_BLOCK),
            )
            regions.append(r)
            sl = _line_for(r.start, line_starts)
            el = _line_for(max(r.end - 1, r.start), line_starts)
            logger.warning(
                f"_group_regions: unrecognised  kind={kind.value}  lines={sl}-{el}"
            )
            diagnostics.append(
                SasDiagnostic(
                    code="UNRECOGNIZED_SOURCE_REGION",
                    message="Source region did not match a known Base SAS semantic unit.",
                    start_line=sl,
                    end_line=el,
                )
            )
            unknown.clear()

        while index < len(units):
            # Deadline check (tick-gated, as in _scan_units).  On expiry, flush
            # any pending unknown run and return the regions built so far so the
            # chunk-building phase still gets a well-formed, if partial, list.
            ticks += 1
            if (ticks & 0x0FFF) == 0 and deadline.expired():
                _record_parser_timeout(
                    diagnostics, line_starts, "region grouping", units[index].start
                )
                logger.warning(
                    f"_group_regions: deadline exceeded at unit {index}/{len(units)} "
                    f"(line {_line_for(units[index].start, line_starts)}); "
                    f"returning {len(regions)} partial region(s)"
                )
                break

            unit = units[index]
            stripped = unit.text.strip()

            if not stripped:
                unknown.append(unit)
                index += 1
                continue

            if unit.is_comment:
                flush_unknown()
                regions.append(
                    _Region(
                        SasChunkKind.COMMENT_BLOCK,
                        unit.start,
                        unit.end,
                        [unit],
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_group_regions: COMMENT_BLOCK  line={_line_for(unit.start, line_starts)}"
                    )
                index += 1
                continue

            cls = _classify(stripped)

            # ── block-opener: collect the whole block ───────────────────────
            if cls in _BLOCK_OPENERS:
                flush_unknown()
                block_units, index, unclosed = self._collect_block(units, index, cls)
                r = _Region(
                    cls,
                    block_units[0].start,
                    block_units[-1].end,
                    block_units,
                    unclosed=unclosed,
                )
                regions.append(r)
                sl = _line_for(r.start, line_starts)
                el = _line_for(max(r.end - 1, r.start), line_starts)
                if unclosed:
                    code = {
                        SasChunkKind.DATA_STEP: "UNCLOSED_DATA_OR_PROC_STEP",
                        SasChunkKind.PROC_STEP: "UNCLOSED_DATA_OR_PROC_STEP",
                        SasChunkKind.MACRO_DEFINITION: "UNCLOSED_MACRO",
                    }[cls]
                    logger.warning(
                        f"_group_regions: unclosed {cls.value}  lines={sl}-{el}  code={code}"
                    )
                    diagnostics.append(
                        SasDiagnostic(
                            code=code,
                            message=f"{cls.value} was not closed before end of file.",
                            start_line=sl,
                            end_line=el,
                        )
                    )
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_group_regions: {cls.value}  lines={sl}-{el}  units={len(block_units)}"
                        )
                continue

            # ── single-statement kinds ──────────────────────────────────────
            if cls is not None:
                flush_unknown()
                regions.append(
                    _Region(
                        cls,
                        unit.start,
                        unit.end,
                        [unit],
                        unclosed=(not unit.terminated),
                    )
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_group_regions: {cls.value}  line={_line_for(unit.start, line_starts)}"
                    )
                index += 1
                continue

            unknown.append(unit)
            index += 1

        flush_unknown()
        logger.debug(f"_group_regions: → {len(regions)} regions")
        return regions

    def _collect_block(
        self,
        units: list[_Unit],
        start: int,
        kind: SasChunkKind,
    ) -> tuple[list[_Unit], int, bool]:
        """
        Collect all _Units belonging to a DATA / PROC / %MACRO block.

        A block ends when one of the following is encountered:
        - An explicit ``RUN;`` or ``QUIT;`` statement     (DATA / PROC)
        - The matching ``%MEND;`` statement                (%MACRO)
        - A new DATA, PROC, or %MACRO header              (implicit close)
        - End of file                                     (unclosed block)

        Nested ``%MACRO`` definitions are balanced: a macro body may contain
        inner ``%MACRO``/``%MEND`` pairs, so the block is closed only by the
        ``%MEND`` that matches *this* macro's own header — inner ``%MEND``
        statements pop the nesting depth without ending the outer block.

        Critically, FORMAT, LABEL, OPTIONS, LIBNAME, ODS, TITLE, and all
        other statement types are treated as ordinary body statements and
        collected without closing the block.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"_collect_block: {kind.value}  start_unit={start}")
        block: list[_Unit] = []
        index = start
        # Nesting depth of *inner* %MACRO definitions currently open inside this
        # macro body.  Only meaningful when kind is MACRO_DEFINITION; each inner
        # %MACRO header increments it and each %MEND decrements it, so the block
        # closes on the %MEND that brings the depth back below zero — i.e. the
        # one matching this macro's own header.
        macro_depth = 0

        while index < len(units):
            unit = units[index]

            # Comments inside a block are just collected — they don't close it
            if unit.is_comment:
                block.append(unit)
                index += 1
                continue

            # Normalise once and reuse for both classification and the
            # %MEND / RUN / QUIT terminator checks below.
            lowered = _norm(unit.text)
            cls = _classify_normed(lowered)

            # ── only DATA/PROC openers close a DATA/PROC block (implicit).
            # A %MACRO block is closed *only* by %MEND, so DATA and PROC
            # steps that appear inside a macro body are collected, not
            # treated as block boundaries.
            implicit_close = (
                cls in _BLOCK_OPENERS
                and block
                and kind in {SasChunkKind.DATA_STEP, SasChunkKind.PROC_STEP}
            )
            if implicit_close:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_collect_block: implicit close  {kind.value} at unit {index}  next_kind={cls.value}"
                    )
                return block, index, True

            # A nested %MACRO header inside our macro body opens an inner
            # definition that must be balanced by its own %MEND before ours can
            # close us.  The first unit (index == start) is *our* header, not a
            # nested one, so it never bumps the depth.
            if (
                kind == SasChunkKind.MACRO_DEFINITION
                and cls == SasChunkKind.MACRO_DEFINITION
                and index != start
            ):
                macro_depth += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_collect_block: nested %MACRO at unit {index}  depth={macro_depth}"
                    )

            block.append(unit)
            index += 1

            # ── explicit terminators ────────────────────────────────────────
            if kind == SasChunkKind.MACRO_DEFINITION and _MEND_RE.match(lowered):
                if macro_depth > 0:
                    macro_depth -= 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"_collect_block: %MEND closes nested macro  depth={macro_depth}"
                        )
                    continue
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_collect_block: %MEND → closed MACRO_DEFINITION  units={len(block)}"
                    )
                return block, index, False

            if kind in {SasChunkKind.DATA_STEP, SasChunkKind.PROC_STEP} and lowered in {
                "run",
                "quit",
            }:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_collect_block: RUN/QUIT → closed {kind.value}  units={len(block)}"
                    )
                return block, index, False

        logger.warning(
            f"_collect_block: EOF without closing {kind.value}  units={len(block)}"
        )
        return block, index, True

    # ------------------------------------------------------------------
    # Phase 3 — _Region → SasChunk(s)
    # ------------------------------------------------------------------

    def _chunks_for_region(
        self,
        source: str,
        source_id: str | None,
        region: _Region,
        line_starts: list[int],
        next_index: int,
        diagnostics: list[SasDiagnostic],
    ) -> list[SasChunk]:
        wc = _wc(region.text)
        sl = _line_for(region.start, line_starts)
        el = _line_for(max(region.end - 1, region.start), line_starts)

        if wc <= self.max_words or len(region.units) <= 1:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"_chunks_for_region: single  kind={region.kind.value}  words={wc}  lines={sl}-{el}"
                )
            single = [self._make_chunk(source_id, region, line_starts, next_index)]
            _record_user_library(single, diagnostics)
            return single

        # Oversized — split at statement boundaries with overlap
        logger.info(
            f"_chunks_for_region: oversized {region.kind.value}  words={wc} > max={self.max_words}  lines={sl}-{el}  splitting"
        )
        parent_meta = _metadata_for(region.text, region.kind)
        parent_meta.has_unclosed_block = region.unclosed
        parent_chunk = self._make_chunk(
            source_id,
            region,
            line_starts,
            next_index,
            metadata=parent_meta,
        )
        chunks: list[SasChunk] = [parent_chunk]
        parent_id = parent_chunk.chunk_id
        current: list[_Unit] = []
        # Running word count of the joined `current` text, maintained
        # incrementally so the split loop stays O(units) instead of re-joining
        # and re-splitting the whole accumulated text on every unit.  When two
        # units abut without whitespace (former ends non-space, latter starts
        # non-space) their touching tokens merge into one, so joining adds
        # `_wc(unit.text)` minus one for that merge — mirroring exactly what
        # `_wc("".join(...))` would count.  `cur_tail_nonws` tracks whether the
        # joined text currently ends in a non-space character.
        cur_wc = 0
        cur_tail_nonws = False

        def _wc_and_tail(units_list: list[_Unit]) -> tuple[int, bool]:
            joined = "".join(u.text for u in units_list)
            return _wc(joined), bool(joined) and not joined[-1].isspace()

        for unit in region.units:
            utext = unit.text
            uw = _wc(utext)
            uhead_nonws = bool(utext) and not utext[0].isspace()
            utail_nonws = bool(utext) and not utext[-1].isspace()
            if current:
                merge = 1 if (cur_tail_nonws and uhead_nonws) else 0
                cand_wc = cur_wc + uw - merge
                cand_tail = utail_nonws if utext else cur_tail_nonws
            else:
                cand_wc = uw
                cand_tail = utail_nonws
            if current and cand_wc > self.max_words:
                cr = _Region(
                    region.kind,
                    current[0].start,
                    current[-1].end,
                    list(current),
                    unclosed=region.unclosed,
                )
                child_meta = _merge_meta(
                    parent_meta,
                    _metadata_for(cr.text, region.kind),
                )
                child_meta.has_unclosed_block = region.unclosed
                child = self._make_chunk(
                    source_id,
                    cr,
                    line_starts,
                    next_index + len(chunks),
                    parent_id=parent_id,
                    metadata=child_meta,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"_chunks_for_region: child {child.chunk_id}  parent={parent_id}  lines={child.start_line}-{child.end_line}"
                    )
                chunks.append(child)
                current = _overlap(current) + [unit]
                cur_wc, cur_tail_nonws = _wc_and_tail(current)
            else:
                current = current + [unit]
                cur_wc = cand_wc
                cur_tail_nonws = cand_tail

        if current:
            cr = _Region(
                region.kind,
                current[0].start,
                current[-1].end,
                list(current),
                unclosed=region.unclosed,
            )
            child_meta = _merge_meta(
                parent_meta,
                _metadata_for(cr.text, region.kind),
            )
            child_meta.has_unclosed_block = region.unclosed
            child = self._make_chunk(
                source_id,
                cr,
                line_starts,
                next_index + len(chunks),
                parent_id=parent_id,
                metadata=child_meta,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"_chunks_for_region: final child {child.chunk_id}  parent={parent_id}  lines={child.start_line}-{child.end_line}"
                )
            chunks.append(child)

        still_big = [c for c in chunks if _wc(c.text) > self.max_words * 2]
        if still_big:
            logger.warning(
                f"_chunks_for_region: {len(still_big)} chunk(s) remain oversized after split"
            )
            diagnostics.append(
                SasDiagnostic(
                    code="OVERSIZED_ATOMIC_CHUNK",
                    message="A chunk remained oversized because it could not be split safely.",
                    start_line=sl,
                    end_line=el,
                )
            )

        logger.info(
            f"_chunks_for_region: {region.kind.value} → {len(chunks)} chunks (1 parent + {len(chunks) - 1} children)"
        )
        _record_user_library(chunks, diagnostics)
        return chunks

    def _make_chunk(
        self,
        source_id: str | None,
        region: _Region,
        line_starts: list[int],
        index: int,
        *,
        parent_id: str | None = None,
        metadata: SasChunkMetadata | None = None,
    ) -> SasChunk:
        meta = metadata or _metadata_for(region.text, region.kind)
        meta.has_unclosed_block = region.unclosed
        return SasChunk(
            chunk_id=f"chunk-{index + 1:04d}",
            source_id=source_id,
            text=region.text,
            kind=region.kind,
            title=_title(region.kind, meta),
            start_line=_line_for(region.start, line_starts),
            end_line=_line_for(max(region.end - 1, region.start), line_starts),
            start_char=region.start,
            end_char=region.end,
            parent_id=parent_id,
            metadata=meta,
        )


def _wc(text: str) -> int:
    # str.split() with no argument splits on runs of whitespace and drops empty
    # fields, so its length equals the number of ``\S+`` runs — identical to the
    # previous ``len(re.findall(r"\S+", text))`` but without the regex overhead.
    return len(text.split())


def _overlap(units: list[_Unit]) -> list[_Unit]:
    overlap: list[_Unit] = []
    words = 0
    for unit in reversed(units[-3:]):
        uw = _wc(unit.text)
        if words + uw > 100 and overlap:
            break
        overlap.insert(0, unit)
        words += uw
    return overlap


def _diag(
    code: str,
    msg: str,
    line_starts: list[int],
    target: _Unit | _Region,
) -> SasDiagnostic:
    return SasDiagnostic(
        code=code,
        message=msg,
        severity=SasDiagnosticSeverity.WARNING,
        start_line=_line_for(target.start, line_starts),
        end_line=_line_for(max(target.end - 1, target.start), line_starts),
    )
