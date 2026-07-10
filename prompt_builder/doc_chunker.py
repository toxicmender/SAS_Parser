"""Word-budget chunker: DocSection -> InstructionChunk. See prompt_builder/README.md.

Turns the reader's sections into retrieval-ready chunks:

* consecutive undersized sections under the *same parent heading* merge up to
  ``min_words`` (SAS function dictionaries have the odd one-line entry);
* a section over ``max_words`` splits into overlapping windows at paragraph
  boundaries — plain windows, not the chunker's parent/child pair, because the
  LLM only ever sees the retrieved window, never the whole document;
* every chunk's stored text is prefixed with its section breadcrumb, so a
  heading term ("MERGE", "INTNX") weighs on retrieval even when the prose
  below never repeats it.

Mirrors the ``min_words`` / ``max_words`` idiom of ``SasSemanticChunker`` with
budgets tuned for prose rather than SAS source.

Logger name: ``prompt_builder.doc_chunker``.
"""

from __future__ import annotations

import logging
import re

import app_config

from .models import DocRole, DocSection, InstructionChunk

logger = logging.getLogger(__name__)

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _wc(text: str) -> int:
    return len(text.split())


def _parent(section_path: str) -> str:
    """Breadcrumb with the leaf heading removed; ``""`` for a top-level path."""
    return " > ".join(section_path.split(" > ")[:-1])


def _tail_overlap(units: list[str], overlap_words: int) -> list[str]:
    """Trailing units that fit within *overlap_words*; empty if none fit."""
    out: list[str] = []
    words = 0
    for unit in reversed(units):
        uw = _wc(unit)
        if words + uw > overlap_words:
            break
        out.insert(0, unit)
        words += uw
    return out


def _split_overlapping(text: str, max_words: int, overlap_words: int) -> list[str]:
    """
    Split *text* into windows of at most ``max_words`` words at paragraph
    boundaries, each seeded with a trailing overlap from the previous window.
    A single paragraph larger than ``max_words`` is hard-split on word count.
    """
    units: list[str] = []
    for para in (p.strip() for p in _PARA_SPLIT_RE.split(text)):
        if not para:
            continue
        words = para.split()
        if len(words) <= max_words:
            units.append(para)
        else:
            for i in range(0, len(words), max_words):
                units.append(" ".join(words[i : i + max_words]))

    windows: list[str] = []
    current: list[str] = []
    current_wc = 0
    for unit in units:
        uw = _wc(unit)
        if current and current_wc + uw > max_words:
            windows.append("\n\n".join(current))
            current = _tail_overlap(current, overlap_words)
            current_wc = sum(_wc(u) for u in current)
        current.append(unit)
        current_wc += uw
    if current:
        windows.append("\n\n".join(current))
    return windows


class InstructionChunker:
    """
    Chunk reader :class:`DocSection`s into word-budgeted
    :class:`InstructionChunk`s.

    Parameters
    ----------
    min_words : int | None
        Soft lower bound: consecutive sections under the same parent heading
        are merged until their combined text reaches this size. A section that
        already meets it stands alone. ``None`` (default) reads
        ``instruction_chunker.min_words`` from config.json, falling back
        to 120 (see the ``app_config`` package).
    max_words : int | None
        Hard upper bound: a chunk larger than this is split into overlapping
        paragraph windows. ``None`` reads ``instruction_chunker.max_words``,
        falling back to 900.
    overlap_words : int | None
        Target size of the trailing overlap carried into each next window.
        ``None`` reads ``instruction_chunker.overlap_words``, falling back
        to 60.
    """

    def __init__(
        self,
        *,
        min_words: int | None = None,
        max_words: int | None = None,
        overlap_words: int | None = None,
    ) -> None:
        self.min_words = app_config.resolve(
            min_words, "instruction_chunker", "min_words", 120
        )
        self.max_words = app_config.resolve(
            max_words, "instruction_chunker", "max_words", 900
        )
        self.overlap_words = app_config.resolve(
            overlap_words, "instruction_chunker", "overlap_words", 60
        )
        logger.debug(
            f"InstructionChunker  min_words={min_words}  max_words={max_words}  "
            f"overlap_words={overlap_words}"
        )

    def chunk(
        self,
        sections: list[DocSection],
        *,
        role: DocRole = DocRole.SAS_REFERENCE,
    ) -> list[InstructionChunk]:
        """Turn *sections* (in document order) into instruction chunks."""
        chunks: list[InstructionChunk] = []
        buffer: list[DocSection] = []
        buffer_wc = 0

        def flush() -> None:
            nonlocal buffer, buffer_wc
            if buffer:
                self._emit(buffer, role, chunks)
                buffer = []
                buffer_wc = 0

        for section in sections:
            if buffer and (
                _parent(section.section_path) != _parent(buffer[0].section_path)
                or buffer_wc >= self.min_words
            ):
                flush()
            buffer.append(section)
            buffer_wc += _wc(section.text)
        flush()

        logger.info(
            f"InstructionChunker.chunk: {len(sections)} section(s) -> "
            f"{len(chunks)} chunk(s)"
        )
        return chunks

    def _emit(
        self,
        buffer: list[DocSection],
        role: DocRole,
        chunks: list[InstructionChunk],
    ) -> None:
        first = buffer[0]
        # A merged group collapses to the shared parent breadcrumb (the members'
        # own headings survive inline in the body text); a lone section keeps
        # its full path.
        if len(buffer) == 1:
            section_path = first.section_path
        else:
            section_path = _parent(first.section_path) or first.section_path

        body = "\n\n".join(s.text for s in buffer)
        page_start = min(s.page_start for s in buffer)
        page_end = max(s.page_end for s in buffer)

        keys = []
        seen = set()
        for section in buffer:
            key = section.construct_key
            if key is not None and key not in seen:
                seen.add(key)
                keys.append(key)

        if _wc(body) <= self.max_words:
            windows = [body]
        else:
            windows = _split_overlapping(body, self.max_words, self.overlap_words)
            logger.info(
                f"_emit: '{section_path}' split into {len(windows)} window(s) "
                f"({_wc(body)} words > max {self.max_words})"
            )

        for window in windows:
            index = len(chunks)
            chunks.append(
                InstructionChunk(
                    chunk_id=f"{first.doc_id}::c{index:04d}",
                    doc_id=first.doc_id,
                    section_path=section_path,
                    # Breadcrumb prefixed onto the body so heading terms weigh on
                    # retrieval even when the prose never repeats them.
                    text=f"{section_path}\n\n{window}",
                    page_start=page_start,
                    page_end=page_end,
                    role=role,
                    construct_keys=list(keys),
                )
            )
