"""Token-budget chunker: DocSection -> InstructionChunk. See prompt_builder/README.md.

Turns the reader's sections into retrieval-ready chunks:

* consecutive undersized sections under the *same parent heading* merge up to
  ``min_tokens`` (SAS function dictionaries have the odd one-line entry);
* a section over ``max_tokens`` splits into overlapping windows at paragraph
  boundaries — plain windows, not the chunker's parent/child pair, because the
  LLM only ever sees the retrieved window, never the whole document;
* every chunk's stored text is prefixed with its section breadcrumb, so a
  heading term ("MERGE", "INTNX") weighs on retrieval even when the prose
  below never repeats it;
* each chunk carries its own ``token_count``, so the selector never has to
  re-tokenise the corpus to fill a budget.

Budgets are **tokens**, not words — the same currency the prompt is priced in,
and not interchangeable with words here: over the bundled corpus the ratio runs
1.01 to 4.41 depending on how much of a section is code. ``SasSemanticChunker``
still sizes SAS *source* in words, deliberately: that is a semantic-unit
question, not a prompt-cost one.

Logger name: ``prompt_builder.doc_chunker``.
"""

from __future__ import annotations

import logging
import re

import app_config
import token_budget

from .models import DocRole, DocSection, InstructionChunk

logger = logging.getLogger(__name__)

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _tc(text: str) -> int:
    """Tokens in *text* — the currency the prompt budget is actually spent in.

    Words were a stand-in, and a poor one for this corpus: measured over the
    bundled reference set the tokens-per-word ratio runs from 1.01 to 4.41
    (median 1.44), because a page of SQL tokenises far denser than a page of
    prose. Sizing windows in words therefore made code-heavy sections quietly
    larger than the budget believed.
    """
    return token_budget.count_text(text)


def _parent(section_path: str) -> str:
    """Breadcrumb with the leaf heading removed; ``""`` for a top-level path."""
    return " > ".join(section_path.split(" > ")[:-1])


def _tail_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    """Trailing units that fit within *overlap_tokens*; empty if none fit."""
    out: list[str] = []
    total = 0
    for unit in reversed(units):
        ut = _tc(unit)
        if total + ut > overlap_tokens:
            break
        out.insert(0, unit)
        total += ut
    return out


def _hard_split(para: str, max_tokens: int) -> list[str]:
    """Cut an over-long paragraph into pieces of about *max_tokens* tokens.

    Splitting still happens on word boundaries — a token boundary can fall
    mid-word and would produce unreadable fragments — but the *step* is sized
    from this paragraph's own token density, so a dense block of SQL is cut
    into more, shorter pieces than the same word count of prose would be.
    """
    words = para.split()
    if not words:
        return []
    per_word = _tc(para) / len(words)
    step = max(1, int(max_tokens / per_word))
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def _split_overlapping(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split *text* into windows of at most ``max_tokens`` tokens at paragraph
    boundaries, each seeded with a trailing overlap from the previous window.
    A single paragraph over the budget is hard-split (see :func:`_hard_split`).
    """
    units: list[str] = []
    for para in (p.strip() for p in _PARA_SPLIT_RE.split(text)):
        if not para:
            continue
        if _tc(para) <= max_tokens:
            units.append(para)
        else:
            units.extend(_hard_split(para, max_tokens))

    windows: list[str] = []
    current: list[str] = []
    current_tc = 0
    for unit in units:
        ut = _tc(unit)
        if current and current_tc + ut > max_tokens:
            windows.append("\n\n".join(current))
            current = _tail_overlap(current, overlap_tokens)
            current_tc = sum(_tc(u) for u in current)
        current.append(unit)
        current_tc += ut
    if current:
        windows.append("\n\n".join(current))
    return windows


class InstructionChunker:
    """
    Chunk reader :class:`DocSection`s into word-budgeted
    :class:`InstructionChunk`s.

    Parameters
    ----------
    min_tokens : int | None
        Soft lower bound: consecutive sections under the same parent heading
        are merged until their combined text reaches this size. A section that
        already meets it stands alone. ``None`` (default) reads
        ``instruction_chunker.min_tokens`` from config.json, falling back
        to 175 (see the ``app_config`` package).
    max_tokens : int | None
        Hard upper bound: a chunk larger than this is split into overlapping
        paragraph windows. ``None`` reads ``instruction_chunker.max_tokens``,
        falling back to 1300.
    overlap_tokens : int | None
        Target size of the trailing overlap carried into each next window.
        ``None`` reads ``instruction_chunker.overlap_tokens``, falling back
        to 90.
    """

    def __init__(
        self,
        *,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        self.min_tokens = app_config.resolve(
            min_tokens, "instruction_chunker", "min_tokens", 175
        )
        self.max_tokens = app_config.resolve(
            max_tokens, "instruction_chunker", "max_tokens", 1300
        )
        self.overlap_tokens = app_config.resolve(
            overlap_tokens, "instruction_chunker", "overlap_tokens", 90
        )
        logger.debug(
            f"InstructionChunker  min_tokens={min_tokens}  "
            f"max_tokens={max_tokens}  overlap_tokens={overlap_tokens}"
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
        buffer_tc = 0

        def flush() -> None:
            nonlocal buffer, buffer_tc
            if buffer:
                self._emit(buffer, role, chunks)
                buffer = []
                buffer_tc = 0

        for section in sections:
            if buffer and (
                _parent(section.section_path) != _parent(buffer[0].section_path)
                or buffer_tc >= self.min_tokens
            ):
                flush()
            buffer.append(section)
            buffer_tc += _tc(section.text)
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

        if _tc(body) <= self.max_tokens:
            windows = [body]
        else:
            windows = _split_overlapping(body, self.max_tokens, self.overlap_tokens)
            logger.info(
                f"_emit: '{section_path}' split into {len(windows)} window(s) "
                f"({_tc(body)} tokens > max {self.max_tokens})"
            )

        for window in windows:
            index = len(chunks)
            # Breadcrumb prefixed onto the body so heading terms weigh on
            # retrieval even when the prose never repeats them.
            text = f"{section_path}\n\n{window}"
            chunks.append(
                InstructionChunk(
                    chunk_id=f"{first.doc_id}::c{index:04d}",
                    doc_id=first.doc_id,
                    section_path=section_path,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    role=role,
                    construct_keys=list(keys),
                    # Counted here, once, and cached on disk with the chunk —
                    # the selector fills its budget without re-tokenising.
                    token_count=_tc(text),
                )
            )
