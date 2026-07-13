"""PromptBuilder facade: reference PDFs -> a formatted guidance block per item.

See prompt_builder/README.md.

Ties the package together: load + chunk + index the reference corpus once at
construction, then :meth:`build` a Markdown guidance block for one pipeline
item's ``(query, constructs)``. Returns ``None`` when nothing is relevant, so
the caller can omit the block entirely.

The metadata -> ``(query, constructs)`` mapping deliberately lives in the
pipeline, not here, so ``prompt_builder`` imports nothing from ``chunker``.

Logger name: ``prompt_builder.builder``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import app_config

from .catalog import CorpusLoader, DocumentSpec, default_catalog
from .models import ConstructKey, DocRole, InstructionChunk
from .selector import InstructionSelector
from .user_instructions import UserInstructionSet

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Build a per-item instruction block from a reference corpus.

    Construct from an in-memory chunk list, from :class:`DocumentSpec`s, or from
    a reference directory (the last two load + cache via :class:`CorpusLoader`).

    Parameters
    ----------
    chunks : Iterable[InstructionChunk]
        The instruction corpus to retrieve over.
    user_instructions : str | UserInstructionSet | None
        Operator-supplied project rules. A plain string is parsed via
        :meth:`UserInstructionSet.from_text` (see ``user_instructions.py``
        for the heading/directive syntax). Selected user chunks render in
        their own ``## Project instructions`` block above the reference
        guidance and take priority over every reference tier.
    top_k : int | None
        Maximum topical (ranked) chunks per item. ``None`` (default) reads
        ``prompt_builder.top_k`` from config.json, falling back to 6 (see
        the ``app_config`` package).
    max_instruction_words : int | None
        Word budget for the whole guidance block. ``None`` reads
        ``prompt_builder.max_instruction_words``, falling back to 1500.
        Keep this >= the instruction chunker's ``max_words`` so any single
        reference section always fits.
    pinned_sections : Iterable[str]
        Section-path substrings always injected first.
    embeddings, embedding_cache_path, rrf_k, reranker :
        Forwarded to :class:`InstructionSelector` (dense retrieval is off unless
        ``embeddings`` is given).
    heading : str
        Markdown H2 heading for the reference-guidance block.
    project_heading : str
        Markdown H2 heading for the user-instruction block.
    """

    def __init__(
        self,
        chunks: Iterable[InstructionChunk],
        *,
        user_instructions: "str | UserInstructionSet | None" = None,
        top_k: int | None = None,
        max_instruction_words: int | None = None,
        pinned_sections: Iterable[str] = (),
        embeddings: Any | None = None,
        embedding_cache_path: str | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        heading: str = "Relevant migration guidance",
        project_heading: str = "Project instructions",
    ) -> None:
        self.top_k = app_config.resolve(top_k, "prompt_builder", "top_k", 6)
        self.max_instruction_words = app_config.resolve(
            max_instruction_words, "prompt_builder", "max_instruction_words", 1500
        )
        self.heading = heading
        self.project_heading = project_heading
        if isinstance(user_instructions, str):
            user_instructions = UserInstructionSet.from_text(user_instructions)
        self.user_instructions = user_instructions
        # Retained so with_user_instructions can rebuild an equivalent
        # selector over the same reference corpus.
        self._pinned_sections = list(pinned_sections)
        self._embeddings = embeddings
        self._embedding_cache_path = embedding_cache_path
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._selector = InstructionSelector(
            chunks,
            user_instructions=user_instructions,
            embeddings=embeddings,
            embedding_cache_path=embedding_cache_path,
            rrf_k=rrf_k,
            reranker=reranker,
            pinned_sections=pinned_sections,
        )

    def with_user_instructions(
        self, user_instructions: "str | UserInstructionSet | None"
    ) -> "PromptBuilder":
        """
        A new builder over the same reference corpus and settings, with
        *user_instructions* replacing any current set. The selector index is
        rebuilt once; the original builder is untouched.
        """
        return PromptBuilder(
            self._selector.reference_chunks,
            user_instructions=user_instructions,
            top_k=self.top_k,
            max_instruction_words=self.max_instruction_words,
            pinned_sections=self._pinned_sections,
            embeddings=self._embeddings,
            embedding_cache_path=self._embedding_cache_path,
            rrf_k=self._rrf_k,
            reranker=self._reranker,
            heading=self.heading,
            project_heading=self.project_heading,
        )

    @classmethod
    def from_specs(
        cls,
        specs: list[DocumentSpec],
        *,
        loader: CorpusLoader | None = None,
        cache_dir: str | None = None,
        pinned_sections: Iterable[str] = (),
        **kwargs: Any,
    ) -> "PromptBuilder":
        """Load + chunk *specs* (with the on-disk cache), then build."""
        if loader is None:
            loader = CorpusLoader(cache_dir=cache_dir) if cache_dir else CorpusLoader()
        chunks = loader.load(specs)
        # Spec-declared pins plus any passed explicitly.
        pins = list(pinned_sections)
        for spec in specs:
            pins.extend(spec.pinned_sections)
        builder = cls(chunks, pinned_sections=pins, **kwargs)
        # A budget below the chunker's window size silently drops whole
        # construct hits — the known misconfiguration; warn loudly.
        if builder.max_instruction_words < loader.chunker.max_words:
            logger.warning(
                f"from_specs: max_instruction_words="
                f"{builder.max_instruction_words} is below the chunker's "
                f"max_words={loader.chunker.max_words}; single reference "
                f"sections may not fit the budget and will be dropped whole"
            )
        return builder

    @classmethod
    def from_reference_dir(
        cls,
        reference_dir: str = "reference_docs",
        *,
        include_unknown: bool = False,
        loader: CorpusLoader | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "PromptBuilder":
        """
        Build from the default catalog of PDFs present under *reference_dir*.
        ``include_unknown=True`` also indexes PDFs the catalog doesn't
        recognise, with a generic auto-strategy spec.
        """
        return cls.from_specs(
            default_catalog(reference_dir, include_unknown=include_unknown),
            loader=loader,
            cache_dir=cache_dir,
            **kwargs,
        )

    def build(
        self, query: str, constructs: Iterable[ConstructKey] = ()
    ) -> str | None:
        """
        The Markdown block(s) for one item — a ``## Project instructions``
        block for selected user rules above a ``## {heading}`` block for
        reference guidance, either omitted when empty — or ``None`` when
        nothing at all is relevant (so the caller injects no block).
        """
        picks = self._selector.select(
            query,
            constructs,
            max_words=self.max_instruction_words,
            top_k=self.top_k,
        )
        if not picks:
            logger.debug("build: no relevant instruction chunks; no block")
            return None
        user = [c for c in picks if c.role is DocRole.USER_INSTRUCTION]
        reference = [c for c in picks if c.role is not DocRole.USER_INSTRUCTION]
        logger.debug(
            f"build: {len(user)} user + {len(reference)} reference chunk(s) injected"
        )
        blocks: list[str] = []
        if user:
            blocks.append(self._format_user(user))
        if reference:
            blocks.append(self._format_reference(reference))
        return "\n\n".join(blocks)

    @staticmethod
    def _body_of(chunk: InstructionChunk) -> str:
        # Strip the retrieval-only title/breadcrumb prefix; the header line
        # rendered above the body already states the location.
        parts = chunk.text.split("\n\n", 1)
        return (parts[1] if len(parts) > 1 else chunk.text).strip()

    def _format_user(self, picks: list[InstructionChunk]) -> str:
        lines = [f"## {self.project_heading}", ""]
        for chunk in picks:
            # Operator rules cite no document or pages — just their heading.
            lines.append(f"### {chunk.section_path}")
            lines.append(self._body_of(chunk))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_reference(self, picks: list[InstructionChunk]) -> str:
        lines = [f"## {self.heading}", ""]
        for chunk in picks:
            pages = (
                f"p. {chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp. {chunk.page_start}-{chunk.page_end}"
            )
            lines.append(f"### [{chunk.doc_id} · {chunk.section_path} · {pages}]")
            lines.append(self._body_of(chunk))
            lines.append("")
        return "\n".join(lines).rstrip()
