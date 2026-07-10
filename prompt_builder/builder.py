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
from .models import ConstructKey, InstructionChunk
from .selector import InstructionSelector

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
        Markdown H2 heading for the block.
    """

    def __init__(
        self,
        chunks: Iterable[InstructionChunk],
        *,
        top_k: int | None = None,
        max_instruction_words: int | None = None,
        pinned_sections: Iterable[str] = (),
        embeddings: Any | None = None,
        embedding_cache_path: str | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        heading: str = "Relevant migration guidance",
    ) -> None:
        self.top_k = app_config.resolve(top_k, "prompt_builder", "top_k", 6)
        self.max_instruction_words = app_config.resolve(
            max_instruction_words, "prompt_builder", "max_instruction_words", 1500
        )
        self.heading = heading
        self._selector = InstructionSelector(
            chunks,
            embeddings=embeddings,
            embedding_cache_path=embedding_cache_path,
            rrf_k=rrf_k,
            reranker=reranker,
            pinned_sections=pinned_sections,
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
        loader: CorpusLoader | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "PromptBuilder":
        """Build from the default catalog of PDFs present under *reference_dir*."""
        return cls.from_specs(
            default_catalog(reference_dir),
            loader=loader,
            cache_dir=cache_dir,
            **kwargs,
        )

    def build(
        self, query: str, constructs: Iterable[ConstructKey] = ()
    ) -> str | None:
        """
        A Markdown guidance block for one item, or ``None`` when nothing is
        relevant (so the caller injects no block at all).
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
        logger.debug(f"build: {len(picks)} instruction chunk(s) injected")
        return self._format(picks)

    def _format(self, picks: list[InstructionChunk]) -> str:
        lines = [f"## {self.heading}", ""]
        for chunk in picks:
            # Strip the retrieval-only breadcrumb prefix; the header line below
            # already states the location.
            parts = chunk.text.split("\n\n", 1)
            body = parts[1] if len(parts) > 1 else chunk.text
            pages = (
                f"p. {chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp. {chunk.page_start}-{chunk.page_end}"
            )
            lines.append(f"### [{chunk.doc_id} · {chunk.section_path} · {pages}]")
            lines.append(body.strip())
            lines.append("")
        return "\n".join(lines).rstrip()
