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
    top_k : int
        Maximum topical (ranked) chunks per item.
    max_instruction_words : int
        Word budget for the whole guidance block.
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
        top_k: int = 6,
        max_instruction_words: int = 1500,
        pinned_sections: Iterable[str] = (),
        embeddings: Any | None = None,
        embedding_cache_path: str | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        heading: str = "Relevant migration guidance",
    ) -> None:
        self.top_k = top_k
        self.max_instruction_words = max_instruction_words
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
        return cls(chunks, pinned_sections=pins, **kwargs)

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
