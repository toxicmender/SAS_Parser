"""Reference-corpus catalog + on-disk extraction cache. See prompt_builder/README.md.

A :class:`DocumentSpec` says *how* to read one reference PDF (strategy, TOC
depth, role); :func:`default_catalog` ships the specs for the bundled
``reference_docs/`` set; :class:`CorpusLoader` reads and chunks each spec into
:class:`InstructionChunk`s, memoised on disk.

The cache matters: reading + chunking the ~7,400-page corpus costs tens of
seconds per document (PyMuPDF text extraction dominates), and none of it
changes between runs. Each document's chunks are cached as JSON keyed by the
file's SHA-256 plus everything else that affects output (extractor version,
spec, reader/chunker parameters); a matching cache entry skips PyMuPDF
entirely.

Logger name: ``prompt_builder.catalog``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .doc_chunker import InstructionChunker
from .models import DocRole, InstructionChunk
from .pdf_reader import PdfReader

logger = logging.getLogger(__name__)

# Bump when the reader or chunker changes in a way that alters output, so old
# cache entries miss and re-extract even if the source file is unchanged.
EXTRACTOR_VERSION = 1

_DEFAULT_CACHE_DIR = ".prompt_builder_cache"


class DocumentSpec(BaseModel):
    """How to read one reference document into instruction chunks."""

    path: str
    doc_id: str
    role: DocRole = DocRole.SAS_REFERENCE
    # "auto" | "toc" | "font" — forwarded to PdfReader.read.
    strategy: str = "auto"
    section_level: int | None = None
    # Section-path substrings whose chunks the builder always injects (Phase 6);
    # stored as configuration here, unused until then.
    pinned_sections: list[str] = Field(default_factory=list)


# Bundled reference set. Section levels are pinned from each manual's TOC shape
# (see the Phase-2 probe): the SAS manuals put one function/statement/PROC per
# leaf entry, the Spark excerpt has no TOC so it takes the font strategy, and
# the 4-page ref sheet is a slide-per-section cheat sheet.
_DEFAULT_SPECS: tuple[tuple[str, str, DocRole, str, int | None], ...] = (
    ("Base_SAS.pdf", "base_sas", DocRole.SAS_REFERENCE, "auto", 4),
    (
        "SAS Programmer's Guide - Essentials.pdf",
        "programmers_guide",
        DocRole.SAS_REFERENCE,
        "auto",
        4,
    ),
    (
        "SAS_Functions_and_Call_Routines.pdf",
        "functions",
        DocRole.SAS_REFERENCE,
        "auto",
        4,
    ),
    ("SAS_Global_Statements.pdf", "global_statements", DocRole.SAS_REFERENCE, "auto", 3),
    ("SAS_Macro_Language_Reference.pdf", "macro_language", DocRole.SAS_REFERENCE, "auto", 4),
    ("SAS_Procedures.pdf", "procedures", DocRole.SAS_REFERENCE, "auto", 4),
    ("SAS_Base_Ref_Sheet.pdf", "base_cheat_sheet", DocRole.CHEAT_SHEET, "auto", 1),
    (
        "Apache-Spark-The-Definitive-Guide-Excerpts-R1.pdf",
        "spark_guide",
        DocRole.TARGET_GUIDE,
        "font",
        None,
    ),
)


def default_catalog(reference_dir: str = "reference_docs") -> list[DocumentSpec]:
    """
    Specs for the bundled reference set that are actually present under
    *reference_dir* — the directory is user-provided and may hold only a
    subset, so missing files are skipped rather than erroring.
    """
    base = Path(reference_dir)
    specs: list[DocumentSpec] = []
    for filename, doc_id, role, strategy, level in _DEFAULT_SPECS:
        path = base / filename
        if path.exists():
            specs.append(
                DocumentSpec(
                    path=str(path),
                    doc_id=doc_id,
                    role=role,
                    strategy=strategy,
                    section_level=level,
                )
            )
        else:
            logger.debug(f"default_catalog: not present, skipping '{path}'")
    logger.info(
        f"default_catalog: {len(specs)}/{len(_DEFAULT_SPECS)} reference doc(s) "
        f"present under '{reference_dir}'"
    )
    return specs


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CorpusLoader:
    """
    Read and chunk :class:`DocumentSpec`s into instruction chunks, memoising
    each document's chunks on disk.

    Parameters
    ----------
    reader : PdfReader | None
        Reader to extract sections; a default is built if omitted.
    chunker : InstructionChunker | None
        Chunker to word-budget the sections; a default is built if omitted.
    cache_dir : str
        Directory for the per-document JSON cache (created on first write).
    use_cache : bool
        When ``False``, always re-extract and never read/write the cache.
    """

    def __init__(
        self,
        *,
        reader: PdfReader | None = None,
        chunker: InstructionChunker | None = None,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self.reader = reader or PdfReader()
        self.chunker = chunker or InstructionChunker()
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def load(self, specs: list[DocumentSpec]) -> list[InstructionChunk]:
        """Load and concatenate the chunks of every spec, in catalog order."""
        chunks: list[InstructionChunk] = []
        for spec in specs:
            chunks.extend(self.load_one(spec))
        logger.info(
            f"CorpusLoader.load: {len(specs)} spec(s) -> {len(chunks)} chunk(s)"
        )
        return chunks

    def load_one(self, spec: DocumentSpec) -> list[InstructionChunk]:
        """Chunks for one *spec*, from cache when valid, else freshly extracted."""
        source = Path(spec.path)
        if not source.exists():
            logger.warning(f"CorpusLoader: file not found, skipping '{spec.path}'")
            return []

        signature = self._signature(spec, source) if self.use_cache else None
        cache_path = self._cache_path(spec.doc_id)
        if self.use_cache:
            cached = self._read_cache(cache_path, signature)
            if cached is not None:
                logger.info(
                    f"CorpusLoader: cache hit doc_id='{spec.doc_id}' "
                    f"({len(cached)} chunk(s))"
                )
                return cached

        logger.info(
            f"CorpusLoader: extracting doc_id='{spec.doc_id}' from '{spec.path}'"
        )
        summary, sections = self.reader.read(
            spec.path,
            doc_id=spec.doc_id,
            role=spec.role,
            strategy=spec.strategy,
            section_level=spec.section_level,
        )
        chunks = self.chunker.chunk(sections, role=spec.role)
        if summary.diagnostics:
            logger.info(
                f"CorpusLoader: doc_id='{spec.doc_id}' "
                f"diagnostics={[d.code for d in summary.diagnostics]}"
            )
        if self.use_cache:
            self._write_cache(cache_path, signature, chunks)
        return chunks

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _cache_path(self, doc_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)
        return self.cache_dir / f"{safe}.json"

    def _signature(self, spec: DocumentSpec, source: Path) -> str:
        """Hash of everything that determines a document's chunks."""
        parts = [
            f"v={EXTRACTOR_VERSION}",
            f"sha={_file_sha256(source)}",
            f"doc={spec.doc_id}",
            f"role={spec.role}",
            f"strategy={spec.strategy}",
            f"level={spec.section_level}",
            f"reader={self.reader.min_body_ratio},{self.reader.max_heading_words},"
            f"{self.reader.header_footer_threshold},{self.reader.min_page_chars},"
            f"{self.reader.max_heading_search_pages}",
            f"chunker={self.chunker.min_words},{self.chunker.max_words},"
            f"{self.chunker.overlap_words}",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def _read_cache(
        self, cache_path: Path, signature: str | None
    ) -> list[InstructionChunk] | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"CorpusLoader: unreadable cache '{cache_path}': {exc}")
            return None
        if payload.get("signature") != signature:
            logger.debug(f"CorpusLoader: stale cache '{cache_path}', re-extracting")
            return None
        return [InstructionChunk.model_validate(c) for c in payload["chunks"]]

    def _write_cache(
        self, cache_path: Path, signature: str | None, chunks: list[InstructionChunk]
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": signature,
            "chunks": [c.model_dump(mode="json") for c in chunks],
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.debug(
            f"CorpusLoader: wrote cache '{cache_path}' ({len(chunks)} chunk(s))"
        )
