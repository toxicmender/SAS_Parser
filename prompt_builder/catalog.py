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

# Manual escape hatch: bump to force a full re-extract even when nothing the
# automatic signature tracks has changed. Routine reader/chunker edits are
# picked up automatically via _code_fingerprint().
EXTRACTOR_VERSION = 1

_DEFAULT_CACHE_DIR = ".prompt_builder_cache"

_CODE_HASH: str | None = None


def _code_fingerprint() -> str:
    """
    Hash of the extractor source files (pdf_reader.py + doc_chunker.py),
    folded into every cache signature so a code change re-extracts
    automatically — no manual EXTRACTOR_VERSION bump needed for the common
    case. Computed once per process.
    """
    global _CODE_HASH
    if _CODE_HASH is None:
        from . import doc_chunker as _dc
        from . import pdf_reader as _pr

        digest = hashlib.sha256()
        for module in (_pr, _dc):
            digest.update(Path(module.__file__).read_bytes())
        _CODE_HASH = digest.hexdigest()[:12]
    return _CODE_HASH


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

    def load_documents(self, specs: list[DocumentSpec]) -> list:
        """
        The corpus as LangChain ``Document``s (see
        :meth:`InstructionChunk.to_document`), for feeding a LangChain vector
        store / retriever / index instead of the built-in selector.
        """
        return [chunk.to_document() for chunk in self.load(specs)]

    def load_one(self, spec: DocumentSpec) -> list[InstructionChunk]:
        """Chunks for one *spec*, from cache when valid, else freshly extracted."""
        source = Path(spec.path)
        if not source.exists():
            logger.warning(f"CorpusLoader: file not found, skipping '{spec.path}'")
            return []

        payload: dict | None = None
        signature: str | None = None
        file_sha = ""
        if self.use_cache:
            payload = self._read_payload(self._cache_path(spec.doc_id))
            file_sha = self._file_sha_fast(source, payload)
            signature = self._signature(spec, file_sha)
            if payload is not None and payload.get("signature") == signature:
                cached = [
                    InstructionChunk.model_validate(c) for c in payload["chunks"]
                ]
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
            self._write_cache(spec.doc_id, signature, source, file_sha, chunks)
        return chunks

    # ------------------------------------------------------------------
    # Freshness API
    # ------------------------------------------------------------------

    def check_freshness(self, spec: DocumentSpec) -> str:
        """
        Cache status for one *spec* without extracting:
        ``"fresh"`` (cache valid), ``"stale"`` (source bytes, extractor code,
        or parameters changed), ``"uncached"`` (no readable cache entry), or
        ``"missing"`` (source file absent).
        """
        source = Path(spec.path)
        if not source.exists():
            return "missing"
        payload = self._read_payload(self._cache_path(spec.doc_id))
        if payload is None:
            return "uncached"
        file_sha = self._file_sha_fast(source, payload)
        return "fresh" if payload.get("signature") == self._signature(spec, file_sha) else "stale"

    def freshness_report(self, specs: list[DocumentSpec]) -> dict[str, str]:
        """Per-``doc_id`` freshness status (see :meth:`check_freshness`)."""
        report = {spec.doc_id: self.check_freshness(spec) for spec in specs}
        logger.info(f"freshness_report: {report}")
        return report

    def prune_stale(self, specs: list[DocumentSpec]) -> list[str]:
        """
        Delete cache entries that no longer serve *specs*: stale entries,
        entries for specs whose source file is gone, and orphaned entries no
        spec refers to. Returns the deleted file names.
        """
        keep: set[Path] = set()
        for spec in specs:
            if self.check_freshness(spec) == "fresh":
                keep.add(self._cache_path(spec.doc_id))
        removed: list[str] = []
        if self.cache_dir.is_dir():
            for entry in sorted(self.cache_dir.glob("*.json")):
                if entry not in keep:
                    entry.unlink()
                    removed.append(entry.name)
        if removed:
            logger.info(f"prune_stale: removed {removed}")
        return removed

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _cache_path(self, doc_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)
        return self.cache_dir / f"{safe}.json"

    @staticmethod
    def _file_sha_fast(source: Path, payload: dict | None) -> str:
        """
        The source file's SHA-256, trusting the cached value when size and
        mtime both match (the stat fast-path — no rehash of a multi-MB PDF on
        every load); any stat difference forces a real rehash.
        """
        stat = source.stat()
        if (
            payload is not None
            and payload.get("file_size") == stat.st_size
            and payload.get("file_mtime_ns") == stat.st_mtime_ns
            and payload.get("file_sha")
        ):
            return payload["file_sha"]
        return _file_sha256(source)

    def _signature(self, spec: DocumentSpec, file_sha: str) -> str:
        """Hash of everything that determines a document's chunks."""
        parts = [
            f"v={EXTRACTOR_VERSION}",
            f"code={_code_fingerprint()}",
            f"sha={file_sha}",
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

    @staticmethod
    def _read_payload(cache_path: Path) -> dict | None:
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"CorpusLoader: unreadable cache '{cache_path}': {exc}")
            return None

    def _write_cache(
        self,
        doc_id: str,
        signature: str | None,
        source: Path,
        file_sha: str,
        chunks: list[InstructionChunk],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        payload = {
            "signature": signature,
            "file_sha": file_sha,
            "file_size": stat.st_size,
            "file_mtime_ns": stat.st_mtime_ns,
            "chunks": [c.model_dump(mode="json") for c in chunks],
        }
        cache_path = self._cache_path(doc_id)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.debug(
            f"CorpusLoader: wrote cache '{cache_path}' ({len(chunks)} chunk(s))"
        )
