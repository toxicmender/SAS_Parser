"""Reference-PDF reader: PDF -> instruction sections. See prompt_builder/README.md.

Two extraction strategies behind one interface:

* **TOC** — segment on the PDF's own table of contents (``doc.get_toc``). Used
  for the SAS language manuals, whose deep, reliable TOCs put each function /
  statement / PROC in its own leaf entry.
* **Font** — segment on font-size heading heuristics, for documents with no
  usable TOC (e.g. the Spark guide). Degrades to one section per page when no
  heading tier can be found.

Like the SAS chunker, the reader never raises on a malformed document: it
emits :class:`InstructionDiagnostic`s (``NO_TOC``, ``NO_HEADINGS_DETECTED``,
``NO_TEXT_LAYER``, ``EMPTY_DOCUMENT``) and returns whatever it could recover.

Logger name: ``prompt_builder.pdf_reader``.
"""

from __future__ import annotations

import bisect
import logging
import re
import unicodedata
from collections import Counter

import pymupdf

from .models import (
    ConstructKey,
    DocRole,
    DocSection,
    ExtractionStrategy,
    InstructionDiagnostic,
    InstructionDoc,
)

logger = logging.getLogger(__name__)

# Front-matter / back-matter TOC entries that carry no translation guidance.
_SKIP_TITLE_RE = re.compile(
    r"^(contents|about this book|what.?s new.*|recommended reading|"
    r"index|glossary|references?|dictionary$|"
    r"(overview of |)syntax conventions.*|style conventions.*|"
    r"syntax components.*|special characters.*)$",
    re.IGNORECASE,
)

# Punctuation normalisation: curly quotes / dashes -> ASCII so tokenisation and
# BM25 see one canonical form. Applied after NFKC (which folds ligatures).
_PUNCT_MAP = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",
    ord(" "): " ",
}


# ---------------------------------------------------------------------------
# Construct-key parsing (SAS reference section titles -> lookup keys)
# ---------------------------------------------------------------------------


def parse_construct_key(title: str) -> ConstructKey | None:
    """
    Parse a SAS reference section title into a :class:`ConstructKey`, or
    ``None`` when the title names no single construct.

    ``"INTNX Function"`` -> ``function:intnx``; ``"%LET Statement"`` ->
    ``macro_statement:let``; ``"CALL SYMPUT Routine"`` ->
    ``call_routine:symput``; ``"The SQL Procedure"`` -> ``proc:sql``.
    """
    t = re.sub(r"\s*\([^)]*\)", "", title.strip())  # drop parentheticals
    t = re.sub(r"^the\s+", "", t, flags=re.IGNORECASE).strip()
    low = t.lower()
    is_macro = t.startswith("%")

    if m := re.match(r"^([a-z_]\w*)\s+procedure$", low):
        return ConstructKey(kind="proc", name=m.group(1))
    if m := re.match(r"^(?:call\s+)?([a-z_]\w*)\s+(?:call\s+)?routine$", low):
        return ConstructKey(kind="call_routine", name=m.group(1))
    if m := re.match(r"^%?([a-z_]\w*)\s+function$", low):
        kind = "macro_function" if is_macro else "function"
        return ConstructKey(kind=kind, name=m.group(1))
    if m := re.match(r"^%?([a-z_]\w*)\s+statement$", low):
        kind = "macro_statement" if is_macro else "global_statement"
        return ConstructKey(kind=kind, name=m.group(1))
    if m := re.match(r"^([a-z_]\w*)=?\s+system\s+option$", low):
        return ConstructKey(kind="system_option", name=m.group(1))
    if m := re.match(r"^([a-z_]\w*)=?\s+option$", low):
        return ConstructKey(kind="option", name=m.group(1))
    if m := re.match(r"^\$?([a-z_]\w*?)w?\.?\s+(in)?format$", low):
        return ConstructKey(kind="informat" if m.group(2) else "format", name=m.group(1))
    return None


# ---------------------------------------------------------------------------
# Text cleanup (shared by both strategies)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """NFKC-fold, straighten punctuation, de-hyphenate line breaks, tidy gaps."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_MAP)
    text = text.replace("�", "")  # drop replacement chars (lost glyphs)
    text = re.sub(r"-\n(?=\w)", "", text)  # join words split across a line break
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank-line runs
    return text.strip()


def _strip_running_headers(
    pages: list[str], threshold: float, edge: int = 2
) -> list[str]:
    """
    Drop repeated running headers/footers and bare page numbers from the top
    and bottom *edge* lines of each page. A line is a running header when its
    exact text recurs on at least ``threshold`` of the pages.
    """
    counts: Counter[str] = Counter()
    for text in pages:
        lines = text.split("\n")
        for line in lines[:edge] + lines[-edge:]:
            stripped = line.strip()
            if stripped:
                counts[stripped] += 1
    min_repeats = max(3, int(threshold * len(pages)))
    common = {s for s, c in counts.items() if c >= min_repeats}

    cleaned: list[str] = []
    for text in pages:
        lines = text.split("\n")
        n = len(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (
                (i < edge or i >= n - edge)
                and (line.strip() in common or re.fullmatch(r"\d+", line.strip()))
            )
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def _concat_pages(pages: list[str]) -> tuple[str, list[int]]:
    """Join *pages* into one string; return it and each page's start offset."""
    parts: list[str] = []
    starts: list[int] = []
    offset = 0
    for page in pages:
        starts.append(offset)
        chunk = page + "\n"
        parts.append(chunk)
        offset += len(chunk)
    return "".join(parts), starts


def _page_of(offset: int, starts: list[int]) -> int:
    """1-based page number containing *offset* in the concatenated text."""
    return max(0, min(bisect.bisect_right(starts, offset) - 1, len(starts) - 1)) + 1


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class PdfReader:
    """
    Extract instruction sections from a reference PDF.

    Parameters
    ----------
    min_body_ratio : float
        Font strategy: a line whose max span size is at least this multiple of
        the modal body size is treated as a heading.
    max_heading_words : int
        Font strategy: lines longer than this are body text, never headings.
    header_footer_threshold : float
        Fraction of pages on which an edge line must recur to be stripped as a
        running header/footer.
    min_page_chars : int
        A page with fewer non-space characters is considered to have no text
        layer (scanned/blank) for diagnostics and page-fallback.
    max_heading_search_pages : int
        TOC strategy: how many pages past a heading's TOC page to search for
        its text before falling back to the page boundary.
    """

    def __init__(
        self,
        *,
        min_body_ratio: float = 1.5,
        max_heading_words: int = 12,
        header_footer_threshold: float = 0.3,
        min_page_chars: int = 4,
        max_heading_search_pages: int = 2,
    ) -> None:
        self.min_body_ratio = min_body_ratio
        self.max_heading_words = max_heading_words
        self.header_footer_threshold = header_footer_threshold
        self.min_page_chars = min_page_chars
        self.max_heading_search_pages = max_heading_search_pages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        path: str,
        *,
        doc_id: str | None = None,
        role: DocRole = DocRole.SAS_REFERENCE,
        strategy: ExtractionStrategy | str = "auto",
        section_level: int | None = None,
    ) -> tuple[InstructionDoc, list[DocSection]]:
        """
        Read the PDF at *path* into an :class:`InstructionDoc` summary and its
        list of :class:`DocSection`s.

        ``strategy`` may be ``"auto"`` (TOC when the PDF has one, else font),
        or an explicit :class:`ExtractionStrategy`. ``section_level`` pins the
        TOC depth to segment at; ``None`` auto-selects the most populated
        level.
        """
        doc_id = doc_id or path
        logger.info(f"PdfReader.read: doc_id='{doc_id}'  strategy={strategy}")
        diagnostics: list[InstructionDiagnostic] = []
        doc = pymupdf.open(path)
        try:
            page_count = doc.page_count
            raw_pages = [doc[i].get_text("text") for i in range(page_count)]
            self._flag_missing_text(raw_pages, doc_id, diagnostics)
            cleaned_pages = [
                _normalize_text(p)
                for p in _strip_running_headers(raw_pages, self.header_footer_threshold)
            ]

            resolved = self._resolve_strategy(doc, strategy, doc_id, diagnostics)
            if resolved is ExtractionStrategy.TOC:
                sections, used = self._read_toc(
                    doc, doc_id, cleaned_pages, section_level, diagnostics
                )
            else:
                sections, used = self._read_font(
                    doc, doc_id, cleaned_pages, diagnostics
                )
        finally:
            doc.close()

        sections = [s for s in sections if s.text.strip()]
        for s in sections:
            s.construct_key = parse_construct_key(s.title)

        summary = InstructionDoc(
            doc_id=doc_id,
            path=path,
            role=role,
            page_count=page_count,
            strategy=used,
            section_count=len(sections),
            diagnostics=diagnostics,
        )
        logger.info(
            f"PdfReader.read: doc_id='{doc_id}'  {len(sections)} section(s) via "
            f"{used}  {len(diagnostics)} diagnostic(s)"
        )
        return summary, sections

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self,
        doc: pymupdf.Document,
        requested: ExtractionStrategy | str,
        doc_id: str,
        diagnostics: list[InstructionDiagnostic],
    ) -> ExtractionStrategy:
        has_toc = bool(doc.get_toc(simple=True))
        if requested == "auto":
            return ExtractionStrategy.TOC if has_toc else ExtractionStrategy.FONT
        requested = ExtractionStrategy(requested)
        if requested is ExtractionStrategy.TOC and not has_toc:
            diagnostics.append(
                InstructionDiagnostic(
                    code="NO_TOC",
                    message="TOC strategy requested but document has no TOC; "
                    "falling back to font heuristics",
                    doc_id=doc_id,
                )
            )
            return ExtractionStrategy.FONT
        return requested

    # ------------------------------------------------------------------
    # TOC strategy
    # ------------------------------------------------------------------

    def _read_toc(
        self,
        doc: pymupdf.Document,
        doc_id: str,
        pages: list[str],
        section_level: int | None,
        diagnostics: list[InstructionDiagnostic],
    ) -> tuple[list[DocSection], ExtractionStrategy]:
        toc = doc.get_toc(simple=True)
        level = section_level or _auto_section_level(toc)
        boundaries = [
            (lvl, _normalize_text(title), page)
            for lvl, title, page in toc
            if page >= 1 and lvl <= level
        ]
        if not boundaries:
            diagnostics.append(
                InstructionDiagnostic(
                    code="NO_HEADINGS_DETECTED",
                    message=f"no TOC entries at or above level {level}",
                    doc_id=doc_id,
                )
            )
            return self._page_fallback(pages, doc_id), ExtractionStrategy.PAGE

        full, page_starts = _concat_pages(pages)
        npages = len(pages)
        starts = self._locate_boundaries(full, page_starts, boundaries, npages)

        crumb: dict[int, str] = {}
        sections: list[DocSection] = []
        for i, (lvl, title, _page) in enumerate(boundaries):
            crumb[lvl] = title
            for deeper in [d for d in crumb if d > lvl]:
                del crumb[deeper]
            if _SKIP_TITLE_RE.match(title.strip()):
                continue
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else len(full)
            text = full[start:end].strip()
            if not text:
                continue
            breadcrumb = " > ".join(crumb[d] for d in sorted(crumb))
            sections.append(
                DocSection(
                    doc_id=doc_id,
                    section_path=breadcrumb,
                    title=title.strip(),
                    text=text,
                    page_start=_page_of(start, page_starts),
                    page_end=_page_of(max(start, end - 1), page_starts),
                    level=lvl,
                )
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"_read_toc: doc_id='{doc_id}'  level={level}  "
                f"{len(boundaries)} boundary/ies -> {len(sections)} section(s)"
            )
        return sections, ExtractionStrategy.TOC

    def _locate_boundaries(
        self,
        full: str,
        page_starts: list[int],
        boundaries: list[tuple[int, str, int]],
        npages: int,
    ) -> list[int]:
        """
        Offset in *full* where each boundary's section begins. Located by
        searching for the heading title from its TOC page onward (so two
        sections sharing a page split at the second title), monotonically
        non-decreasing, falling back to the page start when not found.
        """
        starts: list[int] = []
        search_from = 0
        for _lvl, title, page in boundaries:
            p0 = min(page - 1, npages - 1)
            lo = max(search_from, page_starts[p0])
            end_page = min(p0 + self.max_heading_search_pages, npages - 1)
            hi = page_starts[end_page + 1] if end_page + 1 < npages else len(full)
            idx = _find_heading(full, title, lo, hi)
            start = max(idx if idx is not None else lo, search_from)
            starts.append(start)
            search_from = start + max(len(title), 1)
        return starts

    # ------------------------------------------------------------------
    # Font strategy
    # ------------------------------------------------------------------

    def _read_font(
        self,
        doc: pymupdf.Document,
        doc_id: str,
        pages: list[str],
        diagnostics: list[InstructionDiagnostic],
    ) -> tuple[list[DocSection], ExtractionStrategy]:
        page_lines = [_page_size_lines(doc[i]) for i in range(doc.page_count)]
        body = _modal_body_size(page_lines)
        heading_min = body * self.min_body_ratio

        heading_sizes = sorted(
            {
                round(size, 1)
                for lines in page_lines
                for text, size in lines
                if size >= heading_min and self._looks_like_heading(text)
            },
            reverse=True,
        )
        if not heading_sizes:
            diagnostics.append(
                InstructionDiagnostic(
                    code="NO_HEADINGS_DETECTED",
                    message="no heading-sized text found; using page fallback",
                    doc_id=doc_id,
                )
            )
            return self._page_fallback(pages, doc_id), ExtractionStrategy.PAGE

        size_to_level = {size: i + 1 for i, size in enumerate(heading_sizes)}
        running = _running_header_texts(page_lines, self.header_footer_threshold)

        sections: list[DocSection] = []
        crumb: dict[int, str] = {}
        buffer: list[str] = []
        cur_title = doc_id
        cur_path = doc_id
        cur_level = 0
        start_page = 1
        last_page = 1

        def flush() -> None:
            text = _normalize_text("\n".join(buffer))
            if text.strip():
                sections.append(
                    DocSection(
                        doc_id=doc_id,
                        section_path=cur_path,
                        title=cur_title,
                        text=text,
                        page_start=start_page,
                        page_end=last_page,
                        level=cur_level,
                    )
                )

        for pno, lines in enumerate(page_lines, start=1):
            for text, size in lines:
                stripped = text.strip()
                if not stripped or stripped in running:
                    continue
                is_heading = (
                    size >= heading_min
                    and round(size, 1) in size_to_level
                    and self._looks_like_heading(stripped)
                )
                if is_heading:
                    flush()
                    buffer = []
                    heading = _normalize_text(stripped)
                    cur_level = size_to_level[round(size, 1)]
                    crumb[cur_level] = heading
                    for deeper in [d for d in crumb if d > cur_level]:
                        del crumb[deeper]
                    cur_title = heading
                    cur_path = " > ".join(crumb[d] for d in sorted(crumb))
                    start_page = pno
                    last_page = pno
                else:
                    buffer.append(stripped)
                    last_page = pno
        flush()
        return sections, ExtractionStrategy.FONT

    def _looks_like_heading(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped or stripped.isdigit():
            return False
        return len(stripped.split()) <= self.max_heading_words

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _page_fallback(self, pages: list[str], doc_id: str) -> list[DocSection]:
        sections: list[DocSection] = []
        for i, text in enumerate(pages, start=1):
            if len(text.strip()) < self.min_page_chars:
                continue
            sections.append(
                DocSection(
                    doc_id=doc_id,
                    section_path=f"Page {i}",
                    title=f"Page {i}",
                    text=text.strip(),
                    page_start=i,
                    page_end=i,
                    level=1,
                )
            )
        return sections

    def _flag_missing_text(
        self,
        raw_pages: list[str],
        doc_id: str,
        diagnostics: list[InstructionDiagnostic],
    ) -> None:
        if not raw_pages:
            diagnostics.append(
                InstructionDiagnostic(
                    code="EMPTY_DOCUMENT", message="document has no pages", doc_id=doc_id
                )
            )
            return
        empty = sum(1 for p in raw_pages if len(p.strip()) < self.min_page_chars)
        if empty > len(raw_pages) // 2:
            diagnostics.append(
                InstructionDiagnostic(
                    code="NO_TEXT_LAYER",
                    message=f"{empty}/{len(raw_pages)} pages have no extractable "
                    "text (scanned or image-only?)",
                    doc_id=doc_id,
                )
            )


# ---------------------------------------------------------------------------
# Module-level extraction helpers
# ---------------------------------------------------------------------------


def _auto_section_level(toc: list) -> int:
    """Pick the most populated TOC level — the leaf 'dictionary' level."""
    if not toc:
        return 1
    counts = Counter(lvl for lvl, _title, _page in toc)
    return max(counts, key=lambda lvl: (counts[lvl], -lvl))


def _find_heading(full: str, title: str, start: int, end: int) -> int | None:
    """Offset of *title* in ``full[start:end]``, tolerant of whitespace runs."""
    words = [re.escape(w) for w in title.split()]
    if not words:
        return None
    pattern = re.compile(r"\s+".join(words), re.IGNORECASE)
    match = pattern.search(full, start, end)
    return match.start() if match else None


def _page_size_lines(page: pymupdf.Page) -> list[tuple[str, float]]:
    """Each line on *page* as ``(text, max-span-size)``, in reading order."""
    lines: list[tuple[str, float]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            size = max(s["size"] for s in spans)
            lines.append((text, size))
    return lines


def _modal_body_size(page_lines: list[list[tuple[str, float]]]) -> float:
    """Most common span size weighted by character count — the body text size."""
    weighted: Counter[float] = Counter()
    for lines in page_lines:
        for text, size in lines:
            weighted[round(size, 1)] += len(text)
    if not weighted:
        return 0.0
    return weighted.most_common(1)[0][0]


def _running_header_texts(
    page_lines: list[list[tuple[str, float]]], threshold: float, edge: int = 2
) -> set[str]:
    """Edge-line texts that recur on at least *threshold* of the pages."""
    counts: Counter[str] = Counter()
    for lines in page_lines:
        texts = [t.strip() for t, _ in lines]
        for text in texts[:edge] + texts[-edge:]:
            if text:
                counts[text] += 1
    min_repeats = max(3, int(threshold * len(page_lines)))
    return {t for t, c in counts.items() if c >= min_repeats}
