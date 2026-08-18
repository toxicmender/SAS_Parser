"""Lazy PyMuPDF instruction-document extraction."""

from __future__ import annotations

from typing import cast

from sas_migrate.application.knowledge.models import (
    DocumentExtraction,
    DocumentSection,
    ExtractionDiagnostic,
)


class PyMuPdfInstructionReader:
    def read(self, content: bytes, *, source_id: str) -> DocumentExtraction:
        import fitz

        diagnostics: list[ExtractionDiagnostic] = []
        sections: list[DocumentSection] = []
        with fitz.open(stream=content, filetype="pdf") as document:
            toc = document.get_toc(simple=True)
            headings = {
                max(0, int(page) - 1): str(title).strip()
                for _level, title, page in toc
                if str(title).strip()
            }
            current_heading = source_id
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                current_heading = headings.get(page_index, current_heading)
                try:
                    text = cast("str", page.get_text("text")).strip()
                except (RuntimeError, ValueError) as exc:
                    diagnostics.append(
                        ExtractionDiagnostic(
                            code="page_extraction_failed",
                            message=str(exc) or type(exc).__name__,
                            page=page_index + 1,
                        )
                    )
                    continue
                if not text:
                    diagnostics.append(
                        ExtractionDiagnostic(
                            code="empty_page",
                            message="page contained no extractable text",
                            page=page_index + 1,
                        )
                    )
                    continue
                sections.append(
                    DocumentSection(
                        source_id=source_id,
                        section_path=current_heading,
                        text=text,
                        page_start=page_index + 1,
                        page_end=page_index + 1,
                    )
                )
            return DocumentExtraction(
                source_id=source_id,
                sections=tuple(sections),
                strategy="toc" if toc else "page",
                page_count=len(document),
                diagnostics=tuple(diagnostics),
            )


__all__ = ["PyMuPdfInstructionReader"]
