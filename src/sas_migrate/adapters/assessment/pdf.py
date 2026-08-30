"""PyMuPDF assessment report presenter."""

from __future__ import annotations

from sas_migrate.application.assessment import AssessmentReport, render_markdown


def render_pdf(report: AssessmentReport) -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    cursor = 54.0
    for source_line in render_markdown(report).splitlines():
        if cursor > page.rect.height - 54:
            page = document.new_page()
            cursor = 54.0
        page.insert_text((54, cursor), source_line.replace("`", "")[:120], fontsize=8)
        cursor += 11
    output = document.tobytes(garbage=4, deflate=True)
    document.close()
    return output


__all__ = ["render_pdf"]
