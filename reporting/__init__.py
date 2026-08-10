"""Shared Markdown-to-PDF reporting utilities. See ``reporting/README.md``."""

from __future__ import annotations

from .pdf import (
    CODE_WIDTH,
    PdfRenderError,
    markdown_to_html,
    render_markdown_bytes,
    render_markdown_pdf,
    render_pdf,
    wrap_code,
)

__all__ = [
    "CODE_WIDTH",
    "PdfRenderError",
    "markdown_to_html",
    "render_markdown_bytes",
    "render_markdown_pdf",
    "render_pdf",
    "wrap_code",
]
