"""Rendering reports for delivery — currently, Markdown to PDF.

A leaf package: it imports ``markdown_it`` and ``pymupdf`` and nothing from
this repo, so any layer may depend on it without a cycle. It exists because
:mod:`complexity` and :mod:`validation` both had a Markdown-to-PDF renderer,
running the same paging loop against two stylesheets that had drifted apart —
and the weaker copy was the one an operator actually received.

Logger names: ``reporting.*``.
"""

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
