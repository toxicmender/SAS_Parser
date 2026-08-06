"""The complexity report as a PDF.

The rendering lives in :mod:`reporting.pdf`, which :mod:`validation` renders
through as well — one stylesheet, one paging loop, one set of rules about code
folding and images. This module is the name ``complexity`` reaches it by, kept
so the CLI's ``--pdf`` path and anything importing ``complexity.pdf`` are
unaffected by where the implementation sits.

Nothing is re-implemented here, and nothing should be: a complexity-specific
tweak to the layout belongs in ``reporting.pdf`` behind a parameter, because
the validation report wants the same page to look the same.

Logger name: ``reporting.pdf``.
"""

from __future__ import annotations

from reporting.pdf import (
    CODE_WIDTH,
    PAGE_MARGIN,
    PAGE_SIZE,
    STYLESHEET,
    PdfRenderError,
    markdown_to_html,
    render_markdown_pdf,
    render_pdf,
    wrap_code,
)

__all__ = [
    "CODE_WIDTH",
    "PAGE_MARGIN",
    "PAGE_SIZE",
    "STYLESHEET",
    "PdfRenderError",
    "markdown_to_html",
    "render_markdown_pdf",
    "render_pdf",
    "wrap_code",
]
