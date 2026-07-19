"""Render a :class:`ValidationReport` to PDF, and publish it to SharePoint.

The report already describes itself as GitHub-flavoured Markdown
(:meth:`ValidationReport.to_markdown`); this module turns that Markdown into a
paginated PDF and, on request, writes the PDF into a SharePoint document
library. Two dependencies carry it, both already in the tree:

- **markdown-it-py** renders the Markdown (the metric *table* included) to HTML;
- **PyMuPDF** (``pymupdf``, a core dependency) lays that HTML across A4 pages
  through its ``Story`` engine and emits the PDF bytes.

Nothing here touches the network at import time. SharePoint access is delegated
to :mod:`app_config.sharepoint`, imported lazily inside
:func:`publish_report_pdf`, so ``import validation`` stays cheap and free of the
optional ``sharepoint`` extra — only a caller that actually publishes pays for
it.

The SharePoint destination follows the repo-wide precedence rule (see
:mod:`app_config`): explicit argument > config.json
``validation.report_sharepoint_path`` > the library root. A destination ending
in ``.pdf`` is the exact file path; anything else names a folder, under which a
timestamped filename is created.

Logger name: ``validation.pdf``.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import app_config

from .models import ValidationReport

if TYPE_CHECKING:
    from app_config.sharepoint import SharePointClient

logger = logging.getLogger(__name__)

# A4 with a ~12.7mm (36pt) margin on every side.
_PAGE = "a4"
_MARGIN = 36.0

# Minimal print CSS: a readable sans body and a ruled, shaded-header table so
# the metric grid stays legible on paper. PyMuPDF's Story understands this
# HTML/CSS subset; callers append their own rules via ``css=`` (later wins).
_DEFAULT_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; }
h1 { font-size: 18pt; margin: 0 0 8pt 0; }
code { font-family: monospace; }
ul { margin: 0 0 8pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; }
th, td { border: 1px solid #999999; padding: 3px 6px; text-align: left; }
th { background: #eeeeee; }
"""


def report_to_pdf(report: ValidationReport | str, *, css: str | None = None) -> bytes:
    """
    Render *report* to PDF and return the raw bytes.

    Accepts a :class:`~validation.models.ValidationReport` (rendered through its
    :meth:`~validation.models.ValidationReport.to_markdown`) or a Markdown
    string directly, so a report reconstructed elsewhere can be published
    without a live run.

    Parameters
    ----------
    css : str | None
        Extra CSS appended after the built-in print stylesheet (later rules
        win), for callers that want to restyle the page.
    """
    import pymupdf
    from markdown_it import MarkdownIt

    markdown = (
        report.to_markdown() if isinstance(report, ValidationReport) else report
    )
    # CommonMark plus the GFM table/strikethrough the report needs — not the
    # full "gfm-like" preset, whose linkify rule pulls an extra dependency.
    renderer = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    html = f"<html><head></head><body>{renderer.render(markdown)}</body></html>"

    story = pymupdf.Story(html=html, user_css=_DEFAULT_CSS + (css or ""))
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page_rect = pymupdf.paper_rect(_PAGE)
    content_rect = page_rect + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)
    pages = 0
    more = 1
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    data = buffer.getvalue()
    logger.info(f"report_to_pdf: rendered {pages} page(s), {len(data)} bytes")
    return data


def _report_stamp(report: ValidationReport | str) -> datetime:
    """The report's ``created_at`` (a real report) or now, in UTC."""
    if isinstance(report, ValidationReport):
        stamp = report.created_at
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp
    return datetime.now(timezone.utc)


def _default_filename(report: ValidationReport | str) -> str:
    """A timestamped name, e.g. ``validation-report-20260720T134501Z.pdf``."""
    return f"validation-report-{_report_stamp(report):%Y%m%dT%H%M%SZ}.pdf"


def _resolve_sharepoint_path(dest: str, report: ValidationReport | str) -> str:
    """
    Turn a SharePoint *dest* into a concrete ``<folder>/<file>.pdf`` path
    relative to the library root: a value ending in ``.pdf`` is used verbatim;
    anything else is a folder under which a timestamped filename is appended.
    """
    clean = dest.strip().strip("/")
    if clean.lower().endswith(".pdf"):
        return clean
    filename = _default_filename(report)
    return f"{clean}/{filename}" if clean else filename


def publish_report_pdf(
    report: ValidationReport | str,
    sharepoint_path: str | None = None,
    *,
    client: "SharePointClient | None" = None,
    css: str | None = None,
) -> dict[str, Any]:
    """
    Render *report* to PDF and upload it to a SharePoint document library.

    Parameters
    ----------
    sharepoint_path : str | None
        Destination in the library. A value ending in ``.pdf`` is the exact
        file path; otherwise it names a folder and a timestamped filename is
        appended. ``None`` resolves config.json
        ``validation.report_sharepoint_path`` (then the library root).
    client : SharePointClient | None
        A pre-built client (tests inject a fake). ``None`` uses the shared
        :func:`app_config.sharepoint.get_sharepoint_client`.
    css : str | None
        Passed through to :func:`report_to_pdf`.

    Returns
    -------
    dict
        The uploaded drive item (``name`` / ``id`` / ``web_url`` / ...), exactly
        as :meth:`app_config.sharepoint.SharePointClient.write_file` returns it.
    """
    dest = app_config.resolve(
        sharepoint_path, "validation", "report_sharepoint_path", ""
    )
    target = _resolve_sharepoint_path(dest, report)
    pdf = report_to_pdf(report, css=css)
    if client is None:
        from app_config.sharepoint import get_sharepoint_client

        client = get_sharepoint_client()
    logger.info(
        f"publish_report_pdf: uploading {len(pdf)} bytes to SharePoint '{target}'"
    )
    item = client.write_file(target, pdf)
    logger.info(
        f"publish_report_pdf: uploaded to {item.get('web_url') or target!r}"
    )
    return item
