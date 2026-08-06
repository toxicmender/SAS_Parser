"""Render a :class:`ValidationReport` to PDF, and publish it to SharePoint.

The report already describes itself as GitHub-flavoured Markdown
(:meth:`ValidationReport.to_markdown`); this module turns that Markdown into a
paginated PDF and, on request, writes the PDF into a SharePoint document
library.

The rendering itself is :func:`reporting.pdf.render_markdown_bytes` — the same
implementation :mod:`complexity` renders through. It used to be a second copy
of the markdown-it → PyMuPDF ``Story`` loop here, with its own stylesheet, no
code folding, no image resolution, and no cap on the paging loop; sharing the
one renderer fixes all four and means the two reports cannot drift apart
again.

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

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import app_config

from .models import ValidationReport

if TYPE_CHECKING:
    from app_config.sharepoint import SharePointClient

logger = logging.getLogger(__name__)


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
        Extra CSS appended after the shared print stylesheet (later rules
        win), for callers that want to restyle the page.
    """
    from reporting.pdf import render_markdown_bytes

    markdown = (
        report.to_markdown() if isinstance(report, ValidationReport) else report
    )
    return render_markdown_bytes(markdown, css=css)


def _report_stamp(report: ValidationReport | str) -> datetime:
    """The report's ``created_at`` (a real report) or now, in UTC."""
    if isinstance(report, ValidationReport):
        stamp = report.created_at
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp
    return datetime.now(timezone.utc)


def _default_filename(report: ValidationReport | str) -> str:
    """A timestamped name, e.g. ``validation-report-20260720T134501Z.pdf``.

    Formatted with :data:`app_config.UTC_STAMP_FORMAT` — the same shape
    :func:`app_config.utc_stamp` produces — because this name sits in the same
    library as the run folders those name, and a reader sorts them together.
    """
    return f"validation-report-{_report_stamp(report):{app_config.UTC_STAMP_FORMAT}}.pdf"


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
