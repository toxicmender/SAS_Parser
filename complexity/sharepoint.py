"""The SharePoint complexity request list, and where a run's reports go.

The list is a *request* list — ``ID``, ``Application``, ``Output_Language``,
``Preferred_LLM`` — and has no result columns. So complexity output is
delivered **only** as uploaded artefacts, and this module never writes to the
list at all.

Read-only, and why that simplifies things
-----------------------------------------
* **No status write-back**, so ``update_list_item`` is not a dependency here
  the way it is for :mod:`conversion` — this flow works against the read half
  of the transport alone.
* **No "pending" concept.** Every row is a valid target on every invocation;
  a run is explicitly triggered, so "which rows are outstanding" was never a
  question this flow had to answer.
* **Idempotence is structural.** Every run lands in a fresh
  ``{label}/{timestamp}`` folder, so re-running is non-destructive by
  construction and needs no marker to make it so.
* **Reporting replaces the column.** A ``run-summary.md`` is uploaded beside
  the reports, so an operator reading SharePoint sees the outcome — including
  a failure — where the artefacts are, rather than only in the logs.

This module lives inside ``complexity/`` rather than beside ``conversion/``
because it depends on :class:`~complexity.models.CorpusComplexityReport` and
nothing depends on it. ``conversion`` and ``xref`` are top-level because the
pipeline consumes both.

**Sourcing the scripts is not here.** It is the same folder and the same
extension set a conversion run reads, so callers use
:func:`conversion.sources.source_files` and :func:`conversion.sources.load`
directly. This module used to re-export both, which only hid the dependency:
the import graph said ``complexity -> conversion`` either way, and a wrapper
that adds nothing is a place for the two to drift apart.

Logger name: ``complexity.sharepoint``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app_config.sharepoint import (
    SharePointConfig,
    SharePointError,
    field_text,
    project_rows,
    resolve_client,
    resolve_config,
)

logger = logging.getLogger(__name__)

# attribute -> SharePoint internal column name. Four columns, all read.
COMPLEXITY_FIELDS: dict[str, str] = {
    "application": "Application",
    "output_language": "Output_Language",
    "preferred_llm": "Preferred_LLM",
}

COMPLEXITY_FOLDER = "complexity"
RUN_SUMMARY_NAME = "run-summary.md"


@dataclass
class ComplexityRequest:
    """One row of the complexity list.

    Attributes
    ----------
    item_id : str
        The list item's ``ID`` — how ``--item-id`` selects a single row.
    application : str
        Which application to score; also the folder its scripts live under.
    output_language : str | None
        The target, which picks the rules profile (``--target``).
    preferred_llm : str | None
        A model in the gateway's ``"provider: model"`` spelling. Setting it
        implies ``--llm-eval``: naming a model is how the row asks for the
        second opinion.
    """

    item_id: str
    application: str
    output_language: str | None = None
    preferred_llm: str | None = None


def format_complexity_item_params(item: dict[str, Any]) -> ComplexityRequest:
    """
    One raw list item (``{id, web_url, fields}``) as a
    :class:`ComplexityRequest`.

    Raises
    ------
    SharePointError
        The row names no application, which is the one field the run cannot
        proceed without.
    """
    fields = item.get("fields") or {}
    application = field_text(fields, COMPLEXITY_FIELDS["application"])
    if not application:
        raise SharePointError(
            f"complexity row {item.get('id')!r} has no "
            f"{COMPLEXITY_FIELDS['application']!r}; it names the folder to "
            f"score, so the row cannot be processed"
        )
    return ComplexityRequest(
        item_id=str(item.get("id") or ""),
        application=application,
        output_language=field_text(fields, COMPLEXITY_FIELDS["output_language"]),
        preferred_llm=field_text(fields, COMPLEXITY_FIELDS["preferred_llm"]),
    )


def requests(
    *,
    application: str | None = None,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> list[ComplexityRequest]:
    """
    The rows of the complexity list, optionally narrowed to one *application*.

    Every row is a valid target — there is no status to filter on. A row
    without an application is skipped with a WARNING rather than failing the
    read, so one bad row does not hide the rest.

    Raises
    ------
    SharePointError
        The complexity list is not configured, or the read failed.
    """
    resolved = resolve_config(config)
    rows = resolve_client(client).list_items(resolved.list_id("complexity"))
    out = project_rows(rows, format_complexity_item_params, label="requests")
    if application is not None:
        wanted = application.strip().casefold()
        out = [row for row in out if row.application.casefold() == wanted]
    logger.info(f"requests: {len(out)} complexity request row(s) selected")
    return out


def request(
    item_id: str | int,
    *,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> ComplexityRequest:
    """
    One row of the complexity list, by its ``ID``.

    Raises
    ------
    SharePointError
        The list is not configured, the row is absent, or it names no
        application.
    """
    resolved = resolve_config(config)
    row = resolve_client(client).get_list_item(resolved.list_id("complexity"), item_id)
    return format_complexity_item_params(row)


def report_folder(
    application: str,
    label: str,
    timestamp: str,
    *,
    config: SharePointConfig | None = None,
) -> str:
    """
    Where one run's reports go: ``{base}/{application}/complexity/{label}/
    {timestamp}``.

    *label* records **what produced the estimate** — the model id when
    ``--llm-eval`` ran, else the resolved rules profile. That mirrors
    conversion's ``{model}/{timestamp}`` while still meaning something for an
    entirely offline run, where there is no model to name.
    """
    return resolve_config(config).drive_path(
        application, COMPLEXITY_FOLDER, label, timestamp
    )


def upload_reports(
    application: str,
    label: str,
    timestamp: str,
    paths: list[Path] | list[str],
    *,
    staging_root: Path | str | None = None,
    client: Any | None = None,
    config: SharePointConfig | None = None,
) -> list[str]:
    """
    Upload every staged file in *paths* into this run's report folder,
    returning the drive-relative paths they landed at.

    *staging_root* is the local directory the reports were written under; a
    file's path relative to it becomes its path inside the upload folder, so
    the ``files/`` and ``prompts/`` sub-trees survive the transfer intact.
    Without it, files are uploaded flat by name.

    Text is uploaded as text and everything else (the PDF, the graph PNG) as
    bytes, decided by extension — a PDF read as UTF-8 would not survive.

    Raises
    ------
    SharePointError
        A folder could not be created, or an upload failed.
    """
    folder = report_folder(application, label, timestamp, config=config)
    transport = resolve_client(client)
    transport.create_folder(folder)
    root = Path(staging_root) if staging_root is not None else None

    uploaded: list[str] = []
    created: set[str] = {folder}
    for entry in paths:
        source = Path(entry)
        if not source.is_file():
            logger.warning(f"upload_reports: {source} is not a file; skipping")
            continue
        relative = _relative_name(source, root)
        parent, _, name = relative.rpartition("/")
        destination = f"{folder}/{parent}" if parent else folder
        if destination not in created:
            transport.create_folder(destination)
            created.add(destination)
        transport.upload_file(destination, name, _read(source))
        uploaded.append(f"{destination}/{name}")
    logger.info(f"upload_reports: uploaded {len(uploaded)} file(s) to {folder!r}")
    return uploaded


def _relative_name(source: Path, root: Path | None) -> str:
    """*source*'s path inside the upload folder, as ``a/b/c.md``."""
    if root is None:
        return source.name
    try:
        return source.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        # Outside the staging tree — upload it flat rather than refuse it.
        return source.name


# Extensions read as text; anything else is uploaded as bytes.
_TEXT_SUFFIXES = frozenset({".md", ".txt", ".json", ".csv", ".sql", ".py", ".ipynb"})


def _read(source: Path) -> str | bytes:
    if source.suffix.lower() in _TEXT_SUFFIXES:
        return source.read_text(encoding="utf-8")
    return source.read_bytes()


def render_run_summary(
    request: ComplexityRequest,
    *,
    target: str,
    model: str | None,
    label: str,
    timestamp: str,
    files_scored: int,
    failures: list[str] | None = None,
    seconds: float | None = None,
    exit_status: int = 0,
) -> str:
    """
    The ``run-summary.md`` uploaded beside the reports.

    This is what stands in for a ``Status`` column: the request row's own
    values, what they resolved to, what was scored, what failed, and how the
    run ended — where the artefacts are, rather than only in the logs.
    """
    failed = failures or []
    lines = [
        f"# Complexity run — {request.application}",
        "",
        "## Request",
        "",
        f"- **Item ID**: {request.item_id or '(unknown)'}",
        f"- **Application**: {request.application}",
        f"- **Output_Language**: {request.output_language or '(unset)'}",
        f"- **Preferred_LLM**: {request.preferred_llm or '(unset)'}",
        "",
        "## Resolved",
        "",
        f"- **Target profile**: {target}",
        f"- **Model**: {model or '(rules only, no LLM evaluation)'}",
        f"- **Upload folder**: `{label}/{timestamp}`",
        "",
        "## Outcome",
        "",
        f"- **Files scored**: {files_scored}",
        f"- **Files that failed to chunk**: {len(failed)}",
    ]
    for name in failed:
        lines.append(f"  - `{name}`")
    if seconds is not None:
        lines.append(f"- **Wall time**: {seconds:.1f}s")
    lines.append(
        f"- **Exit status**: {exit_status} "
        f"({'success' if exit_status == 0 else 'failed'})"
    )
    lines.append("")
    return "\n".join(lines)
