"""Render pipeline outputs as Jupyter notebooks (nbformat v4.5).

The pipeline's deliverable is code, so it ships as something a person can open
and run. This module turns the list of output dicts ``SasLLMPipeline._process``
returns into ``.ipynb`` files — **one per SAS source file**, in the dependency
order the batcher established.

A batch whose members span several files is **split back per source file**
when its structured document allows it: every code cell tagged with the
``chunk_id`` it implements (the model is asked to tag them; see
``pipeline.constants``) routes to that chunk's source-file notebook via the
output's ``chunk_sources`` map. The split is all-or-nothing per item — if any
code cell is untagged or unresolvable, or the item has no structured document
at all (the Markdown-fallback path), the **whole item** falls back to the
shared ``_cross_file.ipynb`` with a pointer cell in each participating
notebook, exactly the pre-split behavior; scattering one item's translation
across both worlds would be worse than either. Ordering caveat: each
notebook's cells respect the corpus order (producers precede consumers
across the whole run), but when two files' steps interleave through shared
batches, running one notebook end to end before the other is only safe if
its steps do not depend on the other's later outputs — the per-item header
names the sibling files, so the coupling stays visible.

Two paths produce the cells, in order of preference:

- :func:`document_to_cells` — the structured path. The pipeline asked for a
  :class:`~pipeline.response_models.TranslationDocument`, so the code cells are
  exactly the cells the model nominated; no guessing.
- :func:`markdown_to_cells` — the fallback, used whenever structured output was
  off, unsupported by the gateway, or failed to parse. It splits the four-section
  Markdown response: fenced blocks inside ``## Translation`` become code cells,
  everything else stays prose.

The format is written directly against the nbformat v4.5 spec rather than
through the ``nbformat`` package: the schema is small and stable, and the
runtime should not grow a jupyter dependency to write JSON. ``nbformat`` is in
the ``dev`` extra, where ``tests/test_notebook.py`` validates what we emit
against the real schema.

Logger name: ``pipeline.notebook``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from target_language import (
    PROSE_FENCE_INFOS,
    PYSPARK,
    SPARKSQL,
    TargetLanguage,
    normalize_language,
    resolve_target_language,
)

from .response_models import TranslationCell, TranslationDocument

logger = logging.getLogger(__name__)

NBFORMAT = 4
NBFORMAT_MINOR = 5

# The notebook every cross-file batch lands in (its members belong to no single
# source file). Leading underscore so it sorts apart from the real programs.
CROSS_FILE_NOTEBOOK = "_cross_file"

# Fenced block: info string (may be empty) + body. Mirrors
# validation.metrics._FENCE_RE so the fallback parser sees the same blocks the
# metrics score.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# A "## " section heading at the start of a line.
_SECTION_RE = re.compile(r"^## +(.+?) *$", re.MULTILINE)

def _target_for(output_language: str | TargetLanguage | None) -> TargetLanguage:
    """The target these notebooks are being written for.

    Callers hand over whatever they hold — the pipeline's resolved
    :class:`TargetLanguage`, a name from a CLI, or nothing. An unrecognised
    name does *not* raise here: by the time a notebook is being written the
    translation has been paid for, so it is written with PySpark's kernel and
    a warning rather than lost. (The pipeline itself rejects the name at
    construction, long before any call is made — see
    ``target_language.resolve_target_language``.)
    """
    if isinstance(output_language, TargetLanguage):
        return output_language
    return resolve_target_language(output_language)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------


def markdown_cell(source: str) -> dict[str, Any]:
    """A v4.5 markdown cell (the ``id`` is stamped by :func:`build_notebook`)."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")}


def code_cell(source: str, *, language: str | None = None) -> dict[str, Any]:
    """A v4.5 code cell.

    *language* is recorded in the cell metadata so a notebook that mixes, say,
    SQL and Python still highlights each cell correctly — the notebook-level
    ``language_info`` can only name one.
    """
    metadata: dict[str, Any] = {}
    if language:
        metadata["language"] = language
        metadata["vscode"] = {"languageId": language}
    return {
        "cell_type": "code",
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": source.strip("\n"),
    }


def document_to_cells(
    doc: TranslationDocument,
    *,
    output_language: str | TargetLanguage | None = None,
) -> list[dict[str, Any]]:
    """Cells for one structured :class:`TranslationDocument`.

    Analysis, Mapping, and Risks become markdown cells; every
    :class:`~pipeline.response_models.TranslationCell` becomes a cell of its own
    kind, in order, so the notebook runs the translation as the model sequenced
    it.

    *output_language* supplies the language a code cell is tagged with when
    the model left ``TranslationCell.language`` unset — the run's target,
    rather than the flat "python" assumption that predates it.
    """
    target = _target_for(output_language)
    cells = _document_prelude_cells(doc)
    for cell in doc.cells:
        cells.extend(_translation_cell_to_cells(cell, target))
    cells.extend(_document_risks_cells(doc))
    return cells


def _document_prelude_cells(doc: TranslationDocument) -> list[dict[str, Any]]:
    """The Analysis + Mapping markdown cells (0–2), shared by the whole-item
    and per-source-split rendering paths."""
    cells: list[dict[str, Any]] = []
    if doc.analysis.strip():
        cells.append(markdown_cell(f"### Analysis\n\n{doc.analysis.strip()}"))
    if doc.mapping:
        lines = ["### Mapping", ""]
        for entry in doc.mapping:
            text = f"- **{entry.sas_construct}** → {entry.equivalent}"
            if entry.difference.strip():
                text += f" — {entry.difference.strip()}"
            lines.append(text)
        cells.append(markdown_cell("\n".join(lines)))
    return cells


def _document_risks_cells(doc: TranslationDocument) -> list[dict[str, Any]]:
    """The Risks markdown cell (0–1) — counterpart of the prelude."""
    if not doc.risks:
        return []
    lines = ["### Risks", ""]
    for risk in doc.risks:
        marker = "⚠️ " if risk.severity == "P0" else ""
        lines.append(f"- {marker}**{risk.severity}** — {risk.note.strip()}")
    return [markdown_cell("\n".join(lines))]


def _translation_cell_to_cells(
    cell: TranslationCell, target: TargetLanguage
) -> list[dict[str, Any]]:
    """One :class:`TranslationCell` — plus its comment heading, if any.

    A code cell that names no language is tagged with *target*'s. One that
    names a different language keeps its own tag — the cell is highlighted as
    what it actually is, and the ``language_compliance`` metric is what fails
    the item — but it is logged, because a notebook quietly mixing languages
    is how an off-target translation used to ship unnoticed.
    """
    out: list[dict[str, Any]] = []
    if cell.comment.strip():
        out.append(markdown_cell(f"#### {cell.comment.strip()}"))
    if not cell.source.strip():
        return out
    if cell.kind == "code":
        language = cell.language or target.cell_language
        if not target.owns_fence(language):
            logger.warning(
                f"_translation_cell_to_cells: cell declares language "
                f"{cell.language!r}, but this run targets "
                f"{target.display_name}; keeping the declared one"
            )
        out.append(code_cell(cell.source, language=language))
    else:
        out.append(markdown_cell(cell.source))
    return out


def markdown_to_cells(
    response: str,
    *,
    output_language: str | TargetLanguage | None = None,
) -> list[dict[str, Any]]:
    """Cells for an unstructured Markdown *response* — the fallback path.

    Only the ``## Translation`` section yields code cells: a fenced block in
    Analysis is illustration, and promoting it would put code in the notebook
    that was never meant to run. When the response carries no ``## Translation``
    heading at all (the model deviated from the prompt) the whole response is
    treated as the translation region, which is the safer failure: the code is
    at least present and runnable.

    *output_language* is the run's target: it decides which fences are code
    and what an untagged one is tagged as.
    """
    target = _target_for(output_language)
    sections = _split_sections(response)
    if not sections:
        return _cells_from_translation_region(response, target)

    cells: list[dict[str, Any]] = []
    saw_translation = False
    for heading, body in sections:
        if heading.strip().lower().startswith("translation"):
            saw_translation = True
            if heading.strip():
                cells.append(markdown_cell(f"### {heading.strip()}"))
            cells.extend(_cells_from_translation_region(body, target))
        elif body.strip() or heading.strip():
            title = f"### {heading.strip()}\n\n" if heading.strip() else ""
            cells.append(markdown_cell(f"{title}{body.strip()}"))
    if not saw_translation:
        logger.debug(
            "markdown_to_cells: no '## Translation' section; treating the whole "
            "response as translation"
        )
        return _cells_from_translation_region(response, target)
    return cells


def _split_sections(response: str) -> list[tuple[str, str]]:
    """``[(heading, body), ...]`` for the ``## `` sections of *response*.

    Text before the first heading is returned under an empty heading. Returns
    ``[]`` when the response has no headings at all. Headings *inside* fenced
    blocks are ignored — a ``## `` line in a code block is a comment, not a
    section.
    """
    masked = _mask_fences(response)
    matches = list(_SECTION_RE.finditer(masked))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    preamble = response[: matches[0].start()]
    if preamble.strip():
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        sections.append((match.group(1), response[match.end() : end]))
    return sections


def _mask_fences(text: str) -> str:
    """*text* with fenced-block bodies blanked out, offsets preserved."""
    return _FENCE_RE.sub(lambda m: " " * len(m.group(0)), text)


def _cells_from_translation_region(
    text: str, target: TargetLanguage
) -> list[dict[str, Any]]:
    """Walk *text*, turning fenced code blocks into code cells.

    Prose between the fences becomes markdown cells; blocks whose info string
    marks them as source echo (``sas``) or plain text stay inside their prose
    cell, fence and all.

    A block in some *other* programming language still becomes a code cell,
    tagged as what it is: the translation has already been paid for, and a
    notebook that runs the wrong language is more useful — and more obviously
    wrong — than one silently missing its translation. ``language_compliance``
    is what fails the item; this only logs.
    """
    cells: list[dict[str, Any]] = []
    prose_start = 0

    def flush(end: int) -> None:
        chunk = text[prose_start:end]
        if chunk.strip():
            cells.append(markdown_cell(chunk))

    for match in _FENCE_RE.finditer(text):
        info = match.group(1).strip().lower()
        if normalize_language(info) in PROSE_FENCE_INFOS:
            continue  # leave it in the surrounding prose cell
        if not target.owns_fence(info):
            logger.warning(
                f"_cells_from_translation_region: ```{info} block in a "
                f"{target.display_name} translation; emitting it as a "
                f"{info} cell"
            )
        flush(match.start())
        cells.append(code_cell(match.group(2), language=info or target.cell_language))
        prose_start = match.end()
    flush(len(text))
    return cells


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build_notebook(
    cells: Iterable[dict[str, Any]],
    *,
    output_language: str | TargetLanguage | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap *cells* into a complete nbformat v4.5 notebook.

    Cell ids are stamped here, where the whole cell list is visible: v4.5
    requires each cell to carry an id unique within the notebook and matching
    ``^[a-zA-Z0-9-_]+$``. They are positional (``cell-0007``) rather than
    random so re-running the same pipeline output produces a byte-identical
    notebook and diffs stay readable.
    """
    target = _target_for(output_language)
    cells = _host_mixed_languages(list(cells), target)
    # A notebook whose cells are not all the target's needs a kernel that can
    # host them. Python can run SQL through the %sql magic _host_mixed_languages
    # adds; a SQL kernel cannot run Python at all, so the mixed case is always
    # hosted by Python regardless of which target the run asked for.
    kernel_target = PYSPARK if _has_language(cells, PYSPARK) else target
    stamped = []
    for index, cell in enumerate(cells):
        stamped.append({**cell, "id": f"cell-{index:04d}"})
    nb_metadata: dict[str, Any] = {
        "kernelspec": dict(kernel_target.kernelspec),
        "language_info": dict(kernel_target.language_info),
    }
    if output_language:
        # The canonical name, not the caller's spelling: a notebook records
        # which target it was written for, and "SparkSQL"/"spark sql" are the
        # same one.
        nb_metadata["sas_parser"] = {"output_language": target.display_name}
    if metadata:
        nb_metadata.update(metadata)
    return {
        "cells": stamped,
        "metadata": nb_metadata,
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


#: The Databricks cell magic that runs one cell as SQL inside a Python
#: notebook. Only ever prepended in a mixed notebook — see
#: :func:`_host_mixed_languages`.
_SQL_MAGIC = "%sql"


def _has_language(cells: list[dict[str, Any]], target: TargetLanguage) -> bool:
    """True when any code cell is tagged as *target*'s language."""
    return any(
        cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("language") == target.cell_language
        for cell in cells
    )


def _host_mixed_languages(
    cells: list[dict[str, Any]], target: TargetLanguage
) -> list[dict[str, Any]]:
    """*cells* made runnable together when they are not all one language.

    A Spark SQL run whose items fell back to PySpark (see
    :mod:`complexity.fallback`) produces both kinds of code cell. The notebook
    is hosted by the Python kernel, so its SQL cells are prefixed with the
    ``%sql`` magic that tells Databricks to run them as SQL.

    Returns *cells* **unchanged** when they are not mixed, which is the case for
    every run today: an all-SQL or all-Python notebook is byte-identical to what
    this module produced before the fallback existed, so nothing about the
    common path moves.
    """
    if not (_has_language(cells, PYSPARK) and _has_language(cells, SPARKSQL)):
        return cells
    out: list[dict[str, Any]] = []
    magicked = 0
    for cell in cells:
        metadata = cell.get("metadata", {})
        source = str(cell.get("source", ""))
        if (
            cell.get("cell_type") == "code"
            and metadata.get("language") == SPARKSQL.cell_language
            and not source.lstrip().startswith(_SQL_MAGIC)
        ):
            out.append({**cell, "source": f"{_SQL_MAGIC}\n{source}"})
            magicked += 1
        else:
            out.append(cell)
    logger.info(
        f"_host_mixed_languages: notebook mixes {SPARKSQL.display_name} and "
        f"{PYSPARK.display_name}; hosting it on the Python kernel and marking "
        f"{magicked} SQL cell(s) with {_SQL_MAGIC}"
    )
    return out


def notebook_to_json(notebook: dict[str, Any]) -> str:
    """*notebook* as the JSON text of an ``.ipynb`` file (trailing newline)."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Pipeline outputs -> notebooks
# ---------------------------------------------------------------------------


def item_cells(
    out: dict[str, Any],
    *,
    output_language: str | TargetLanguage | None = None,
) -> list[dict[str, Any]]:
    """Header + body cells for one pipeline output dict.

    Prefers the structured ``document`` the pipeline attached; falls back to
    parsing ``response`` when there is none.

    *output_language* is the **run's** target. The item's own wins when it has
    one (``target_language``, set for every output by
    :func:`pipeline.engine._target_fields`): an item that fell back to PySpark
    is tagged, fenced and highlighted as PySpark, and its untagged cells default
    to PySpark rather than to the run's SQL. Reading the run's target here is
    what would put a ```sql``` fence around Python.
    """
    cells = [markdown_cell(_item_header_markdown(out))]
    language = out.get("target_language") or output_language
    document = out.get("document")
    if document:
        try:
            doc = TranslationDocument.model_validate(document)
        except Exception:
            logger.warning(
                f"item_cells: item={out.get('item_id')!r} has an unreadable "
                "document; falling back to the Markdown response",
                exc_info=True,
            )
        else:
            return cells + document_to_cells(doc, output_language=language)
    return cells + markdown_to_cells(
        str(out.get("response", "")), output_language=language
    )


def _item_header_markdown(out: dict[str, Any]) -> str:
    """The ``## <item_id>`` banner cell: what this item is and how it scored."""
    # Every item the pipeline emits is a batch (singletons arrive wrapped),
    # so there is nothing to report here that the source list does not already
    # say — the "Kind: batch" line it used to carry was a constant.
    lines = [f"## {out.get('item_id', 'item')}", ""]
    sources = out.get("source_files") or []
    if sources:
        lines.append(f"- Source file(s): {', '.join(f'`{s}`' for s in sources)}")
    chunk_ids = out.get("chunk_ids") or []
    if chunk_ids:
        lines.append(f"- Chunk(s): {', '.join(f'`{c}`' for c in chunk_ids)}")
    # Only when the item did not stay on the run's target: a reader looking at
    # Python in a Spark SQL notebook needs to know it was deliberate, and what
    # forced it. An item that kept the run's target says nothing, because the
    # notebook already records the run's language.
    reasons = out.get("fallback_reasons") or []
    if reasons:
        lines.append(
            f"- **Translated to {out.get('target_language', 'another target')}**: "
            f"{'; '.join(reasons)}"
        )
    verdict = out.get("validation")
    if verdict:
        status = "PASS" if verdict.get("passed") else "FAIL"
        lines.append(f"- Validation: **{status}** (score {verdict.get('score', 0):.2f})")
        failed = [
            m["metric"]
            for m in verdict.get("metrics", [])
            if not m.get("passed") and not m.get("skipped")
        ]
        if failed:
            lines.append(f"- Failing metrics: {', '.join(failed)}")
    return "\n".join(lines)


def _notebook_key(out: dict[str, Any]) -> str:
    """Which notebook this item belongs in: its source file, or the shared one."""
    sources = [s for s in (out.get("source_files") or []) if s]
    if len(sources) == 1:
        return sources[0]
    return CROSS_FILE_NOTEBOOK


def _split_item_by_source(
    out: dict[str, Any], target: TargetLanguage
) -> dict[str, list[dict[str, Any]]] | None:
    """Route one multi-source item's cells per source file, or ``None``.

    ``None`` means "cannot split cleanly" and the caller falls back to the
    shared ``_cross_file`` notebook for the **whole item** — the split is
    all-or-nothing: every *code* cell must carry a ``chunk_id`` that resolves
    through the output's ``chunk_sources`` map to one of the item's source
    files. Prose is forgiving: an untagged (or unresolvable) markdown cell is
    duplicated into every participating notebook, like the document-level
    Analysis / Mapping / Risks cells, so each file's notebook stands alone.
    """
    document = out.get("document")
    chunk_sources = out.get("chunk_sources")
    if not document or not isinstance(chunk_sources, dict) or not chunk_sources:
        return None
    try:
        doc = TranslationDocument.model_validate(document)
    except Exception:
        logger.warning(
            f"_split_item_by_source: item={out.get('item_id')!r} has an "
            "unreadable document; falling back to the shared notebook",
            exc_info=True,
        )
        return None
    sources = [s for s in (out.get("source_files") or []) if s]

    # Resolve every cell first (all-or-nothing), then render.
    routed: list[tuple[Any, str | None]] = []
    for cell in doc.cells:
        source = chunk_sources.get(cell.chunk_id) if cell.chunk_id else None
        if source is not None and source not in sources:
            source = None  # a tag pointing outside the item resolves nowhere
        if cell.kind == "code" and source is None:
            logger.info(
                f"_split_item_by_source: item={out.get('item_id')!r} has a "
                f"code cell with no resolvable chunk_id "
                f"({cell.chunk_id!r}); keeping the whole item in "
                f"'{CROSS_FILE_NOTEBOOK}'"
            )
            return None
        routed.append((cell, source))

    header = markdown_cell(
        _item_header_markdown(out)
        + "\n- Split: translation cells are routed into each source file's "
        "notebook by their `chunk_id`"
    )
    shared_prelude = [header, *_document_prelude_cells(doc)]
    per_source: dict[str, list[dict[str, Any]]] = {
        source: list(shared_prelude) for source in sources
    }
    for cell, source in routed:
        rendered = _translation_cell_to_cells(cell, target)
        if source is None:  # untagged prose: every participating notebook
            for cells in per_source.values():
                cells.extend(rendered)
        else:
            per_source[source].extend(rendered)
    for cells in per_source.values():
        cells.extend(_document_risks_cells(doc))
    return per_source


def _filename_for(key: str, taken: dict[str, str]) -> str:
    """A unique, filesystem-safe ``.ipynb`` stem for source id *key*.

    Two programs with the same basename in different directories would collide
    on stem alone, so a suffix is appended to the later one rather than letting
    one silently overwrite the other.
    """
    if key in taken:
        return taken[key]
    if key == CROSS_FILE_NOTEBOOK:
        # Not a source id — a fixed name, and its leading underscore is the
        # point (it sorts apart from the programs), so skip the sanitiser that
        # would strip it.
        taken[key] = CROSS_FILE_NOTEBOOK
        return CROSS_FILE_NOTEBOOK
    stem = Path(key).stem or "translation"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "translation"
    candidate = stem
    used = set(taken.values())
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    taken[key] = candidate
    return candidate


def notebooks_from_outputs(
    outputs: list[dict[str, Any]],
    *,
    output_language: str | TargetLanguage | None = None,
) -> dict[str, dict[str, Any]]:
    """Group *outputs* into ``{filename_stem: notebook}``.

    *outputs* is consumed in order — the batcher's dependency order — so each
    notebook's cells run in an order that respects the dependencies discovered
    across the corpus. A multi-source batch is **split per source file** when
    its document's cells are cleanly attributed (see
    :func:`_split_item_by_source`); otherwise it goes into ``_cross_file``
    once, and every participating file's notebook gets a pointer to it in its
    place, so a reader following one program never silently skips a step.
    """
    # Resolved once for the whole set: every notebook in a run targets the
    # same language, and re-resolving per item would re-log the same warning.
    target = _target_for(output_language)
    grouped: dict[str, list[dict[str, Any]]] = {}
    names: dict[str, str] = {}
    for out in outputs:
        key = _notebook_key(out)
        if key != CROSS_FILE_NOTEBOOK:
            name = _filename_for(key, names)
            grouped.setdefault(name, []).extend(
                item_cells(out, output_language=target)
            )
            continue
        per_source = _split_item_by_source(out, target)
        if per_source is not None:
            for source, cells in per_source.items():
                name = _filename_for(source, names)
                grouped.setdefault(name, []).extend(cells)
            continue
        name = _filename_for(key, names)
        grouped.setdefault(name, []).extend(item_cells(out, output_language=target))
        item_id = out.get("item_id", "batch")
        for source in out.get("source_files") or []:
            pointer = _filename_for(source, names)
            grouped.setdefault(pointer, []).append(
                markdown_cell(
                    f"## {item_id} (cross-file)\n\n"
                    f"This step spans {len(out.get('source_files') or [])} "
                    f"source files and is translated once in "
                    f"`{CROSS_FILE_NOTEBOOK}.ipynb`. Run it there, at this "
                    "point in the sequence."
                )
            )

    notebooks = {
        name: build_notebook(
            cells, output_language=target if output_language else None
        )
        for name, cells in grouped.items()
    }
    logger.info(
        f"notebooks_from_outputs: {len(outputs)} item(s) -> {len(notebooks)} "
        f"notebook(s): {', '.join(sorted(notebooks))}"
    )
    return notebooks


def write_notebooks(
    outputs: list[dict[str, Any]],
    out_dir: Path | str,
    *,
    output_language: str | None = None,
) -> list[Path]:
    """Write one ``.ipynb`` per source file under *out_dir*; return the paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, notebook in notebooks_from_outputs(
        outputs, output_language=output_language
    ).items():
        dest = directory / f"{name}.ipynb"
        dest.write_text(notebook_to_json(notebook), encoding="utf-8")
        logger.debug(f"write_notebooks: wrote {dest}")
        written.append(dest)
    logger.info(f"write_notebooks: wrote {len(written)} notebook(s) to {directory}")
    return written
