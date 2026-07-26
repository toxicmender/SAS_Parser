"""Render pipeline outputs as Jupyter notebooks (nbformat v4.5).

The pipeline's deliverable is code, so it ships as something a person can open
and run. This module turns the list of output dicts ``SasLLMPipeline._process``
returns into ``.ipynb`` files — **one per SAS source file**, in the dependency
order the batcher established, plus a shared ``_cross_file.ipynb`` for batches
whose members span several files (those cannot belong to any single program's
notebook, and duplicating them would run them twice).

Two paths produce the cells, in order of preference:

- :func:`document_to_cells` — the structured path. The pipeline asked for a
  :class:`~chunker.response_models.TranslationDocument`, so the code cells are
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

Logger name: ``chunker.notebook``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

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

# Fence info strings that are NOT target-language code: the echoed SAS source
# and plain prose/log blocks. Everything else in ## Translation (empty info
# included — an untagged fence in a translation is the translation) is code.
_NON_TARGET_INFO_STRINGS = {"sas", "text", "txt", "log", "output", "console"}

# output_language -> (kernelspec, language_info). Keyed on a lowercased,
# punctuation-stripped form of the name so "Spark SQL", "spark-sql", and
# "SparkSQL" all land together.
_PYTHON_KERNEL = (
    {"name": "python3", "display_name": "Python 3", "language": "python"},
    {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
)
_SQL_KERNEL = (
    {"name": "sql", "display_name": "SQL", "language": "sql"},
    {"name": "sql", "file_extension": ".sql", "mimetype": "application/sql"},
)
_SCALA_KERNEL = (
    {"name": "scala", "display_name": "Scala", "language": "scala"},
    {"name": "scala", "file_extension": ".scala", "mimetype": "text/x-scala"},
)
_KERNELS = {
    "pyspark": _PYTHON_KERNEL,
    "python": _PYTHON_KERNEL,
    "python3": _PYTHON_KERNEL,
    "py": _PYTHON_KERNEL,
    "sparksql": _SQL_KERNEL,
    "sql": _SQL_KERNEL,
    "sparkscala": _SCALA_KERNEL,
    "scala": _SCALA_KERNEL,
}
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _normalise_language(name: str | None) -> str:
    return _PUNCT_RE.sub("", (name or "").lower())


def _kernel_for(output_language: str | None) -> tuple[dict, dict]:
    """Kernelspec + language_info for *output_language*.

    Unknown targets fall back to Python 3: a notebook that opens everywhere and
    reports one wrong language beats one no front-end will start.
    """
    key = _normalise_language(output_language)
    kernel = _KERNELS.get(key)
    if kernel is None:
        logger.debug(
            f"_kernel_for: no kernel mapping for {output_language!r}; using python3"
        )
        return _PYTHON_KERNEL
    return kernel


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


def document_to_cells(doc: TranslationDocument) -> list[dict[str, Any]]:
    """Cells for one structured :class:`TranslationDocument`.

    Analysis, Mapping, and Risks become markdown cells; every
    :class:`~chunker.response_models.TranslationCell` becomes a cell of its own
    kind, in order, so the notebook runs the translation as the model sequenced
    it.
    """
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

    for cell in doc.cells:
        cells.extend(_translation_cell_to_cells(cell))

    if doc.risks:
        lines = ["### Risks", ""]
        for risk in doc.risks:
            marker = "⚠️ " if risk.severity == "P0" else ""
            lines.append(f"- {marker}**{risk.severity}** — {risk.note.strip()}")
        cells.append(markdown_cell("\n".join(lines)))
    return cells


def _translation_cell_to_cells(cell: TranslationCell) -> list[dict[str, Any]]:
    """One :class:`TranslationCell` — plus its comment heading, if any."""
    out: list[dict[str, Any]] = []
    if cell.comment.strip():
        out.append(markdown_cell(f"#### {cell.comment.strip()}"))
    if not cell.source.strip():
        return out
    if cell.kind == "code":
        out.append(code_cell(cell.source, language=cell.language))
    else:
        out.append(markdown_cell(cell.source))
    return out


def markdown_to_cells(response: str) -> list[dict[str, Any]]:
    """Cells for an unstructured Markdown *response* — the fallback path.

    Only the ``## Translation`` section yields code cells: a fenced block in
    Analysis is illustration, and promoting it would put code in the notebook
    that was never meant to run. When the response carries no ``## Translation``
    heading at all (the model deviated from the prompt) the whole response is
    treated as the translation region, which is the safer failure: the code is
    at least present and runnable.
    """
    sections = _split_sections(response)
    if not sections:
        return _cells_from_translation_region(response)

    cells: list[dict[str, Any]] = []
    saw_translation = False
    for heading, body in sections:
        if heading.strip().lower().startswith("translation"):
            saw_translation = True
            if heading.strip():
                cells.append(markdown_cell(f"### {heading.strip()}"))
            cells.extend(_cells_from_translation_region(body))
        elif body.strip() or heading.strip():
            title = f"### {heading.strip()}\n\n" if heading.strip() else ""
            cells.append(markdown_cell(f"{title}{body.strip()}"))
    if not saw_translation:
        logger.debug(
            "markdown_to_cells: no '## Translation' section; treating the whole "
            "response as translation"
        )
        return _cells_from_translation_region(response)
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


def _cells_from_translation_region(text: str) -> list[dict[str, Any]]:
    """Walk *text*, turning fenced target-language blocks into code cells.

    Prose between the fences becomes markdown cells; blocks whose info string
    marks them as source echo (``sas``) or plain text stay inside their prose
    cell, fence and all.
    """
    cells: list[dict[str, Any]] = []
    prose_start = 0

    def flush(end: int) -> None:
        chunk = text[prose_start:end]
        if chunk.strip():
            cells.append(markdown_cell(chunk))

    for match in _FENCE_RE.finditer(text):
        info = match.group(1).strip().lower()
        if info in _NON_TARGET_INFO_STRINGS:
            continue  # leave it in the surrounding prose cell
        flush(match.start())
        cells.append(code_cell(match.group(2), language=info or None))
        prose_start = match.end()
    flush(len(text))
    return cells


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build_notebook(
    cells: Iterable[dict[str, Any]],
    *,
    output_language: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap *cells* into a complete nbformat v4.5 notebook.

    Cell ids are stamped here, where the whole cell list is visible: v4.5
    requires each cell to carry an id unique within the notebook and matching
    ``^[a-zA-Z0-9-_]+$``. They are positional (``cell-0007``) rather than
    random so re-running the same pipeline output produces a byte-identical
    notebook and diffs stay readable.
    """
    kernelspec, language_info = _kernel_for(output_language)
    stamped = []
    for index, cell in enumerate(cells):
        stamped.append({**cell, "id": f"cell-{index:04d}"})
    nb_metadata: dict[str, Any] = {
        "kernelspec": dict(kernelspec),
        "language_info": dict(language_info),
    }
    if output_language:
        nb_metadata["sas_parser"] = {"output_language": output_language}
    if metadata:
        nb_metadata.update(metadata)
    return {
        "cells": stamped,
        "metadata": nb_metadata,
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


def notebook_to_json(notebook: dict[str, Any]) -> str:
    """*notebook* as the JSON text of an ``.ipynb`` file (trailing newline)."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Pipeline outputs -> notebooks
# ---------------------------------------------------------------------------


def item_cells(out: dict[str, Any]) -> list[dict[str, Any]]:
    """Header + body cells for one pipeline output dict.

    Prefers the structured ``document`` the pipeline attached; falls back to
    parsing ``response`` when there is none.
    """
    cells = [markdown_cell(_item_header_markdown(out))]
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
            return cells + document_to_cells(doc)
    return cells + markdown_to_cells(str(out.get("response", "")))


def _item_header_markdown(out: dict[str, Any]) -> str:
    """The ``## <item_id>`` banner cell: what this item is and how it scored."""
    kind = "batch" if out.get("is_batch") else (out.get("kind") or "chunk")
    lines = [f"## {out.get('item_id', 'item')}", "", f"- Kind: `{kind}`"]
    sources = out.get("source_files") or []
    if sources:
        lines.append(f"- Source file(s): {', '.join(f'`{s}`' for s in sources)}")
    chunk_ids = out.get("chunk_ids") or []
    if chunk_ids:
        lines.append(f"- Chunk(s): {', '.join(f'`{c}`' for c in chunk_ids)}")
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
    outputs: list[dict[str, Any]], *, output_language: str | None = None
) -> dict[str, dict[str, Any]]:
    """Group *outputs* into ``{filename_stem: notebook}``.

    *outputs* is consumed in order — the batcher's dependency order — so each
    notebook's cells run in an order that respects the dependencies discovered
    across the corpus. A cross-file batch goes into ``_cross_file`` once, and
    every participating file's notebook gets a pointer to it in its place, so a
    reader following one program never silently skips a step.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    names: dict[str, str] = {}
    for out in outputs:
        key = _notebook_key(out)
        name = _filename_for(key, names)
        grouped.setdefault(name, []).extend(item_cells(out))
        if key == CROSS_FILE_NOTEBOOK:
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
        name: build_notebook(cells, output_language=output_language)
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
