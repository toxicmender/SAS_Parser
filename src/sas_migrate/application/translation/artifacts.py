"""Canonical prompt, response, Markdown, and notebook artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePath
from typing import Any

from pydantic import Field

from sas_migrate.application.ports import ArtifactRepository, ArtifactWrite
from sas_migrate.core.ids import ItemId, RunId
from sas_migrate.core.models import ContractModel
from sas_migrate.core.responses import (
    ResponseEnvelope,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)
from sas_migrate.core.targets import ResolvedTarget, TargetId
from sas_migrate.core.tokens import PromptAssembly

from .models import TranslationItem


class ArtifactLocator(ContractModel):
    artifact_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class NotebookTranslation(ContractModel):
    item: TranslationItem
    target: ResolvedTarget
    document: TranslationDocument
    recovered: bool = False


def _safe(value: str, *, fallback: str = "item") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or fallback
    if len(stem) <= 80:
        return stem
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{stem[:69]}-{digest}"


def render_effective_prompt(
    item: TranslationItem,
    target: ResolvedTarget,
    attempt: int,
    prompt: PromptAssembly,
) -> str:
    lines = [
        f"# Effective prompt: {item.item_id}",
        "",
        f"- Attempt: `{attempt}`",
        f"- Target: `{target.target.value}`",
        "- Sources: " + ", ".join(f"`{source}`" for source in item.source_files),
        "",
        "## Attributed components",
        "",
    ]
    for index, component in enumerate(prompt.components, start=1):
        lines.extend(
            [
                f"### {index}. {component.category.value}",
                "",
                f"- Role: `{component.message_role.value}`",
                f"- Source: `{component.source_id or '(none)'}`",
                f"- Estimated tokens: `{component.token_count}`",
                "",
            ]
        )
        if component.text:
            fence = "`" * max(
                3, max(map(len, re.findall(r"`+", component.text)), default=0) + 1
            )
            lines.extend([f"{fence}text", component.text, fence, ""])
    lines.extend(["## Provider messages", ""])
    for index, message in enumerate(prompt.render_messages(), start=1):
        lines.extend(
            [
                f"### {index}. {message.role.value}",
                "",
                "```text",
                message.content,
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip()}


def _code_cell(cell: TranslationCell) -> dict[str, Any]:
    language = cell.language or "python"
    return {
        "cell_type": "code",
        "metadata": {
            "language": language,
            "vscode": {"languageId": language},
        },
        "execution_count": None,
        "outputs": [],
        "source": cell.source.strip("\n"),
    }


def _document_cells(
    translation: NotebookTranslation,
    cells: tuple[TranslationCell, ...] | None = None,
) -> list[dict[str, Any]]:
    document = translation.document
    header = [
        f"## {translation.item.item_id}",
        "",
        "- Source file(s): "
        + ", ".join(f"`{source}`" for source in translation.item.source_files),
        f"- Target: `{translation.target.target.value}`",
    ]
    if translation.recovered:
        header.append("- Recovered: `true`")
    result = [_markdown_cell("\n".join(header))]
    if document.analysis.strip():
        result.append(_markdown_cell(f"### Analysis\n\n{document.analysis.strip()}"))
    if document.mapping:
        mapping = ["### Mapping", ""]
        for entry in document.mapping:
            line = f"- **{entry.sas_construct}** → {entry.equivalent}"
            if entry.difference.strip():
                line += f" — {entry.difference.strip()}"
            mapping.append(line)
        result.append(_markdown_cell("\n".join(mapping)))
    for cell in cells if cells is not None else document.cells:
        result.append(
            _code_cell(cell)
            if cell.kind is TranslationCellKind.CODE
            else _markdown_cell(cell.source)
        )
    if document.risks:
        risks = ["### Risks", ""] + [
            f"- **{risk.severity.value}** — {risk.note.strip()}"
            for risk in document.risks
        ]
        result.append(_markdown_cell("\n".join(risks)))
    return result


def _build_notebook(cells: list[dict[str, Any]], target: TargetId) -> dict[str, Any]:
    languages = {
        str(cell.get("metadata", {}).get("language"))
        for cell in cells
        if cell.get("cell_type") == "code"
    }
    mixed = "python" in languages and "sql" in languages
    if mixed:
        for cell in cells:
            if (
                cell.get("cell_type") == "code"
                and cell.get("metadata", {}).get("language") == "sql"
                and not str(cell["source"]).lstrip().startswith("%sql")
            ):
                cell["source"] = f"%sql\n{cell['source']}"
    python_kernel = mixed or target is TargetId.PYSPARK
    language = "python" if python_kernel else "sql"
    stamped = [{**cell, "id": f"cell-{index:04d}"} for index, cell in enumerate(cells)]
    return {
        "cells": stamped,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3" if python_kernel else "Databricks SQL",
                "language": language,
                "name": "python3" if python_kernel else "databricks-sql",
            },
            "language_info": {"name": language},
            "sas_migrate": {"target": target.value},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebooks(
    translations: tuple[NotebookTranslation, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    names: dict[str, str] = {}

    def name_for(source: str) -> str:
        if source == "_cross_file":
            return source
        if source in names:
            return names[source]
        base = _safe(PurePath(source).stem, fallback="translation")
        candidate = base
        suffix = 2
        while candidate in names.values():
            candidate = f"{base}_{suffix}"
            suffix += 1
        names[source] = candidate
        return candidate

    targets: dict[str, TargetId] = {}
    for translation in translations:
        item = translation.item
        if len(item.source_files) == 1:
            name = name_for(item.source_files[0])
            grouped.setdefault(name, []).extend(_document_cells(translation))
            targets[name] = translation.target.target
            continue
        code_cells = translation.document.code_cells
        splittable = bool(code_cells) and all(
            cell.chunk_id is not None and cell.chunk_id in item.chunk_sources
            for cell in code_cells
        )
        if splittable:
            for source in item.source_files:
                routed = tuple(
                    cell
                    for cell in translation.document.cells
                    if cell.kind is TranslationCellKind.MARKDOWN
                    or item.chunk_sources.get(cell.chunk_id or "") == source
                )
                name = name_for(source)
                grouped.setdefault(name, []).extend(
                    _document_cells(translation, routed)
                )
                targets[name] = translation.target.target
            continue
        cross = name_for("_cross_file")
        grouped.setdefault(cross, []).extend(_document_cells(translation))
        targets[cross] = translation.target.target
        for source in item.source_files:
            name = name_for(source)
            grouped.setdefault(name, []).append(
                _markdown_cell(
                    f"## {item.item_id} (cross-file)\n\nRun this step in "
                    "`_cross_file.ipynb` at this point in the sequence."
                )
            )
            targets.setdefault(name, translation.target.target)
    return {
        name: _build_notebook(cells, targets[name]) for name, cells in grouped.items()
    }


class TranslationArtifactService:
    def __init__(self, repository: ArtifactRepository) -> None:
        self._repository = repository

    async def _write(
        self,
        run_id: RunId,
        artifact_id: str,
        media_type: str,
        content: str,
        kind: str,
        **metadata: str,
    ) -> ArtifactLocator:
        write = ArtifactWrite(
            artifact_id=artifact_id,
            media_type=media_type,
            content=content.encode("utf-8"),
            metadata={"kind": kind, **metadata},
        )
        location = await self._repository.write(run_id, write)
        return ArtifactLocator(
            artifact_id=artifact_id,
            location=location,
            kind=kind,
            media_type=media_type,
        )

    async def persist_attempt(
        self,
        run_id: RunId,
        item: TranslationItem,
        target: ResolvedTarget,
        attempt: int,
        prompt: PromptAssembly,
        envelope: ResponseEnvelope | None,
    ) -> tuple[ArtifactLocator, ...]:
        stem = f"{_safe(item.item_id)}-attempt-{attempt}"
        prompt_artifact = await self._write(
            run_id,
            f"prompts/{stem}.md",
            "text/markdown",
            render_effective_prompt(item, target, attempt, prompt),
            "effective_prompt",
            item_id=item.item_id,
            attempt=str(attempt),
        )
        if envelope is None:
            return (prompt_artifact,)
        response_artifact = await self._write(
            run_id,
            f"responses/{stem}.json",
            "application/json",
            json.dumps(
                envelope.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "response_envelope",
            item_id=item.item_id,
            attempt=str(attempt),
            response_mode=envelope.mode.value,
        )
        return prompt_artifact, response_artifact

    async def persist_canonical(
        self,
        run_id: RunId,
        item_id: ItemId,
        document: TranslationDocument,
    ) -> ArtifactLocator:
        return await self._write(
            run_id,
            f"translations/{_safe(item_id)}.md",
            "text/markdown",
            document.to_markdown(),
            "canonical_translation",
            item_id=item_id,
            target=document.target.value,
        )

    async def persist_notebooks(
        self,
        run_id: RunId,
        translations: tuple[NotebookTranslation, ...],
    ) -> tuple[ArtifactLocator, ...]:
        locators: list[ArtifactLocator] = []
        for name, notebook in render_notebooks(translations).items():
            locators.append(
                await self._write(
                    run_id,
                    f"notebooks/{name}.ipynb",
                    "application/x-ipynb+json",
                    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                    "notebook",
                    notebook=name,
                )
            )
        return tuple(locators)


__all__ = [
    "ArtifactLocator",
    "NotebookTranslation",
    "TranslationArtifactService",
    "render_effective_prompt",
    "render_notebooks",
]
