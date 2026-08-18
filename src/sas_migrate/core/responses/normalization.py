"""Normalize raw provider Markdown into the v2 translation document."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from ..models import ContractModel
from ..targets.models import ResolvedTarget
from ..targets.registry import PYSPARK, SPARK_SQL
from ..targets.validation import TargetIssueCode, TargetValidationIssue
from .models import (
    MappingEntry,
    RiskNote,
    RiskSeverity,
    TranslationCell,
    TranslationCellKind,
    TranslationDocument,
)

_SECTION_RE = re.compile(
    r"^##[ \t]+(?P<name>Analysis|Mapping|Translation|Risks)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE_RE = re.compile(
    r"^\x60{3}[ \t]*(?P<info>[^\n\x60]*)\r?\n"
    r"(?P<source>.*?)^\x60{3}[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_MAPPING_RE = re.compile(
    r"^-\s+\*\*(?P<sas>.+?)\*\*\s*(?:→|->|:)\s*(?P<rest>.+?)\s*$"
)
_RISK_RE = re.compile(
    r"^-\s*(?:⚠️\s*)?\*\*(?P<severity>P[0-2])\*\*\s*(?:—|-|:)\s*"
    r"(?P<note>.+?)\s*$",
    re.IGNORECASE,
)
_CHUNK_HEADING_RE = re.compile(
    r"(?:^|\n)###[ \t]+(?:Chunk[ \t]+)?\x60?"
    r"(?P<chunk>[^\x60\n]+?)\x60?[ \t]*$",
    re.IGNORECASE,
)
_PLACEHOLDERS = frozenset(
    {
        "_no translation produced._",
        "_no construct mapping reported._",
        "_no risks flagged._",
    }
)


class RawNormalizationResult(ContractModel):
    """Normalized raw response plus issues only the raw parser can observe."""

    document: TranslationDocument
    issues: tuple[TargetValidationIssue, ...] = Field(default_factory=tuple)


def _sections(raw_message: str) -> dict[str, str]:
    matches = list(_SECTION_RE.finditer(raw_message))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_message)
        name = match.group("name").casefold()
        content = raw_message[match.end() : end].strip()
        if name in sections and content:
            sections[name] = f"{sections[name]}\n\n{content}".strip()
        else:
            sections[name] = content
    return sections


def _mapping_entries(text: str) -> tuple[MappingEntry, ...]:
    entries: list[MappingEntry] = []
    for line in text.splitlines():
        match = _MAPPING_RE.match(line.strip())
        if match is None:
            continue
        equivalent, separator, difference = match.group("rest").partition(" — ")
        entries.append(
            MappingEntry(
                sas_construct=match.group("sas").strip(),
                equivalent=equivalent.strip(),
                difference=difference.strip() if separator else "",
            )
        )
    return tuple(entries)


def _risk_entries(text: str) -> tuple[RiskNote, ...]:
    risks: list[RiskNote] = []
    for line in text.splitlines():
        match = _RISK_RE.match(line.strip())
        if match is None:
            continue
        risks.append(
            RiskNote(
                severity=RiskSeverity(match.group("severity").upper()),
                note=match.group("note").strip(),
            )
        )
    return tuple(risks)


def _normalize_fence(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.casefold())


def _fence_language(info: str) -> Literal["python", "sql"] | None:
    if not info:
        return None
    normalized = _normalize_fence(info.split(maxsplit=1)[0])
    for definition in (PYSPARK, SPARK_SQL):
        names = {
            definition.target.value,
            definition.display_name,
            definition.canonical_language,
            definition.fence,
            *definition.aliases,
        }
        if normalized in {_normalize_fence(name) for name in names}:
            return "python" if definition is PYSPARK else "sql"
    return None


def _append_markdown(cells: list[TranslationCell], value: str) -> str | None:
    text = value.strip()
    if not text or text.casefold() in _PLACEHOLDERS:
        return None
    chunk_match = _CHUNK_HEADING_RE.search(text)
    chunk_id = None
    if chunk_match is not None:
        chunk_id = chunk_match.group("chunk").strip()
        text = text[: chunk_match.start()].strip()
    if text:
        cells.append(
            TranslationCell(
                kind=TranslationCellKind.MARKDOWN,
                source=text,
            )
        )
    return chunk_id


def _translation_cells(
    text: str,
) -> tuple[tuple[TranslationCell, ...], tuple[TargetValidationIssue, ...]]:
    cells: list[TranslationCell] = []
    issues: list[TargetValidationIssue] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        chunk_id = _append_markdown(cells, text[cursor : match.start()])
        info = match.group("info").strip()
        language = _fence_language(info)
        cell_index = len(cells)
        if info and language is None:
            issues.append(
                TargetValidationIssue(
                    code=TargetIssueCode.FOREIGN_LANGUAGE,
                    message=f"raw response declared unsupported fence {info!r}",
                    cell_index=cell_index,
                    chunk_id=chunk_id,
                )
            )
        cells.append(
            TranslationCell(
                kind=TranslationCellKind.CODE,
                source=match.group("source").strip("\r\n"),
                language=language,
                chunk_id=chunk_id,
            )
        )
        cursor = match.end()
    _append_markdown(cells, text[cursor:])
    return tuple(cells), tuple(issues)


def normalize_raw_response(
    raw_message: str,
    target: ResolvedTarget,
) -> RawNormalizationResult:
    """Parse raw four-section Markdown using the already resolved target."""

    sections = _sections(raw_message)
    translation = sections.get("translation", "")
    cells, issues = _translation_cells(translation)
    analysis = sections.get("analysis", "")
    if not sections:
        analysis = raw_message.strip()
    return RawNormalizationResult(
        document=TranslationDocument(
            target=target.target,
            analysis=analysis,
            mapping=_mapping_entries(sections.get("mapping", "")),
            cells=cells,
            risks=_risk_entries(sections.get("risks", "")),
        ),
        issues=issues,
    )


__all__ = ["RawNormalizationResult", "normalize_raw_response"]
