"""Versioned structured and raw-fallback response envelopes."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..ids import ChunkId
from ..models import ContractModel, VersionedContract
from ..targets.models import ResolvedTarget, TargetId
from ..targets.validation import ResponseValidationResult


class TranslationCellKind(StrEnum):
    CODE = "code"
    MARKDOWN = "markdown"


class RiskSeverity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class ResponseMode(StrEnum):
    STRUCTURED = "structured"
    RAW_FALLBACK = "raw_fallback"


class MappingEntry(ContractModel):
    sas_construct: str
    equivalent: str
    difference: str = ""


class TranslationCell(ContractModel):
    kind: TranslationCellKind
    source: str
    language: Literal["python", "sql"] | None = None
    chunk_id: ChunkId | None = None

    @model_validator(mode="after")
    def validate_language(self) -> TranslationCell:
        if self.kind is TranslationCellKind.MARKDOWN and self.language is not None:
            raise ValueError("markdown cells cannot declare a code language")
        return self


class RiskNote(ContractModel):
    severity: RiskSeverity
    note: str


class TranslationDocument(VersionedContract):
    """One normalized translation, independent of provider response mode."""

    target: TargetId
    analysis: str
    mapping: tuple[MappingEntry, ...] = Field(default_factory=tuple)
    cells: tuple[TranslationCell, ...] = Field(default_factory=tuple)
    risks: tuple[RiskNote, ...] = Field(default_factory=tuple)

    @property
    def code_cells(self) -> tuple[TranslationCell, ...]:
        return tuple(cell for cell in self.cells if cell.kind is TranslationCellKind.CODE)

    def to_markdown(self, *, default_fence: str | None = None) -> str:
        """Render the one canonical four-section Markdown representation."""

        fence = default_fence or ("python" if self.target is TargetId.PYSPARK else "sql")
        lines = ["## Analysis", "", self.analysis.strip(), "", "## Mapping", ""]
        if self.mapping:
            for entry in self.mapping:
                mapping = f"- **{entry.sas_construct}** → {entry.equivalent}"
                if entry.difference.strip():
                    mapping += f" — {entry.difference.strip()}"
                lines.append(mapping)
        else:
            lines.append("_No construct mapping reported._")

        lines.extend(["", "## Translation", ""])
        if self.cells:
            for cell in self.cells:
                if cell.kind is TranslationCellKind.CODE:
                    lines.extend(
                        [f"```{cell.language or fence}", cell.source.strip("\n"), "```", ""]
                    )
                else:
                    lines.extend([cell.source.strip(), ""])
        else:
            lines.extend(["_No translation produced._", ""])

        lines.extend(["## Risks", ""])
        if self.risks:
            lines.extend(f"- **{risk.severity}** — {risk.note.strip()}" for risk in self.risks)
        else:
            lines.append("_No risks flagged._")
        return "\n".join(lines).rstrip() + "\n"


class ResponseEnvelope(VersionedContract):
    """Auditable provider result before publication acceptance."""

    mode: ResponseMode
    raw_message: str
    document: TranslationDocument | None = None
    structured_error: str | None = None
    resolved_target: ResolvedTarget
    validation: ResponseValidationResult

    @model_validator(mode="after")
    def validate_consistency(self) -> ResponseEnvelope:
        if self.mode is ResponseMode.STRUCTURED and self.structured_error is not None:
            raise ValueError("structured responses cannot carry a structured parsing error")
        if self.mode is ResponseMode.RAW_FALLBACK and not self.structured_error:
            raise ValueError("raw fallback must retain the structured parsing error")
        if self.document is not None and self.validation.reported_target != self.document.target:
            raise ValueError("validation reported_target must match the normalized document")
        if self.validation.resolved_target != self.resolved_target.target:
            raise ValueError("validation must use the envelope's resolved target")
        if self.validation.valid and self.document is None:
            raise ValueError("an accepted response requires a normalized document")
        return self
