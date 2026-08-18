"""Application contracts for ordered corpus translation."""

from __future__ import annotations

from pydantic import Field, model_validator

from sas_migrate.core.ids import ChunkId, ItemId
from sas_migrate.core.models import ContractModel, VersionedContract
from sas_migrate.core.sas import SasBatch, SasBatchResult, SasChunk
from sas_migrate.core.targets import CompatibilityAssessment


class TranslationMember(ContractModel):
    chunk_id: ChunkId
    source_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source: str


class TranslationItem(VersionedContract):
    item_id: ItemId
    members: tuple[TranslationMember, ...] = Field(min_length=1)
    source_files: tuple[str, ...] = Field(min_length=1)
    batch_reason: str = ""
    batch_context: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    compatibility: CompatibilityAssessment = Field(
        default_factory=lambda: CompatibilityAssessment(
            spark_sql_implementable=True
        )
    )

    @model_validator(mode="after")
    def validate_attribution(self) -> TranslationItem:
        chunk_ids = [member.chunk_id for member in self.members]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("translation item chunk ids must be unique")
        member_sources = {member.source_id for member in self.members}
        if not member_sources.issubset(self.source_files):
            raise ValueError("every translation member source must be listed")
        return self

    @property
    def known_chunk_ids(self) -> frozenset[str]:
        return frozenset(member.chunk_id for member in self.members)

    @property
    def chunk_sources(self) -> dict[str, str]:
        return {member.chunk_id: member.source_id for member in self.members}

    @classmethod
    def from_sas(cls, item: SasBatch | SasChunk) -> TranslationItem:
        if isinstance(item, SasBatch):
            chunks = item.chunks
            item_id = item.batch_id
            source_files = tuple(item.source_files) or tuple(
                dict.fromkeys(chunk.source_id or "<inline>" for chunk in chunks)
            )
            reason = item.reason
            context = {
                "input_datasets": tuple(item.input_datasets),
                "output_datasets": tuple(item.output_datasets),
                "required_macros": tuple(item.required_macros),
                "required_librefs": tuple(item.required_librefs),
                "defined_macros": tuple(item.defined_macros),
                "produced_macrovars": tuple(item.produced_macrovars),
                "required_macrovars": tuple(item.required_macrovars),
            }
        else:
            chunks = [item]
            item_id = item.chunk_id
            source_files = (item.source_id or "<inline>",)
            reason = ""
            context = {
                "input_datasets": tuple(item.metadata.input_datasets),
                "output_datasets": tuple(item.metadata.output_datasets),
                "required_macros": tuple(item.metadata.invokes_macros),
                "defined_macros": tuple(item.metadata.defines_macros),
            }
        members = tuple(
            TranslationMember(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id or "<inline>",
                kind=chunk.kind.value,
                source=chunk.text,
            )
            for chunk in chunks
        )
        return cls(
            item_id=item_id,
            members=members,
            source_files=source_files,
            batch_reason=reason,
            batch_context=context,
        )


def translation_items(result: SasBatchResult) -> tuple[TranslationItem, ...]:
    return tuple(TranslationItem.from_sas(item) for item in result.all_ordered_items)


__all__ = ["TranslationItem", "TranslationMember", "translation_items"]
