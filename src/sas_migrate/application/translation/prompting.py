"""Deterministic, attributed prompts for one translation item."""

from __future__ import annotations

import json

from pydantic import Field, model_validator

from sas_migrate.core.models import VersionedContract
from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget, choose_item_target
from sas_migrate.core.tokens import (
    MessageRole,
    PromptAssembly,
    PromptComponentDraft,
    TokenCategory,
)

from .models import TranslationItem
from .prompt_assembly import PromptAssembler

_CONTEXT_CATEGORIES = frozenset(
    {
        TokenCategory.REFERENCE_GUIDANCE,
        TokenCategory.PROJECT_INSTRUCTIONS,
        TokenCategory.TASK_POLICY,
        TokenCategory.THREAD_NOTES,
        TokenCategory.ROLLING_SUMMARY,
        TokenCategory.SELECTED_HISTORY,
    }
)


class PromptContext(VersionedContract):
    components: tuple[PromptComponentDraft, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_context_categories(self) -> PromptContext:
        invalid = {
            component.category
            for component in self.components
            if component.category not in _CONTEXT_CATEGORIES
        }
        if invalid:
            values = ", ".join(sorted(category.value for category in invalid))
            raise ValueError(f"prompt context contains owned categories: {values}")
        return self


class TranslationPromptBuilder:
    def __init__(self, assembler: PromptAssembler) -> None:
        self._assembler = assembler

    def target_for(
        self,
        item: TranslationItem,
        run_target: ResolvedTarget,
    ) -> ResolvedTarget:
        return choose_item_target(run_target, item.compatibility)

    def build(
        self,
        item: TranslationItem,
        target: ResolvedTarget,
        *,
        context: PromptContext | None = None,
        retry_feedback: tuple[str, ...] = (),
    ) -> PromptAssembly:
        schema = json.dumps(
            TranslationDocument.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        batch = json.dumps(
            {
                "batch_context": item.batch_context,
                "batch_reason": item.batch_reason,
                "item_id": item.item_id,
                "members": [
                    {
                        "chunk_id": member.chunk_id,
                        "kind": member.kind,
                        "source_id": member.source_id,
                    }
                    for member in item.members
                ],
                "source_files": item.source_files,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        drafts: list[PromptComponentDraft] = [
            PromptComponentDraft(
                category=TokenCategory.SYSTEM_STATIC,
                text=(
                    "Translate SAS into an auditable structured document. Preserve "
                    "behavior, ordering, source attribution, and unresolved risks."
                ),
                message_role=MessageRole.SYSTEM,
                source_id="translation-system-v2",
                cacheable=True,
            ),
            PromptComponentDraft(
                category=TokenCategory.STRUCTURED_SCHEMA,
                text=f"Return a document matching this JSON Schema:\n{schema}",
                message_role=MessageRole.SYSTEM,
                source_id="translation-document-schema-v2",
                cacheable=True,
            ),
            PromptComponentDraft(
                category=TokenCategory.TARGET_DIRECTIVE,
                text=(
                    f"Target {target.display_name} ({target.target.value}). "
                    f"Use {target.canonical_language} code and ```{target.fence}``` fences."
                ),
                message_role=MessageRole.SYSTEM,
                source_id=target.target.value,
            ),
        ]
        if context is not None:
            drafts.extend(context.components)
        drafts.append(
            PromptComponentDraft(
                category=TokenCategory.BATCH_CONTEXT,
                text=f"Translation item context:\n{batch}",
                message_role=MessageRole.USER,
                source_id=item.item_id,
            )
        )
        drafts.extend(
            PromptComponentDraft(
                category=TokenCategory.SAS_SOURCE,
                text=(
                    f"SAS source {member.source_id} / {member.chunk_id} "
                    f"({member.kind}):\n```sas\n{member.source}\n```"
                ),
                message_role=MessageRole.USER,
                source_id=member.chunk_id,
            )
            for member in item.members
        )
        if retry_feedback:
            drafts.append(
                PromptComponentDraft(
                    category=TokenCategory.RETRY_FEEDBACK,
                    text="Correct the rejected response:\n- "
                    + "\n- ".join(retry_feedback),
                    message_role=MessageRole.USER,
                    source_id=f"{item.item_id}-retry",
                    ephemeral=True,
                )
            )
        return self._assembler.assemble(drafts)


__all__ = ["PromptContext", "TranslationPromptBuilder"]
