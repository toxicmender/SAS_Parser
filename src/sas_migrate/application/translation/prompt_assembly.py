"""Build provider messages without flattening away token attribution."""

from __future__ import annotations

from collections.abc import Iterable

from sas_migrate.core.tokens import (
    MessageRole,
    PromptAssembly,
    PromptComponent,
    PromptComponentDraft,
    TokenCategory,
    TokenCounter,
)


class PromptAssembler:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def assemble(self, drafts: Iterable[PromptComponentDraft]) -> PromptAssembly:
        components = tuple(
            PromptComponent(
                category=draft.category,
                text=draft.text,
                message_role=draft.message_role,
                token_count=self._counter.count_text(draft.text),
                source_id=draft.source_id,
                cacheable=draft.cacheable,
                ephemeral=draft.ephemeral,
            )
            for draft in drafts
        )
        provisional = PromptAssembly(
            components=components,
            estimator=self._counter.estimator,
            encoding=self._counter.encoding,
            approximate=self._counter.approximate,
        )
        messages = provisional.render_messages()
        rendered_tokens = sum(
            self._counter.count_text(message.content) for message in messages
        )
        component_tokens = sum(component.token_count for component in components)
        composition_tokens = max(0, rendered_tokens - component_tokens)
        framing = PromptComponent(
            category=TokenCategory.CHAT_FRAMING,
            text="",
            message_role=MessageRole.SYSTEM,
            token_count=self._counter.framing_tokens(len(messages))
            + composition_tokens,
            source_id="chat_framing",
        )
        return PromptAssembly(
            components=(*components, framing),
            estimator=self._counter.estimator,
            encoding=self._counter.encoding,
            approximate=self._counter.approximate,
        )


__all__ = ["PromptAssembler"]
