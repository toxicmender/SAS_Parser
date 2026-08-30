"""Provider-neutral LLM invocation port."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from sas_migrate.core.models import ContractModel
from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.tokens import PromptAssembly


class ProviderTokenUsage(ContractModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


class ProviderResponse(ContractModel):
    raw_message: str
    structured_document: TranslationDocument | None = None
    structured_error: str | None = None
    usage: ProviderTokenUsage = Field(default_factory=ProviderTokenUsage)


class LLMPort(Protocol):
    async def invoke(
        self,
        prompt: PromptAssembly,
        target: ResolvedTarget,
        *,
        attempt: int,
    ) -> ProviderResponse:
        """Invoke one target-specific structured request."""

        ...
