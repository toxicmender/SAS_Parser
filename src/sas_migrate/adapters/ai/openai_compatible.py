"""OpenAI-compatible gateway implementation of the provider-neutral LLM port."""

from __future__ import annotations

from typing import Any

from sas_migrate.application.ports import (
    CredentialValue,
    ProviderResponse,
    ProviderTokenUsage,
)
from sas_migrate.config import GatewaySettings
from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.tokens import PromptAssembly


class GatewayLLMError(RuntimeError):
    """The configured gateway cannot produce a usable provider response."""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                values.append(text)
        return "\n".join(values)
    return ""


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _usage(value: Any) -> ProviderTokenUsage:
    if value is None:
        return ProviderTokenUsage()
    details = getattr(value, "prompt_tokens_details", None)
    return ProviderTokenUsage(
        input_tokens=_integer(getattr(value, "prompt_tokens", None)),
        output_tokens=_integer(getattr(value, "completion_tokens", None)),
        cache_read_tokens=_integer(getattr(details, "cached_tokens", None)),
        cache_write_tokens=_integer(
            getattr(value, "cache_creation_input_tokens", None)
        ),
    )


class OpenAICompatibleLLM:
    """Invoke one model through an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        settings: GatewaySettings,
        credential: CredentialValue,
        model: str,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("gateway model cannot be blank")
        self._settings = settings
        self._credential = credential
        self._model = model.strip()
        self._client = client

    def _build_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - core dependency contract
            raise GatewayLLMError(
                "OpenAI-compatible conversion requires the core 'openai' dependency"
            ) from exc

        secret = self._credential.value.get_secret_value()
        headers = {"api-key": secret}
        if self._settings.gateway_version:
            headers["ai-gateway-version"] = self._settings.gateway_version
        kwargs: dict[str, Any] = {
            "api_key": secret,
            "default_headers": headers,
            "max_retries": self._settings.max_retries,
            "timeout": self._settings.timeout,
        }
        if self._settings.base_url:
            kwargs["base_url"] = self._settings.base_url
        return AsyncOpenAI(**kwargs)

    async def invoke(
        self,
        prompt: PromptAssembly,
        target: ResolvedTarget,
        *,
        attempt: int,
    ) -> ProviderResponse:
        del target, attempt
        client = self._client
        if client is None:
            client = self._build_client()
            self._client = client
        completion = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": message.role.value, "content": message.content}
                for message in prompt.render_messages()
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_document_v2",
                    "strict": True,
                    "schema": TranslationDocument.model_json_schema(),
                },
            },
        )
        choices = getattr(completion, "choices", None)
        if not choices:
            raise GatewayLLMError("gateway response did not contain a completion choice")
        raw = _content_text(getattr(choices[0].message, "content", None))
        if not raw.strip():
            raise GatewayLLMError("gateway response did not contain message content")
        document = None
        structured_error = None
        try:
            document = TranslationDocument.model_validate_json(raw)
        except ValueError as exc:
            structured_error = str(exc)
        return ProviderResponse(
            raw_message=raw,
            structured_document=document,
            structured_error=structured_error,
            usage=_usage(getattr(completion, "usage", None)),
        )


__all__ = ["GatewayLLMError", "OpenAICompatibleLLM"]
