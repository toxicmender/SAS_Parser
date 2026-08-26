"""Redaction shared by logs, diagnostics, and adapter error normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import SecretStr

REDACTED = "<redacted>"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9\-._~+/]{8,}=*)"),
    re.compile(
        r"(?i)([\"']?(?:access_token|refresh_token|id_token|client_secret"
        r"|api_key|password|secret_id|role_id|token)[\"']?\s*[:=]\s*[\"']?)"
        r"([^\s\"',&}]{4,})"
    ),
    re.compile(
        r"(?i)([?&](?:access_token|client_secret|api_key|code|password|sig|token)=)"
        r"([^&\s]{4,})"
    ),
)

_SENSITIVE_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|id[_-]?token|token|"
    r"client[_-]?secret|api[_-]?key|password|passphrase|secret[_-]?id|role[_-]?id)"
    r"(?:$|[_-])"
)


def redact_text(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
    return text


def redact_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(item) for item in value]
    return value


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): redact_value(item, key=str(key))
        for key, item in value.items()
    }


__all__ = ["REDACTED", "redact_mapping", "redact_text", "redact_value"]
