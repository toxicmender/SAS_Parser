"""Repo-wide runtime configuration loaded from ``config.json``.

Centralises the tunable word/token limits that were previously hard-coded
defaults scattered across ``SasSemanticChunker``, ``InstructionChunker``,
``PromptBuilder``, and ``LLMClientConfig``. Resolution precedence, applied by
each consumer via :func:`get_value`:

    explicit constructor argument  >  config.json value  >  hard default

A JSON ``null`` (or absent key/section/file) means "unset" and falls through
to the default, so a sparse or missing file is always valid.

Two access levels: :func:`get_value` returns raw JSON values, while
:func:`get_typed_value` also checks the JSON type and degrades a wrong-typed
entry to the default with a WARNING. The ``llm_client`` section is parsed
through a schema (:func:`llm_client_value`) so every LLM knob read from the
file is type-checked in one place.

The file is searched in order: the ``SAS_PARSER_CONFIG`` environment variable
(explicit path), ``config.json`` in the current working directory, then
``config.json`` at the repo root (next to this package — present in a source
checkout, absent in an installed wheel). The first readable hit wins and is
cached for the process; call :func:`clear_cache` after changing the
environment (tests do).

This package imports nothing from ``chunker``, ``memory``, ``llm_client``, or
``prompt_builder`` — it is a leaf, like ``chunker.keywords``, so any package
may depend on it without violating the downward-import rule.

Logger name: ``app_config``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "SAS_PARSER_CONFIG"
_FILENAME = "config.json"

_MISSING = object()
_cache: dict[str, Any] | None = None


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get(ENV_VAR)
    if env:
        paths.append(Path(env))
    paths.append(Path.cwd() / _FILENAME)
    paths.append(Path(__file__).resolve().parents[1] / _FILENAME)
    return paths


def load_config() -> dict[str, Any]:
    """The parsed config mapping ({} when no file is found), process-cached."""
    global _cache
    if _cache is not None:
        return _cache
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            # utf-8-sig also accepts BOM-less files; Windows editors and
            # PowerShell 5.1 commonly prepend a BOM, which plain utf-8 rejects.
            _cache = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"load_config: unreadable '{path}': {exc}; skipping")
            continue
        logger.info(f"load_config: using '{path}'")
        return _cache
    logger.info("load_config: no config.json found; using hard defaults")
    _cache = {}
    return _cache


def get_value(section: str, key: str, default: Any = None) -> Any:
    """
    ``config[section][key]``, or *default* when the file, section, or key is
    absent — or when the value is JSON ``null`` (null means "unset", so a
    template config listing every key changes nothing until edited).
    """
    value = load_config().get(section, {}).get(key, _MISSING)
    if value is _MISSING or value is None:
        return default
    return value


def resolve(explicit: Any, section: str, key: str, default: Any) -> Any:
    """Apply the precedence rule: *explicit* (if not None) > config > *default*."""
    if explicit is not None:
        return explicit
    return get_value(section, key, default)


def get_typed_value(
    section: str,
    key: str,
    expected: type | tuple[type, ...],
    default: Any = None,
) -> Any:
    """
    :func:`get_value` with a JSON-type check: a present value of the wrong
    type is ignored with a WARNING and *default* applies, so one bad entry
    in config.json degrades that key instead of crashing the consumer.

    ``bool`` is rejected where ``int``/``float`` is expected unless ``bool``
    itself is listed (JSON ``true`` is not a number).
    """
    value = get_value(section, key, _MISSING)
    if value is _MISSING:
        return default
    types = expected if isinstance(expected, tuple) else (expected,)
    ok = isinstance(value, types) and not (
        isinstance(value, bool) and bool not in types
    )
    if not ok:
        expected_names = "/".join(t.__name__ for t in types)
        logger.warning(
            f"get_typed_value: config.json {section}.{key} is "
            f"{type(value).__name__} ({value!r}), expected {expected_names}; "
            f"ignoring it (default {default!r} applies)"
        )
        return default
    return value


# Chat-model identifiers this deployment can actually reach. An
# llm_client.model config value that names anything else is ignored with a
# WARNING (the default applies), the same degrade-don't-crash rule as a
# wrong-typed entry. Provider-prefixed ("anthropic:claude-opus-4-6") and
# date-suffixed ("claude-sonnet-4-5-20250929") forms of an accessible model
# are accepted.
ACCESSIBLE_MODELS: tuple[str, ...] = (
    "claude-sonnet-4-5",  # Anthropic Claude Sonnet 4.5
    "claude-opus-4-6",    # Anthropic Claude Opus 4.6
    "gpt-5.4",            # OpenAI GPT-5.4
    "gemini-3.1-pro",     # Google Gemini 3.1 Pro
)


def is_accessible_model(model: str) -> bool:
    """
    True when *model* names one of :data:`ACCESSIBLE_MODELS`, tolerating a
    LangChain provider prefix ("anthropic:...") and a dated snapshot suffix
    ("-20250929").
    """
    bare = model.split(":", 1)[-1]
    return any(
        bare == known or bare.startswith(f"{known}-")
        for known in ACCESSIBLE_MODELS
    )


# JSON types accepted per llm_client key. The section's parse rules live
# here beside the loader — one schema — instead of scattered through the
# LLMClientConfig default factories. api_key is deliberately absent:
# secrets are not read from config.json.
_LLM_CLIENT_TYPES: dict[str, type | tuple[type, ...]] = {
    "model": str,
    "base_url": str,
    "url_headers": dict,
    "timeout": (int, float),
    "temperature": (int, float),
    "max_retries": int,
    "model_kwargs": dict,
    "max_input_tokens": int,
    "max_output_tokens": int,
}


def llm_client_value(key: str, default: Any = None) -> Any:
    """
    Type-checked value from the ``llm_client`` section of config.json.

    *key* must appear in the section's schema (:data:`_LLM_CLIENT_TYPES`);
    an unknown key raises ``KeyError`` — that is a programming error, not a
    config error. Wrong-typed file values are ignored with a WARNING and
    *default* applies. ``url_headers`` must additionally map to string
    values (JSON object keys are always strings) or the whole mapping is
    ignored, and ``model`` must name one of :data:`ACCESSIBLE_MODELS` or the
    entry is likewise ignored.
    """
    expected = _LLM_CLIENT_TYPES[key]
    value = get_typed_value("llm_client", key, expected, default)
    if (
        key == "model"
        and isinstance(value, str)
        and value != default
        and not is_accessible_model(value)
    ):
        logger.warning(
            f"llm_client_value: config.json llm_client.model {value!r} is not "
            f"an accessible model (accessible: {', '.join(ACCESSIBLE_MODELS)}); "
            f"ignoring it (default {default!r} applies)"
        )
        return default
    if (
        key == "url_headers"
        and isinstance(value, dict)
        and not all(isinstance(v, str) for v in value.values())
    ):
        logger.warning(
            f"llm_client_value: config.json llm_client.url_headers must map "
            f"header names to string values; ignoring it "
            f"(default {default!r} applies)"
        )
        return default
    return value


def clear_cache() -> None:
    """Forget the cached file so the next access re-searches (for tests)."""
    global _cache
    _cache = None
