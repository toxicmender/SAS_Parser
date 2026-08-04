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
file is type-checked in one place; :func:`role_value` layers a named role's
sparse overrides (``llm_client.roles.<role>``) on top of it, which is how the
validator and complexity runs get their own timeouts and models.

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
            config = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"load_config: unreadable '{path}': {exc}; skipping")
            continue
        logger.info(f"load_config: using '{path}'")
        _cache = config
        return config
    logger.info("load_config: no config.json found; using hard defaults")
    _cache = {}
    return _cache


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})


def _verify_setting(value: Any) -> bool | str | None:
    """
    A TLS-verification setting in the shape ``requests``, ``hvac`` and ``msal``
    all take: ``True`` (verify against the system CAs), ``False`` (verification
    off — dev only), or a path to a CA bundle. ``None`` when *value* is unset.

    Booleans pass through. A string naming a truth value (``true``/``1``/
    ``yes``/``on``, or their negatives) becomes that boolean, since an
    environment variable can only carry text; any other non-empty string is a
    bundle path. Shared by :mod:`app_config.vault` and :mod:`app_config.azure`,
    which must agree on what ``verify`` means — the reference deployment uses
    one setting for both legs of the credential chain.
    """
    if value is None or isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    low = text.lower()
    if low in _TRUTHY:
        return True
    if low in _FALSEY:
        return False
    return text


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


def _typed(
    value: Any,
    expected: type | tuple[type, ...],
    label: str,
    default: Any,
) -> Any:
    """
    *value* when it matches *expected*, else *default* with a WARNING naming
    *label* (the ``section.key`` path the value came from).

    ``bool`` is rejected where ``int``/``float`` is expected unless ``bool``
    itself is listed (JSON ``true`` is not a number).
    """
    types = expected if isinstance(expected, tuple) else (expected,)
    ok = isinstance(value, types) and not (
        isinstance(value, bool) and bool not in types
    )
    if ok:
        return value
    expected_names = "/".join(t.__name__ for t in types)
    logger.warning(
        f"config.json {label} is {type(value).__name__} ({value!r}), expected "
        f"{expected_names}; ignoring it (default {default!r} applies)"
    )
    return default


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
    return _typed(value, expected, f"{section}.{key}", default)


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
    "model_provider": str,
    "gateway_version": str,
    "provider_client": str,
    "base_url": str,
    "url_headers": dict,
    "timeout": (int, float),
    "cert_file": str,
    "temperature": (int, float),
    "max_retries": int,
    "model_kwargs": dict,
    "max_input_tokens": int,
    "max_output_tokens": int,
    "prompt_caching": bool,
    "requests_per_second": (int, float),
    "max_bucket_size": int,
    "roles": dict,
}

# How the chat model is constructed. "openai_compatible" builds the
# LangChain ChatOpenAI the whole client is written around; "native" wraps a
# raw provider SDK client for a gateway that will not take the LangChain
# payload. See llm_client.client for what the native path gives up.
PROVIDER_CLIENT_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_CLIENT_NATIVE = "native"
PROVIDER_CLIENTS: tuple[str, ...] = (
    PROVIDER_CLIENT_OPENAI_COMPATIBLE,
    PROVIDER_CLIENT_NATIVE,
)

# llm_client keys that describe the *section* rather than one model, so a
# per-role overlay may not carry them.
_ROLE_EXCLUDED_KEYS = frozenset({"roles"})


def _validate_llm_value(key: str, value: Any, default: Any, label: str) -> Any:
    """
    The value-level (not type-level) rules for an ``llm_client`` entry:
    ``model`` must be accessible, ``provider_client`` must name a known
    strategy, and ``url_headers`` must map to strings. A violation is
    ignored with a WARNING and *default* applies, the same degrade-don't-crash
    rule as a wrong-typed entry. *label* names the config path in the message.
    """
    if (
        key == "model"
        and isinstance(value, str)
        and value != default
        and not is_accessible_model(value)
    ):
        logger.warning(
            f"llm_client_value: config.json {label} {value!r} is not an "
            f"accessible model (accessible: {', '.join(ACCESSIBLE_MODELS)}); "
            f"ignoring it (default {default!r} applies)"
        )
        return default
    if (
        key == "provider_client"
        and isinstance(value, str)
        and value != default
        and value not in PROVIDER_CLIENTS
    ):
        logger.warning(
            f"llm_client_value: config.json {label} {value!r} is not one of "
            f"{'/'.join(PROVIDER_CLIENTS)}; ignoring it "
            f"(default {default!r} applies)"
        )
        return default
    if (
        key == "url_headers"
        and isinstance(value, dict)
        and not all(isinstance(v, str) for v in value.values())
    ):
        logger.warning(
            f"llm_client_value: config.json {label} must map header names to "
            f"string values; ignoring it (default {default!r} applies)"
        )
        return default
    return value


def llm_client_value(key: str, default: Any = None) -> Any:
    """
    Type-checked value from the ``llm_client`` section of config.json.

    *key* must appear in the section's schema (:data:`_LLM_CLIENT_TYPES`);
    an unknown key raises ``KeyError`` — that is a programming error, not a
    config error. Wrong-typed file values are ignored with a WARNING and
    *default* applies. ``url_headers`` must additionally map to string
    values (JSON object keys are always strings), ``model`` must name one of
    :data:`ACCESSIBLE_MODELS`, and ``provider_client`` one of
    :data:`PROVIDER_CLIENTS`, or the entry is likewise ignored.
    """
    expected = _LLM_CLIENT_TYPES[key]
    value = get_typed_value("llm_client", key, expected, default)
    return _validate_llm_value(key, value, default, f"llm_client.{key}")


def role_value(role: str | None, key: str, default: Any = None) -> Any:
    """
    An ``llm_client`` value with a per-role overlay applied:

        ``llm_client.roles.<role>.<key>``  >  ``llm_client.<key>``  >  *default*

    The reference deployment configures one gateway per *role* — its
    ``ai_gateway_details`` (timeout 6000) and ``ai_validator_config``
    (timeout 12000) are siblings differing in a couple of keys — so a role is
    a **sparse overlay** on the base section, not a section of its own: a key
    the role does not mention resolves exactly as it would without one.

    *role* may be ``None`` (no overlay), and an unknown role name, a
    wrong-typed ``roles`` entry, or a wrong-typed/invalid overlay value all
    fall through to the base section with a WARNING where one is warranted.
    ``roles`` itself cannot be overlaid.

    Parameters
    ----------
    role : str | None
        Role name, e.g. ``"validator"`` or ``"complexity"``.
    key : str
        An ``llm_client`` schema key (:data:`_LLM_CLIENT_TYPES`).
    default : Any
        Applied when neither the overlay nor the base section has a usable
        value.
    """
    base = llm_client_value(key, default)
    if not role or key in _ROLE_EXCLUDED_KEYS:
        return base
    roles = llm_client_value("roles")
    if not isinstance(roles, dict):
        return base
    overlay = roles.get(role)
    if overlay is None:
        return base
    if not isinstance(overlay, dict):
        logger.warning(
            f"role_value: config.json llm_client.roles.{role} is "
            f"{type(overlay).__name__}, expected object; ignoring it (the "
            f"llm_client section applies)"
        )
        return base
    value = overlay.get(key, _MISSING)
    if value is _MISSING or value is None:
        return base
    label = f"llm_client.roles.{role}.{key}"
    value = _typed(value, _LLM_CLIENT_TYPES[key], label, base)
    return _validate_llm_value(key, value, base, label)


def clear_cache() -> None:
    """Forget the cached file so the next access re-searches (for tests)."""
    global _cache
    _cache = None
