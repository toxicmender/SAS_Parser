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
import re
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
        _reject_removed_keys(config, path)
        _cache = config
        return config
    logger.info("load_config: no config.json found; using hard defaults")
    _cache = {}
    return _cache


# Keys that were removed, and what replaced them. A removed key is worse than
# a wrong one: `get_value` would fall through to the default and the run would
# proceed quietly on a setting the operator believes they changed. These moved
# from words to tokens, so silently ignoring the old value would also mean
# running with a budget ~40% off what the file says.
_REMOVED_KEYS: dict[tuple[str, str], str] = {
    ("prompt_builder", "max_instruction_words"): (
        "prompt_builder.max_instruction_tokens (tokens, not words)"
    ),
    ("user_instructions", "max_words"): (
        "user_instructions.max_tokens (tokens, not words)"
    ),
    ("instruction_chunker", "min_words"): "instruction_chunker.min_tokens",
    ("instruction_chunker", "max_words"): "instruction_chunker.max_tokens",
    ("instruction_chunker", "overlap_words"): "instruction_chunker.overlap_tokens",
}


def _reject_removed_keys(config: dict[str, Any], path: Path) -> None:
    """Raise when *config* still sets a key this version no longer reads.

    Deliberately not a fallback: the value is not converted or honoured. The
    point is that a stale config fails visibly at load rather than running with
    a budget nobody chose.
    """
    stale = [
        f"{section}.{key} -> {replacement}"
        for (section, key), replacement in _REMOVED_KEYS.items()
        if isinstance(config.get(section), dict) and key in config[section]
    ]
    if stale:
        raise ValueError(
            f"'{path}' sets configuration keys that were removed: "
            + "; ".join(stale)
            + ". These budgets are counted in tokens now, not words, so the "
            "old values are not carried over — set the new keys explicitly."
        )


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


def resolve_path(value: str) -> Path:
    """A path *from* config, anchored so it survives the working directory.

    ``config.json`` is found relative to the repo root even when the process
    runs elsewhere (see :func:`_candidate_paths`), so a relative path *inside*
    it that only resolved against ``cwd`` would silently miss for any caller
    not started from the root — a Docker entrypoint, a scheduled job, an
    editor's test runner. Absolute paths and paths that exist relative to
    ``cwd`` are honoured as given; anything else falls back to the repo root,
    which is what a value like ``prompt_builder/instructions`` means.
    """
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parents[1] / path


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

# Delta memory tables are deliberately required to be fully qualified. Unlike
# SQL values, table identifiers cannot be passed through parameter markers;
# keeping one Catalog.Schema.Table spelling in config makes it unambiguous
# which Unity Catalog object production memory will mutate.
_MEMORY_TABLE_TYPES: dict[str, type] = {
    "delta_table": str,
    "cdf_audit_table": str,
}
_FULL_TABLE_NAME = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z"
)


def memory_table_value(key: str, default: str | None = None) -> str | None:
    """Validated ``memory.<key>`` table name from ``config.json``.

    Both the primary KV table and its CDF audit table must use a normal,
    three-part Unity Catalog identifier (``Catalog.Schema.Table``). ``null``
    retains the supplied default, so a shipped config can document both
    settings without enabling Delta for local runs.
    """
    expected = _MEMORY_TABLE_TYPES.get(key)
    if expected is None:
        raise KeyError(f"unknown memory table config key {key!r}")
    value = get_typed_value("memory", key, expected, default)
    if value is not None and not _FULL_TABLE_NAME.fullmatch(value):
        logger.warning(
            "memory_table_value: config.json memory.%s must be a fully "
            "qualified Catalog.Schema.Table; ignoring %r",
            key,
            value,
        )
        return default
    return value

# How the chat model is constructed. "openai_compatible" builds the
# LangChain ChatOpenAI the whole client is written around; "native" wraps a
# raw provider SDK client for a gateway that will not take the LangChain
# payload. "auto" (the default) picks between them from the model's provider,
# which is what the reference deployment does against this same gateway.
# See llm_client.client for what the native path gives up.
PROVIDER_CLIENT_AUTO = "auto"
PROVIDER_CLIENT_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_CLIENT_NATIVE = "native"
PROVIDER_CLIENTS: tuple[str, ...] = (
    PROVIDER_CLIENT_AUTO,
    PROVIDER_CLIENT_OPENAI_COMPATIBLE,
    PROVIDER_CLIENT_NATIVE,
)

# Which client each model provider is reached through under "auto". The
# reference's llm_client.ai_sas_converter does exactly this:
#
#     match model_provider.lower():
#         case "openai" | "google": init_llm_model(...)    # ChatOpenAI
#         case "anthropic":         init_claude_model(...) # raw openai SDK
#         case _: raise RuntimeError("UNKNOWN model provider.")
#
# A provider absent from this table is an error rather than a guess: silently
# choosing a client for an unknown provider is how a run fails deep in the
# gateway instead of at construction.
PROVIDER_CLIENT_BY_PROVIDER: dict[str, str] = {
    "anthropic": PROVIDER_CLIENT_NATIVE,
    "openai": PROVIDER_CLIENT_OPENAI_COMPATIBLE,
    "google": PROVIDER_CLIENT_OPENAI_COMPATIBLE,
}

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


# The one spelling of a timestamp that goes in a path. Basic ISO 8601 with the
# separators stripped, because ':' is not a filename character on Windows and
# is percent-encoded in a SharePoint URL. Exposed as well as applied, since
# validation.pdf formats an existing datetime with it rather than taking now().
UTC_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_stamp() -> str:
    """
    A path-safe UTC timestamp, ``20260806T120000Z``.

    Names a run's output folder in the document library and in the local
    report tree. It lives here — beside the config every one of those callers
    already reads — because the two CLIs must agree on it: a run folder whose
    name is formatted differently by the tool that wrote it and the tool that
    lists it is a folder nobody finds.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime(UTC_STAMP_FORMAT)


def clear_cache() -> None:
    """Forget the cached file so the next access re-searches (for tests)."""
    global _cache
    _cache = None
