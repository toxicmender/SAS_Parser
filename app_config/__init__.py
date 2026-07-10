"""Repo-wide runtime configuration loaded from ``config.json``.

Centralises the tunable word/token limits that were previously hard-coded
defaults scattered across ``SasSemanticChunker``, ``InstructionChunker``,
``PromptBuilder``, and ``LLMClientConfig``. Resolution precedence, applied by
each consumer via :func:`get_value`:

    explicit constructor argument  >  config.json value  >  hard default

A JSON ``null`` (or absent key/section/file) means "unset" and falls through
to the default, so a sparse or missing file is always valid.

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
            _cache = json.loads(path.read_text(encoding="utf-8"))
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


def clear_cache() -> None:
    """Forget the cached file so the next access re-searches (for tests)."""
    global _cache
    _cache = None
