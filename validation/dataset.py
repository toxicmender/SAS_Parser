"""Loading validation cases from JSON files. See validation/README.md.

A case file is either one JSON object or a list of them. Each object maps
onto :class:`~validation.models.ValidationCase`, with one extension: instead
of an inline ``sas_source`` string, a case may give ``sas_path`` — a path to
a ``.sas`` file resolved relative to the JSON file — so real programs don't
have to be JSON-escaped.

Logger name: ``validation.dataset``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import ValidationCase

logger = logging.getLogger(__name__)


def _case_from_mapping(raw: dict[str, Any], json_path: Path) -> ValidationCase:
    data = dict(raw)
    sas_path = data.pop("sas_path", None)
    if sas_path is not None:
        if "sas_source" in data:
            raise ValueError(
                f"{json_path}: case '{data.get('case_id', '?')}' has both "
                f"'sas_source' and 'sas_path'; use exactly one"
            )
        resolved = (json_path.parent / sas_path).resolve()
        logger.debug(f"_case_from_mapping: reading SAS source from '{resolved}'")
        data["sas_source"] = resolved.read_text(encoding="utf-8")
    return ValidationCase(**data)


def load_cases(path: str | Path) -> list[ValidationCase]:
    """
    Load every ``*.json`` case file under *path* (a directory, searched
    non-recursively and sorted for deterministic order — case order is
    report order), or a single case file when *path* is a file.
    """
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no case files (*.json) found under '{root}'")

    cases: list[ValidationCase] = []
    for file in files:
        raw = json.loads(file.read_text(encoding="utf-8"))
        mappings = raw if isinstance(raw, list) else [raw]
        cases.extend(_case_from_mapping(m, file) for m in mappings)
        logger.debug(f"load_cases: '{file.name}' -> {len(mappings)} case(s)")

    ids = [c.case_id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate case_id(s): {', '.join(duplicates)}")
    logger.info(f"load_cases: {len(cases)} case(s) from '{root}'")
    return cases
