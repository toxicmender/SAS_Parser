"""Pure Unity Catalog target-name rendering for hydration plans."""

from __future__ import annotations

import re

DEFAULT_TEMPLATE = "<catalog_name>.<schema_name>.<table_name>"
PLACEHOLDERS = frozenset(
    {"catalog_name", "schema_name", "table_name", "stage", "date", "libref", "source", "partition"}
)
_PLACEHOLDER_RE = re.compile(r"<([a-z_]+)>")
_ILLEGAL_RE = re.compile(r"[^a-z0-9_]+")


class TableNameError(ValueError):
    pass


def sanitise_part(value: str) -> str:
    return _ILLEGAL_RE.sub("_", value.strip().lower()).strip("_")


def placeholders_in(template: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(template))


def validate_template(template: str) -> None:
    unknown = placeholders_in(template) - PLACEHOLDERS
    if unknown:
        names = ", ".join(f"<{name}>" for name in sorted(unknown))
        raise TableNameError(f"table_template names unknown placeholder(s) {names}")
    if template.count(".") != 2:
        raise TableNameError("table_template must produce a three-level catalog.schema.table name")


def render(template: str, **values: str | None) -> str:
    validate_template(template)
    needed = placeholders_in(template)
    missing = sorted(name for name in needed if not (values.get(name) or "").strip())
    if missing:
        names = ", ".join(f"<{name}>" for name in missing)
        raise TableNameError(f"table_template uses {names} but no value was supplied")

    rendered = _PLACEHOLDER_RE.sub(
        lambda match: sanitise_part(values[match.group(1)] or ""),
        template,
    )
    if len(rendered.split(".")) != 3 or not all(rendered.split(".")):
        raise TableNameError(f"template rendered {rendered!r}, which is not a three-level name")
    return rendered


__all__ = ["DEFAULT_TEMPLATE", "TableNameError", "render", "sanitise_part", "validate_template"]
