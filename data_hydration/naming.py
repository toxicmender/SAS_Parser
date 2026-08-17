"""Rendering the Unity Catalog name a hydrated table lands under.

The default is the plain three-level ``<catalog_name>.<schema_name>.<table_name>``,
but a deployment that stages loads wants the stage and the run date in the name::

    <catalog_name>.<schema_name>.<table_name>_<stage>_<date>

so the shape is a template, configured as ``data_hydration.table_template``.
Placeholders are written ``<name>``, not ``{name}``: SQL and table names are full
of braces-adjacent punctuation, and the angle form is what the operators writing
these templates already use to describe them.

Three rules, each of which exists because of a specific way this goes wrong:

* **The date is rendered once per run, not once per item.** It arrives here
  already formatted, from :attr:`~data_hydration.models.HydrationPlan.run_date`.
  A run starting at 23:59 must not write half its partitions into yesterday's
  table and half into today's.
* **Validation happens at plan time.** :func:`render` raises on an unknown
  placeholder, a missing value, or a name that is not three-part — so a typo in
  the template fails during ``--dry-run``, before any data has moved, rather
  than after the first partition has landed somewhere unintended.
* **Every part is sanitised.** A stage label like ``pre-prod`` would otherwise
  render a name that cannot be referenced without backticks. Sanitising is
  quieter than failing and matches what the chunker does to every other
  identifier it handles.

Logger name: ``data_hydration.naming``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: The default template: an ordinary three-level managed-table name.
DEFAULT_TEMPLATE = "<catalog_name>.<schema_name>.<table_name>"

#: The date format ``run_date`` is rendered with when nothing else is configured.
DEFAULT_DATE_FORMAT = "%Y%m%d"

#: Every placeholder :func:`render` will substitute. A template naming anything
#: else is a configuration error, not an empty string.
PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "catalog_name",
        "schema_name",
        "table_name",
        "stage",
        "date",
        "libref",
        "source",
        "partition",
    }
)

_PLACEHOLDER_RE = re.compile(r"<([a-z_]+)>")

# Anything outside the Unity Catalog unquoted-identifier charset. Runs collapse
# to a single underscore so `pre-prod` and `pre - prod` do not render differently.
_ILLEGAL_RE = re.compile(r"[^a-z0-9_]+")


class TableNameError(ValueError):
    """The template, or a value filled into it, cannot produce a valid name.

    A ``ValueError`` rather than a ``RuntimeError`` because every case is a bad
    input: an unknown placeholder, a value the template needs and was not given,
    or a result that is not a three-level identifier.
    """


def sanitise_part(value: str) -> str:
    """*value* as one Unity Catalog identifier part.

    Lowercased, non-identifier runs collapsed to ``_``, and leading/trailing
    underscores trimmed — ``Pre-Prod`` becomes ``pre_prod``. Trimming matters
    because a template ending ``_<stage>`` with an empty stage would otherwise
    leave a trailing underscore on every table name.
    """
    return _ILLEGAL_RE.sub("_", value.strip().lower()).strip("_")


def placeholders_in(template: str) -> set[str]:
    """The placeholder names *template* uses."""
    return set(_PLACEHOLDER_RE.findall(template))


def validate_template(template: str) -> None:
    """Check *template* before a run starts.

    Raises
    ------
    TableNameError
        A placeholder this module does not know, or a template that cannot
        produce a three-level name because it has the wrong number of dots.

    The dot count is checked on the template rather than the result so that a
    template like ``<catalog_name>.<table_name>`` is rejected once at startup
    instead of once per table.
    """
    unknown = placeholders_in(template) - PLACEHOLDERS
    if unknown:
        raise TableNameError(
            f"table_template names unknown placeholder(s) "
            f"{', '.join('<' + u + '>' for u in sorted(unknown))}; "
            f"available: {', '.join('<' + p + '>' for p in sorted(PLACEHOLDERS))}"
        )
    # Placeholder values are sanitised to contain no dots, so every dot in the
    # rendered name is one written literally here.
    if template.count(".") != 2:
        raise TableNameError(
            f"table_template must produce a three-level "
            f"catalog.schema.table name, but {template!r} has "
            f"{template.count('.')} dot(s), not 2"
        )


def render(template: str, **values: str | None) -> str:
    """*template* with its placeholders filled and each part sanitised.

    Parameters
    ----------
    template
        A validated template — :func:`validate_template` is called here too, so
        a caller that skipped it still gets the error before a name is built.
    values
        Placeholder values. Only the ones *template* actually uses need to be
        supplied; a placeholder it does use with a ``None`` or empty value is an
        error, because the alternative is silently rendering ``sales__20260815``
        and writing to a table nobody meant.

    Raises
    ------
    TableNameError
        As :func:`validate_template`, plus a missing value or an empty rendered
        part.
    """
    validate_template(template)
    needed = placeholders_in(template)
    filled = {k: v for k, v in values.items() if k in needed}

    missing = sorted(
        name for name in needed if not (filled.get(name) or "").strip()
    )
    if missing:
        raise TableNameError(
            f"table_template uses {', '.join('<' + m + '>' for m in missing)} "
            f"but no value was supplied; set it in config.json "
            f"(data_hydration.*) or drop the placeholder from the template"
        )

    def _sub(match: re.Match[str]) -> str:
        return sanitise_part(filled[match.group(1)] or "")

    rendered = _PLACEHOLDER_RE.sub(_sub, template)

    parts = rendered.split(".")
    if len(parts) != 3 or not all(parts):
        raise TableNameError(
            f"template {template!r} rendered {rendered!r}, which is not a "
            f"three-level catalog.schema.table name"
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"render: {template} -> {rendered}")
    return rendered
