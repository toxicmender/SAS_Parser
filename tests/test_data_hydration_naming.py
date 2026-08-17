"""
Tests for data_hydration/naming.py — the target-table name template.

This is a small module with an outsized failure mode: it decides where data
lands. The behaviours pinned here are the ones whose absence would be silent —
a template that renders a two-level name, an empty stage that yields
``sales__20260815``, a stage label carrying a hyphen that produces a name
nothing can reference without backticks. Each of those would write somewhere
plausible-looking and wrong.

The other half is *when* it fails. Validation happens at plan time, so a broken
template stops a ``--dry-run`` rather than surfacing after the first partition
has already been written.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from data_hydration.naming import (
    DEFAULT_TEMPLATE,
    PLACEHOLDERS,
    TableNameError,
    placeholders_in,
    render,
    sanitise_part,
    validate_template,
)

REQUESTED = "<catalog_name>.<schema_name>.<table_name>_<stage>_<date>"


class TestRendering:
    def test_the_default_template_is_a_plain_three_level_name(self):
        assert (
            render(
                DEFAULT_TEMPLATE,
                catalog_name="main",
                schema_name="edwprod",
                table_name="accounts",
            )
            == "main.edwprod.accounts"
        )

    def test_the_stage_and_date_form(self):
        assert (
            render(
                REQUESTED,
                catalog_name="main",
                schema_name="edwprod",
                table_name="accounts",
                stage="bronze",
                date="20260815",
            )
            == "main.edwprod.accounts_bronze_20260815"
        )

    def test_values_the_template_does_not_use_are_ignored(self):
        # A caller passes every placeholder it can fill; only the ones the
        # template names have to be there.
        assert (
            render(
                DEFAULT_TEMPLATE,
                catalog_name="main",
                schema_name="s",
                table_name="t",
                stage=None,
                date="20260815",
                libref="x",
            )
            == "main.s.t"
        )

    def test_every_documented_placeholder_renders(self):
        # PLACEHOLDERS is the contract config.json's comment advertises; a name
        # in it that render rejects would be documentation pointing at nothing.
        template = "<catalog_name>.<schema_name>." + "_".join(
            f"<{p}>" for p in sorted(PLACEHOLDERS - {"catalog_name", "schema_name"})
        )
        rendered = render(template, **{p: p for p in PLACEHOLDERS})
        assert rendered.count(".") == 2


class TestSanitising:
    def test_a_hyphenated_stage_becomes_underscored(self):
        assert (
            render(
                REQUESTED,
                catalog_name="main",
                schema_name="s",
                table_name="t",
                stage="pre-prod",
                date="20260815",
            )
            == "main.s.t_pre_prod_20260815"
        )

    def test_case_is_folded(self):
        assert sanitise_part("MixedCase") == "mixedcase"

    def test_runs_collapse_and_edges_are_trimmed(self):
        # Collapsing matters so `pre - prod` and `pre-prod` do not render two
        # different tables; trimming so a template ending `_<stage>` cannot
        # leave a dangling underscore.
        assert sanitise_part("  pre - prod  ") == "pre_prod"
        assert sanitise_part("_leading_") == "leading"

    def test_a_dot_in_a_value_cannot_add_a_level(self):
        # Every dot in the result must be one the template wrote, or a value
        # could silently retarget the write to another schema.
        assert (
            render(
                DEFAULT_TEMPLATE,
                catalog_name="main",
                schema_name="s",
                table_name="evil.name",
            )
            == "main.s.evil_name"
        )


class TestValidation:
    def test_an_unknown_placeholder_is_rejected(self):
        with pytest.raises(TableNameError, match="unknown placeholder"):
            validate_template("<catalog_name>.<schema_name>.<tabel_name>")

    def test_the_error_lists_what_is_available(self):
        with pytest.raises(TableNameError, match="<table_name>"):
            validate_template("<catalog_name>.<schema_name>.<tabel_name>")

    @pytest.mark.parametrize(
        "template",
        [
            "<catalog_name>.<table_name>",
            "<table_name>",
            "<catalog_name>.<schema_name>.<table_name>.<stage>",
        ],
    )
    def test_a_template_that_is_not_three_level_is_rejected(self, template):
        with pytest.raises(TableNameError, match="three-level"):
            validate_template(template)

    def test_a_missing_value_for_a_used_placeholder_is_rejected(self):
        # Not rendered as an empty string: `sales__20260815` is a real table
        # name and writing to it would look like it worked.
        with pytest.raises(TableNameError, match="<stage>"):
            render(
                REQUESTED,
                catalog_name="main",
                schema_name="s",
                table_name="t",
                stage=None,
                date="20260815",
            )

    def test_a_whitespace_only_value_counts_as_missing(self):
        with pytest.raises(TableNameError, match="<stage>"):
            render(
                REQUESTED,
                catalog_name="main",
                schema_name="s",
                table_name="t",
                stage="   ",
                date="20260815",
            )

    def test_a_value_that_sanitises_to_nothing_is_rejected(self):
        # `!!!` is not empty, but it is empty *as an identifier*, and the result
        # would be `main.s.` — which must not reach a CREATE TABLE.
        with pytest.raises(TableNameError):
            render(
                DEFAULT_TEMPLATE,
                catalog_name="main",
                schema_name="s",
                table_name="!!!",
            )

    def test_render_validates_even_when_the_caller_did_not(self):
        with pytest.raises(TableNameError):
            render("<catalog_name>.<nonsense>", catalog_name="main")

    def test_placeholders_in_finds_them_all(self):
        assert placeholders_in(REQUESTED) == {
            "catalog_name",
            "schema_name",
            "table_name",
            "stage",
            "date",
        }


class TestConfigIntegration:
    def test_the_run_date_comes_from_one_fixed_instant(self):
        """A run must not straddle two dates.

        ``HydrationConfig.started_at`` is fixed at construction and
        :attr:`run_date` formats it, so reading the property repeatedly — which
        is what planning a hundred tables does — can never return two values.
        """
        from data_hydration.config import HydrationConfig

        config = HydrationConfig()
        assert config.run_date == config.run_date

    def test_a_bad_date_format_degrades_rather_than_crashing(self):
        """The house rule for a wrong config value: warn and fall back.

        The rejection is forced through a stub clock rather than written as a
        literal bad format string, because which formats ``strftime`` refuses is
        a platform question — the C library decides, so a string that raises on
        Windows may pass silently on the Linux CI runner and quietly stop
        testing anything.
        """
        from data_hydration.config import DEFAULT_DATE_FORMAT, HydrationConfig

        class _PickyClock:
            """Accepts only the default format, as some platforms would."""

            def strftime(self, fmt: str) -> str:
                if fmt != DEFAULT_DATE_FORMAT:
                    raise ValueError(f"bad directive in format {fmt!r}")
                return "20260815"

        config = HydrationConfig(date_format="%Q")
        config.started_at = _PickyClock()  # type: ignore[assignment]
        assert config.run_date == "20260815"
