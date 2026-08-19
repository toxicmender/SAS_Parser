"""Phase 8 assessment profiles, sizing, graph, review, and report gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pymupdf
import pytest

from sas_migrate.adapters.assessment import (
    DirectoryAssessmentProfileRepository,
    PackageAssessmentProfileRepository,
    render_pdf,
)
from sas_migrate.application.assessment import (
    AssessmentProfileError,
    AssessmentService,
    AssessmentUnit,
    ComplexityTier,
    ConstructOccurrence,
    TranslationParity,
    dependency_edges,
    load_profile,
    render_json,
    render_markdown,
)
from sas_migrate.core.targets import TargetId


def _unit(
    source_id: str = "consumer.sas",
    *,
    constructs: tuple[ConstructOccurrence, ...] = (),
    inputs: tuple[str, ...] = ("work.stage",),
    outputs: tuple[str, ...] = ("work.result",),
) -> AssessmentUnit:
    return AssessmentUnit(
        source_id=source_id,
        line_count=80,
        chunk_count=6,
        step_count=4,
        parameter_count=2,
        input_datasets=inputs,
        output_datasets=outputs,
        constructs=constructs,
    )


def test_package_repository_exposes_both_supported_profiles() -> None:
    repository = PackageAssessmentProfileRepository()
    assert repository.names() == ("pyspark", "sparksql")
    assert repository.load("sparksql")["schema_version"] == 2


def test_pyspark_profile_inherits_complete_spark_sql_catalogue() -> None:
    profile = load_profile("pyspark", PackageAssessmentProfileRepository())
    assert profile.extends == "sparksql"
    assert profile.constructs["proc"]["sql"].parity is TranslationParity.DIRECT
    assert profile.constructs["kind"]["MACRO_DEFINITION".casefold()].parity is TranslationParity.HARD
    assert profile.sizes["scale"] == {
        "SMALL": 2,
        "MEDIUM": 3,
        "LARGE": 5,
        "EXTRA_LARGE": 8,
    }
    anchor = profile.sizes["anchor"]
    assert isinstance(anchor, dict) and anchor["raw"] == 81.5
    assert "describes" in anchor


def test_directory_profiles_merge_sizes_constructs_and_flags(tmp_path: Path) -> None:
    (tmp_path / "base.json").write_text(
        json.dumps(
            {
                "target": "base",
                "weights": {"LOW": 1},
                "sizes": {"anchor": {"raw": 10, "describes": "base"}, "scale": {"SMALL": 1}},
                "constructs": {"proc": {"sql": {"category": "sql", "tier": "LOW", "parity": "DIRECT"}}},
                "flags": [{"name": "flag", "tier": "LOW", "attr": "a"}],
            }
        ),
        "utf-8",
    )
    (tmp_path / "child.json").write_text(
        json.dumps(
            {
                "target": "child",
                "extends": "base",
                "sizes": {"anchor": {"raw": 20}},
                "flags": [{"name": "flag", "tier": "HIGH"}],
            }
        ),
        "utf-8",
    )
    profile = load_profile("child", DirectoryAssessmentProfileRepository(tmp_path))
    assert profile.sizes["anchor"] == {"raw": 20, "describes": "base"}
    assert "sql" in profile.constructs["proc"]
    assert profile.flags[0]["attr"] == "a" and profile.flags[0]["tier"] == "HIGH"


def test_profile_cycles_are_rejected() -> None:
    class Cycles:
        def load(self, name: str) -> dict[str, object]:
            return {"target": name, "extends": "b" if name == "a" else "a"}

        def names(self) -> tuple[str, ...]:
            return ("a", "b")

    with pytest.raises(AssessmentProfileError, match="circular"):
        load_profile("a", Cycles())


def test_cross_file_dataset_dependencies_are_deduplicated() -> None:
    producer = _unit("producer.sas", inputs=(), outputs=("WORK.STAGE",))
    consumer = _unit(inputs=("work.stage", "work.stage"), outputs=())
    edges = dependency_edges((producer, consumer))
    assert len(edges) == 1
    assert edges[0].model_dump() == {
        "producer": "producer.sas",
        "consumer": "consumer.sas",
        "dataset": "work.stage",
    }


def test_assessment_applies_target_specific_parity_and_shared_dependencies() -> None:
    producer = _unit("producer.sas", inputs=(), outputs=("work.stage",))
    macro = ConstructOccurrence(kind="kind", name="MACRO_DEFINITION")
    consumer = _unit(constructs=(macro,))
    service = AssessmentService(PackageAssessmentProfileRepository())
    sql = service.assess((producer, consumer), TargetId.SPARK_SQL)
    python = service.assess((producer, consumer), TargetId.PYSPARK)
    assert sql.files[1].tier is ComplexityTier.HIGH
    assert sql.files[1].parity is TranslationParity.MANUAL
    assert python.files[1].parity is TranslationParity.HARD
    assert sql.dependencies == python.dependencies
    assert sql.files[1].dependencies == sql.dependencies


def test_unknown_constructs_do_not_invent_profile_rules() -> None:
    report = AssessmentService(PackageAssessmentProfileRepository()).assess(
        (_unit(constructs=(ConstructOccurrence(kind="proc", name="not-real"),)),),
        TargetId.PYSPARK,
    )
    assert report.files[0].signals == ()
    assert report.files[0].tier is ComplexityTier.LOW
    assert report.files[0].parity is TranslationParity.DIRECT


def test_uncertainty_increases_raw_score() -> None:
    base = _unit(inputs=(), outputs=())
    uncertain = base.model_copy(
        update={"unresolved_references": ("external.macro",), "diagnostics": ("unclosed",)}
    )
    service = AssessmentService(PackageAssessmentProfileRepository())
    left = service.assess((base,), TargetId.PYSPARK).files[0]
    right = service.assess((uncertain,), TargetId.PYSPARK).files[0]
    assert right.raw_score > left.raw_score


def test_assessment_json_round_trips_without_computed_fields() -> None:
    report = AssessmentService(PackageAssessmentProfileRepository()).assess(
        (_unit(),), TargetId.PYSPARK
    )
    payload = json.loads(render_json(report))
    assert "total_story_points" not in payload
    assert type(report).from_json(render_json(report)) == report


def test_markdown_and_pdf_render_sizing_and_dependency_tables() -> None:
    service = AssessmentService(PackageAssessmentProfileRepository())
    report = service.assess(
        (
            _unit("producer.sas", inputs=(), outputs=("work.stage",)),
            _unit("consumer.sas", inputs=("work.stage",), outputs=()),
        ),
        TargetId.PYSPARK,
    )
    markdown = render_markdown(report)
    assert "total story points" in markdown
    assert "| producer.sas | consumer.sas | work.stage |" in markdown
    with pymupdf.open(stream=render_pdf(report), filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    assert "Migration assessment" in text
    assert "Cross-file dependencies" in text


def test_optional_review_uses_port_and_returns_a_new_report() -> None:
    class Reviewer:
        prompt = ""

        async def review(self, prompt: str) -> str:
            self.prompt = prompt
            return "Migrate producer first."

    service = AssessmentService(PackageAssessmentProfileRepository())
    report = service.assess((_unit(),), TargetId.PYSPARK)
    reviewer = Reviewer()
    reviewed = asyncio.run(service.review(report, reviewer))
    assert reviewed.review == "Migrate producer first."
    assert report.review is None
    assert "consumer.sas" in reviewer.prompt


def test_legacy_complexity_uses_the_single_v2_owned_catalogue() -> None:
    from complexity.rules import PROFILE_DIR, load_ruleset

    assert PROFILE_DIR.name == "assessment"
    assert "sas_migrate" in PROFILE_DIR.parts
    assert load_ruleset("pyspark", use_cache=False).target == "pyspark"
