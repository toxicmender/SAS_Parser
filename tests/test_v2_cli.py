"""Operational contracts for the Phase 10 v2 CLI composition root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sas_migrate.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assessment_units(path: Path) -> Path:
    return _write_json(
        path,
        [
            {
                "source_id": "producer.sas",
                "line_count": 12,
                "chunk_count": 1,
                "step_count": 1,
                "output_datasets": ["work.customer"],
            },
            {
                "source_id": "consumer.sas",
                "line_count": 8,
                "chunk_count": 1,
                "step_count": 1,
                "input_datasets": ["work.customer"],
            },
        ],
    )


def _evaluation_run(path: Path, *, response: str = "```sql\nSELECT 1\n```") -> Path:
    return _write_json(
        path,
        {
            "schema_version": 2,
            "run_id": "offline-case",
            "target": "spark_sql",
            "units": [
                {
                    "unit_id": "unit-1",
                    "source": "proc sql; select 1; quit;",
                    "response": response,
                }
            ],
        },
    )


def test_parser_exposes_operational_commands_and_only_supported_targets() -> None:
    parser = build_parser()
    args = parser.parse_args(["assess", "units.json", "--target", "pyspark"])
    assert args.command == "assess"
    assert args.target == "pyspark"

    with pytest.raises(SystemExit) as unsupported:
        parser.parse_args(["assess", "units.json", "--target", "spark-scala"])
    assert unsupported.value.code == 2


def test_assess_emits_json_and_markdown_from_packaged_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _assessment_units(tmp_path / "units.json")

    assert main(["assess", str(input_path), "--target", "pyspark"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["target"] == "pyspark"
    assert report["profile"] == "pyspark"
    assert report["dependencies"] == [
        {
            "producer": "producer.sas",
            "consumer": "consumer.sas",
            "dataset": "work.customer",
        }
    ]

    output = tmp_path / "assessment.md"
    assert (
        main(
            [
                "assess",
                str(input_path),
                "--format",
                "markdown",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Migration assessment" in output.read_text("utf-8")

    assert (
        main(
            [
                "assess",
                str(input_path),
                "--profiles",
                str(ROOT / "src" / "sas_migrate" / "resources" / "assessment"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    assert "Migration assessment" in capsys.readouterr().out


def test_assess_pdf_requires_a_path_and_writes_a_real_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _assessment_units(tmp_path / "units.json")
    assert main(["assess", str(input_path), "--format", "pdf"]) == 2
    assert "PDF output requires --output PATH" in capsys.readouterr().err

    output = tmp_path / "assessment.pdf"
    assert (
        main(
            [
                "assess",
                str(input_path),
                "--format",
                "pdf",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes().startswith(b"%PDF")


def test_validate_emits_report_and_returns_failure_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing = _evaluation_run(tmp_path / "passing.json")
    assert main(["validate", str(passing), "--model", "offline-test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "offline-test"
    assert report["target"] == "spark_sql"
    assert all(metric["passed"] for metric in report["results"][0]["metrics"])

    failing = _evaluation_run(tmp_path / "failing.json", response="not target code")
    assert main(["validate", str(failing)]) == 1
    failed_report = json.loads(capsys.readouterr().out)
    language = next(
        metric
        for metric in failed_report["results"][0]["metrics"]
        if metric["metric"] == "language_compliance"
    )
    assert not language["passed"]


def test_validate_reports_component_token_budget_and_pdf(
    tmp_path: Path,
) -> None:
    run = _evaluation_run(tmp_path / "run.json")
    ledger = _write_json(
        tmp_path / "ledger.json",
        {"schema_version": 2, "records": []},
    )
    policy = _write_json(tmp_path / "policy.json", {"max_run_tokens": 10})
    markdown = tmp_path / "validation.md"

    assert (
        main(
            [
                "validate",
                str(run),
                "--translation-ledger",
                str(ledger),
                "--translation-policy",
                str(policy),
                "--format",
                "markdown",
                "--output",
                str(markdown),
            ]
        )
        == 0
    )
    text = markdown.read_text("utf-8")
    assert "Translation token budget" in text
    assert "token_budget_compliance: **PASS**" in text

    pdf = tmp_path / "validation.pdf"
    assert (
        main(
            [
                "validate",
                str(run),
                "--format",
                "pdf",
                "--output",
                str(pdf),
            ]
        )
        == 0
    )
    assert pdf.read_bytes().startswith(b"%PDF")


def test_cli_returns_operator_error_for_invalid_or_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert main(["assess", str(invalid)]) == 2
    assert "invalid assessment units" in capsys.readouterr().err

    assert main(["validate", str(tmp_path / "missing.json")]) == 2
    assert "could not read" in capsys.readouterr().err

    empty = _write_json(tmp_path / "empty.json", [])
    assert main(["assess", str(empty)]) == 2
    assert "at least one unit" in capsys.readouterr().err


def test_cli_reports_invalid_optional_contract_and_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _evaluation_run(tmp_path / "run.json")
    invalid_ledger = _write_json(tmp_path / "ledger.json", {"records": "invalid"})
    assert (
        main(["validate", str(run), "--translation-ledger", str(invalid_ledger)])
        == 2
    )
    assert "invalid TokenCallLedger" in capsys.readouterr().err

    missing_parent = tmp_path / "missing" / "report.json"
    assert main(["validate", str(run), "--output", str(missing_parent)]) == 2
    assert "could not write report" in capsys.readouterr().err


def test_smoke_human_and_quiet_presentations(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke"]) == 0
    output = capsys.readouterr().out
    assert "v2 deployment smoke: PASSED" in output
    assert "v2_application_flow: pass" in output

    assert main(["smoke", "--quiet"]) == 0
    assert capsys.readouterr().out == ""
