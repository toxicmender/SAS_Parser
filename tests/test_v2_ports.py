"""Direct tests for v2 application-port values and the CLI composition shell."""

from __future__ import annotations

import pathlib
import sys

import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.application.ports import (
    ArtifactWrite,
    CredentialValue,
    ProviderResponse,
    ProviderTokenUsage,
    SourceObject,
)
from sas_migrate.cli import build_parser, main


def test_port_value_models_are_strict_and_secret_safe() -> None:
    credential = CredentialValue(
        name="gateway-token",
        value="do-not-log-this",
        source="test",
    )
    assert "do-not-log-this" not in repr(credential)
    assert "do-not-log-this" not in credential.model_dump_json()

    with pytest.raises(ValidationError):
        ArtifactWrite(artifact_id="", media_type="application/json", content=b"{}")


def test_source_and_artifact_port_values_round_trip() -> None:
    source = SourceObject(
        source_id="source-1",
        name="program.sas",
        content=b"data out; run;",
        metadata={"library": "WORK"},
    )
    restored_source = SourceObject.model_validate_json(source.model_dump_json())
    assert restored_source == source

    artifact = ArtifactWrite(
        artifact_id="response-1",
        media_type="application/json",
        content=b'{"schema_version":2}',
        metadata={"kind": "response_audit"},
    )
    assert ArtifactWrite.model_validate_json(artifact.model_dump_json()) == artifact


def test_provider_response_keeps_usage_optional_and_separate() -> None:
    response = ProviderResponse(
        raw_message="provider output",
        structured_error="schema not supported",
        usage=ProviderTokenUsage(
            input_tokens=120,
            output_tokens=30,
            cache_read_tokens=50,
        ),
    )
    assert response.usage.input_tokens == 120
    assert response.usage.cache_read_tokens == 50
    assert response.structured_document is None


def test_cli_shell_parses_version_and_has_no_subcommands(capsys) -> None:
    parser = build_parser()
    assert parser.prog == "sas-migrate"
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "operational commands are not enabled yet" in output
    assert "{convert" not in output

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert "sas-migrate 0.1.0" in capsys.readouterr().out
