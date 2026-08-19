"""Regression tests for the wheel's explicit package manifest."""

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _project_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as config_file:
        return tomllib.load(config_file)


def test_build_backend_is_declared() -> None:
    config = _project_config()
    assert config["build-system"]["build-backend"] == "setuptools.build_meta"


def test_runtime_packages_are_explicitly_included() -> None:
    config = _project_config()
    included = set(config["tool"]["setuptools"]["packages"]["find"]["include"])
    assert {"sas_migrate*", "token_budget*", "validation*"} <= included
    assert "main" in config["tool"]["setuptools"]["py-modules"]


def test_runtime_package_data_is_declared() -> None:
    package_data = _project_config()["tool"]["setuptools"]["package-data"]
    assert "complexity" not in package_data
    assert "instructions/**/*.md" in package_data["prompt_builder"]
    assert "resources/**/*.json" in package_data["sas_migrate"]
    assert (ROOT / "src/sas_migrate/resources/assessment/pyspark.json").is_file()
    assert (ROOT / "src/sas_migrate/resources/assessment/sparksql.json").is_file()


def test_quality_gates_cover_new_runtime_modules() -> None:
    config = _project_config()["tool"]
    coverage = config["coverage"]["run"]
    assert coverage["branch"] is True
    assert coverage["relative_files"] is True
    assert {"token_budget", "main", "sas_migrate"} <= set(coverage["source"])
    assert {"token_budget", "src/sas_migrate"} <= set(config["pyright"]["include"])


def test_v2_console_script_is_declared() -> None:
    scripts = _project_config()["project"]["scripts"]
    assert scripts["sas-migrate"] == "sas_migrate.cli:main"
