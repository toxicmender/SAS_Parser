"""Shared pytest fixtures.

The one thing every test needs: independence from the repo's ``config.json``.

That file used to be an all-null template, so tests that never mentioned it
behaved as though no config existed. It no longer is — it enables the bundled
SparkSQL instruction set and sizes the retrieval budget for it — and a unit
test asserting "no prompt builder means no guidance" would otherwise fail for
a reason that has nothing to do with the code under test.

Isolating to an empty config restores exactly the old baseline while making
the dependency explicit: a test that wants a config value now sets one.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import app_config


@pytest.fixture(autouse=True)
def _isolate_repo_config(monkeypatch, tmp_path_factory):
    """Point every test at an empty config, whatever the repo ships."""
    cfg = tmp_path_factory.mktemp("cfg") / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(app_config.ENV_VAR, str(cfg))
    app_config.clear_cache()
    yield
    app_config.clear_cache()
