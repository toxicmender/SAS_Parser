"""No-skip install/import matrix for hydration's optional source drivers."""

from __future__ import annotations

import importlib

import pytest

from sas_migrate.adapters.hydration import OPTIONAL_DEPENDENCIES
from sas_migrate.application.hydration import SourceKind


@pytest.mark.parametrize("kind", tuple(SourceKind))
def test_every_hydration_driver_dependency_is_installed_and_importable(kind: SourceKind) -> None:
    module_name, extra = OPTIONAL_DEPENDENCIES[kind]
    assert extra == "hydration"
    assert importlib.import_module(module_name) is not None
