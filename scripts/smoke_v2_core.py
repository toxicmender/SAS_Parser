"""Import representative v2 core contracts with only Pydantic available."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sas_migrate.core.responses import ResponseEnvelope, TranslationDocument
from sas_migrate.core.runs import RunEvent
from sas_migrate.core.targets import ResolvedTarget, resolve_local_target
from sas_migrate.core.tokens import CallTokenRecord, TokenBudgetPolicy


def main() -> int:
    contracts = (
        CallTokenRecord,
        ResolvedTarget,
        ResponseEnvelope,
        RunEvent,
        TokenBudgetPolicy,
        TranslationDocument,
    )
    if any(contract.model_fields["schema_version"].default != 2 for contract in contracts):
        raise RuntimeError("a v2 core wire contract lost schema_version=2")
    if resolve_local_target().target.value != "spark_sql":
        raise RuntimeError("the v2 default target is not Spark SQL")
    forbidden = (
        "sas_migrate.adapters",
        "sas_migrate.application",
        "sas_migrate.config",
        "sas_migrate.observability",
    )
    imported = sorted(name for name in sys.modules if name.startswith(forbidden))
    if imported:
        raise RuntimeError(f"core import loaded outer layers: {imported}")
    print("v2 core-only import passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
