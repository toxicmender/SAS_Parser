"""Import the data a SAS corpus reads into Databricks. See ``data_hydration/README.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Eager: models, config and naming import nothing beyond app_config and pydantic,
# so they cost nothing and every caller needs them.
from .config import HydrationConfig
from .models import (
    HydrationItem,
    HydrationPlan,
    HydrationReport,
    HydrationSource,
    ItemOutcome,
    ItemStatus,
    Partition,
    PartitionStrategy,
    SourceKind,
    WriteMode,
)
from .naming import TableNameError, render, validate_template

if TYPE_CHECKING:  # real types for a checker, no import cost at run time
    from .inventory import (
        InventoryRow,
        RefKind,
        inventory_rows,
        plan_from_inventory,
        read_inventory,
        write_inventory,
    )
    from .planner import build_corpus_plan, build_plan
    from .runner import execute
    from .secrets import HydrationCredentialError, resolve_secret

# Lazily re-exported. The planner pulls in partition selection and the runner
# pulls in the sinks, which reach for pyspark; neither should be paid for by a
# caller that only wanted to read a HydrationPlan off a report.
_LAZY = {
    "build_plan": ".planner",
    "build_corpus_plan": ".planner",
    "InventoryRow": ".inventory",
    "RefKind": ".inventory",
    "inventory_rows": ".inventory",
    "plan_from_inventory": ".inventory",
    "read_inventory": ".inventory",
    "write_inventory": ".inventory",
    "execute": ".runner",
    "resolve_secret": ".secrets",
    "HydrationCredentialError": ".secrets",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'data_hydration' has no attribute '{name}'")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


__all__ = [
    # planning and execution
    "build_plan",
    "build_corpus_plan",
    "execute",
    # the reference inventory
    "InventoryRow",
    "RefKind",
    "inventory_rows",
    "plan_from_inventory",
    "read_inventory",
    "write_inventory",
    # models
    "HydrationItem",
    "HydrationPlan",
    "HydrationReport",
    "HydrationSource",
    "ItemOutcome",
    "ItemStatus",
    "Partition",
    "PartitionStrategy",
    "SourceKind",
    "WriteMode",
    # configuration and naming
    "HydrationConfig",
    "TableNameError",
    "render",
    "validate_template",
    # credentials
    "resolve_secret",
    "HydrationCredentialError",
]
