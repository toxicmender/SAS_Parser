"""Offline tests for Delta-production controls in memory.store/operations."""

import sys
from types import SimpleNamespace

import pytest

from memory.databricks_ai import chat_model, embeddings
from memory.operations import (
    MAX_VACUUM_HOURS,
    MIN_VACUUM_HOURS,
    VacuumPolicy,
)
from memory.store import KVStore, MemoryHub, _quoted_table_name


def test_delta_identifier_is_quoted_part_by_part():
    assert _quoted_table_name("main.memory.history") == "`main`.`memory`.`history`"


@pytest.mark.parametrize(
    "table",
    ["", "main..history", "main.memory; DROP TABLE x", "main.memory.history.extra"],
)
def test_delta_identifier_rejects_sql_and_invalid_shapes(table):
    with pytest.raises(ValueError):
        _quoted_table_name(table)


def test_cdf_requires_delta_and_independent_audit_table():
    with pytest.raises(ValueError, match="audit_table requires"):
        KVStore(audit_table="main.memory.audit")

    with pytest.raises(RuntimeError, match="Delta-backed"):
        KVStore().sync_cdf("worker")

    with pytest.raises(ValueError, match="requires cdf_audit_table"):
        MemoryHub(cdf_consumer_id="worker")


def test_vacuum_policy_enforces_delta_and_cdf_safety_boundaries():
    assert VacuumPolicy(MIN_VACUUM_HOURS).retention_hours == MIN_VACUUM_HOURS
    assert VacuumPolicy(MAX_VACUUM_HOURS).retention_hours == MAX_VACUUM_HOURS

    with pytest.raises(ValueError, match="between"):
        VacuumPolicy(MIN_VACUUM_HOURS - 1)
    with pytest.raises(ValueError, match="between"):
        VacuumPolicy(MAX_VACUUM_HOURS + 1)
    with pytest.raises(ValueError, match="must exceed"):
        VacuumPolicy(MIN_VACUUM_HOURS, max_cdf_outage_hours=MIN_VACUUM_HOURS)


def test_databricks_ai_factories_are_lazy_and_preserve_adapter_settings(monkeypatch):
    calls = []

    def make(kind):
        def factory(**kwargs):
            calls.append((kind, kwargs))
            return kwargs

        return factory

    monkeypatch.setitem(
        sys.modules,
        "databricks_langchain",
        SimpleNamespace(
            ChatDatabricks=make("chat"),
            DatabricksEmbeddings=make("embeddings"),
        ),
    )

    assert chat_model("chat-endpoint", temperature=0, max_retries=2) == {
        "endpoint": "chat-endpoint",
        "temperature": 0,
        "max_retries": 2,
    }
    assert embeddings("embedding-endpoint", query_params={"truncate": True}) == {
        "endpoint": "embedding-endpoint",
        "target_uri": "databricks",
        "query_params": {"truncate": True},
    }
    assert [kind for kind, _ in calls] == ["chat", "embeddings"]
