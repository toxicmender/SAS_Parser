"""MemorySetup resolution of the typed Delta table configuration."""

import json

import app_config
from memory.store import MemoryHub
from pipeline.setup import MemorySetup


def test_memory_setup_uses_configured_delta_tables(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "memory": {
                    "delta_table": "main.agent.memory",
                    "cdf_audit_table": "main.agent.memory_audit",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(app_config.ENV_VAR, str(config))
    app_config.clear_cache()
    captured = {}

    def fake_hub(spark, table, **kwargs):
        captured.update(spark=spark, table=table, **kwargs)
        return MemoryHub()

    monkeypatch.setattr(MemorySetup, "_default_hub", staticmethod(fake_hub))
    try:
        MemorySetup(cdf_consumer_id="worker_1").build()
    finally:
        app_config.clear_cache()

    assert captured["table"] == "main.agent.memory"
    assert captured["cdf_audit_table"] == "main.agent.memory_audit"
    assert captured["cdf_consumer_id"] == "worker_1"
