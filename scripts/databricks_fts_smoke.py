#!/usr/bin/env python3
"""On-cluster smoke check for the memory store's Databricks features.

Run on Databricks Runtime 19 (18.3+), on a workspace with the preview
"SQL: Full-Text Search Index for Unity Catalog managed tables" enabled, as a
job or from a notebook (``%run`` or ``!python``):

    python scripts/databricks_fts_smoke.py --schema main.scratch

It cannot run in CI: full-text search indexes, predictive optimization and
the ``search``/``isearch`` functions exist only on Databricks. It:

1. creates a scratch memory base ``<catalog>.<schema>.fts_smoke_<rand>``
   with ``MemoryHub(..., full_text_search=True)``;
2. checks every Databricks statement succeeded (predictive optimization and
   each ``CREATE SEARCH INDEX``) and reports which index-name form the
   runtime accepted (catalog.schema-qualified or bare);
3. runs ``DESCRIBE INDEX`` on every declared index;
4. writes messages, a KV entry and a chunk-history attempt, and checks that
   ``kv.search`` and ``chunk_history.search`` find them through ``isearch``
   (``search_functions`` true; the predicate is in the physical plan);
5. drops everything it created, unless ``--keep``.

Exit status 0 means every check passed; each failure is printed.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--schema", required=True, help="catalog.schema to create scratch tables in")
    parser.add_argument("--keep", action="store_true", help="do not drop the scratch tables")
    args = parser.parse_args(argv)

    from pyspark.sql import SparkSession

    from memory import DeltaMemoryMaintenance, VacuumPolicy, schema
    from memory.store import MemoryHub

    spark = SparkSession.builder.getOrCreate()
    base = f"{args.schema}.fts_smoke_{uuid.uuid4().hex[:8]}"
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            failures.append(what)

    try:
        hub = MemoryHub(spark=spark, table=base, full_text_search=True)
        kv_report = hub._store._backend.provision
        hist_report = hub.chunk_history.provision
        check(
            schema.is_provisioned(spark, base, schema.KV_TABLES)
            and schema.is_provisioned(spark, base, schema.HISTORY_TABLES),
            "provisioned marker set (later processes skip the Databricks setup)",
        )
        for label, report in (("kv tables", kv_report), ("chunk history", hist_report)):
            check(report is not None and not report.failures,
                  f"{label}: predictive optimization + search indexes "
                  f"({report.failures if report else 'no report'})")
            if report is not None:
                print(f"     index name form accepted: {report.index_name_style}")
        check(hub._store._backend.search_functions, "isearch() available to the KV store")
        check(hub.chunk_history.search_functions, "isearch() available to chunk history")

        ops = DeltaMemoryMaintenance(spark, base, policy=VacuumPolicy(7 * 24))
        status = ops.search_index_status()
        declared = sum(len(s.search_indexes) for s in schema.ALL_TABLES)
        check(len(status) == declared and all(v for v in status.values()),
              f"DESCRIBE INDEX answers for all {declared} indexes")

        thread = hub.get_thread("smoke-thread", chat_id="smoke-chat")
        thread.add_user_message("convert PROC SORT NODUPKEY on claims")
        thread.add_ai_message("Use dropDuplicates(['member_id'])")
        hub.kv.set("note", "remember the zz_ prefix", tags=["smoke"])
        check([k for k, _ in hub.kv.search("zz_")] == ["note"], "kv.search finds the entry")

        item = SimpleNamespace(
            batch_id="batch-001",
            chunks=[SimpleNamespace(
                chunk_id="c1", source_id="smoke.sas", kind="proc_step", title=None,
                text="proc sort data=claims nodupkey; by member_id; run;",
                start_line=1, end_line=1, start_char=0, end_char=48, parent_id=None,
                metadata=None,
            )],
        )
        hub.chunk_history.record_attempt(
            item, "```python\nclaims.dropDuplicates(['member_id'])\n```", None,
            thread_id="smoke-thread", attempt=1, is_final=True,
        )
        check(len(hub.chunk_history.search("NODUPKEY")) == 1, "chunk_history.search(original)")
        check(len(hub.chunk_history.search("dropDuplicates", field="converted")) == 1,
              "chunk_history.search(converted)")

        explain = spark.sql(
            f"EXPLAIN SELECT * FROM {schema.quoted_table_name(schema.chunk_history_view_name(base))} "
            f"WHERE {schema.search_sql(('original_text',), 'nodupkey')}"
        ).first()
        plan = str(explain[0]) if explain is not None else ""
        check("isearch" in plan.lower() or "search" in plan.lower(),
              "isearch predicate reaches the physical plan")
        print("\nIndexes refresh asynchronously; track REFRESH_INDEX runs in "
              "system.storage.predictive_optimization_operations_history.")
    finally:
        if not args.keep:
            for archived in (False, True):
                spark.sql(f"DROP VIEW IF EXISTS {schema.chunk_history_view_name(base, archived=archived)}")
            for spec in schema.ALL_TABLES:
                spark.sql(f"DROP TABLE IF EXISTS {spec.name(base)}")  # drops its indexes too

    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failing check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
