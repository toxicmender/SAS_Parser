"""Spark SQL names its permanent tables through session variables.

The Spark SQL instructions ask for every table outside WORK to be named
through Databricks SQL session variables — a ``<LIBREF>_PREFIX`` per SAS
library, a variable per table built from it, and ``IDENTIFIER(session.<VAR>)``
wherever the table is read or written — so its catalog and schema are set in
one place. ``session_variables`` is the check that generated code does.
"""

from __future__ import annotations

import pathlib

import pytest

from target_language import SPARKSQL
from validation import (
    EvaluationRun,
    Evaluator,
    SessionVariablesMetric,
    default_metrics,
)
from validation.metrics import _fenced_blocks
from validation.table_names import check_table_names

_FENCE = chr(96) * 3

GOOD = """\
DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';
DECLARE OR REPLACE VARIABLE RAW_SALES STRING DEFAULT CONCAT(session.MART_PREFIX, '.raw_sales');
DECLARE OR REPLACE VARIABLE SALES_CLEAN STRING DEFAULT CONCAT(session.MART_PREFIX, '.sales_clean');

CREATE OR REPLACE TABLE IDENTIFIER(session.SALES_CLEAN) AS
SELECT * FROM IDENTIFIER(session.RAW_SALES) WHERE amount > 0;

CREATE OR REPLACE TEMP VIEW summary AS
WITH s AS (SELECT region, amount FROM IDENTIFIER(session.SALES_CLEAN))
SELECT region, SUM(amount) AS total FROM s GROUP BY region;
"""


def _response(*blocks: str, fence: str = "sql") -> str:
    body = "\n\n".join(f"{_FENCE}{fence}\n{block}\n{_FENCE}" for block in blocks)
    return f"## Analysis\n\nx\n\n## Translation\n\n{body}\n\n## Risks\n\nnone"


def _run(*responses: str) -> EvaluationRun:
    return EvaluationRun(
        run_id="case",
        prompts=["data mart.x; set mart.y; run;"] * len(responses),
        outputs=[{"response": r} for r in responses],
    )


def _score(*blocks: str):
    return SessionVariablesMetric(output_language="Spark SQL").evaluate(
        _run(_response(*blocks))
    )


# ── what passes ──────────────────────────────────────────────────────────────


def test_tables_named_through_declared_variables_pass():
    result = _score(GOOD)
    assert result.metric == "session_variables"
    assert (result.score, result.passed, result.skipped) == (1.0, True, False)
    assert result.details.startswith("3/3 permanent table reference(s)")


def test_a_table_variable_given_its_value_by_set_var_passes():
    result = _score(
        "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
        "DECLARE OR REPLACE VARIABLE DEMO_TABLE STRING;\n"
        "SET VAR DEMO_TABLE = CONCAT(session.MART_PREFIX, '.demo_table');\n"
        "INSERT INTO IDENTIFIER(session.DEMO_TABLE) SELECT 1 AS a;"
    )
    assert result.score == 1.0


def test_every_write_the_identifier_clause_takes_passes():
    prefix = (
        "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
        "DECLARE OR REPLACE VARIABLE T STRING DEFAULT CONCAT(session.MART_PREFIX, '.t');\n"
        "DECLARE OR REPLACE VARIABLE S STRING DEFAULT CONCAT(session.MART_PREFIX, '.s');\n"
    )
    statements = [
        "MERGE INTO IDENTIFIER(session.T) AS t USING IDENTIFIER(session.S) AS s "
        "ON t.k = s.k WHEN MATCHED THEN UPDATE SET *;",
        "UPDATE IDENTIFIER(session.T) SET a = 1 WHERE b = 2;",
        "DELETE FROM IDENTIFIER(session.T) WHERE a = 1;",
        "COPY INTO IDENTIFIER(session.T) FROM '/Volumes/c/s/v/in/' FILEFORMAT = CSV;",
        "DROP TABLE IF EXISTS IDENTIFIER(system.session.T);",
        "TRUNCATE TABLE IDENTIFIER(session.T);",
        "CREATE TABLE IF NOT EXISTS IDENTIFIER(session.T) (a INT, b STRING);",
    ]
    result = _score(prefix + "\n".join(statements))
    assert (result.score, result.details.split(" ")[0]) == (1.0, "8/8")


def test_a_declaration_in_an_earlier_block_of_the_answer_counts():
    result = _score(
        "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';",
        "DECLARE OR REPLACE VARIABLE T STRING DEFAULT CONCAT(session.MART_PREFIX, '.t');\n"
        "SELECT * FROM IDENTIFIER(session.T);",
    )
    assert result.score == 1.0


def test_a_stored_view_built_through_execute_immediate_passes():
    result = _score(
        "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
        "DECLARE OR REPLACE VARIABLE A STRING DEFAULT CONCAT(session.MART_PREFIX, '.a');\n"
        "DECLARE OR REPLACE VARIABLE B STRING DEFAULT CONCAT(session.MART_PREFIX, '.b');\n"
        "DECLARE OR REPLACE VARIABLE STMT STRING;\n"
        "SET VAR STMT = CONCAT('CREATE OR REPLACE VIEW ', session.A, "
        "' AS SELECT * FROM ', session.B);\n"
        "EXECUTE IMMEDIATE STMT;\n"
        "OPTIMIZE IDENTIFIER(session.B) FULL;"
    )
    assert (result.score, result.details.split(" ")[0]) == (1.0, "1/1")


def test_a_qualified_declaration_is_read():
    # sqlglot reads `DECLARE ... session.x` only as a command.
    result = _score(
        "DECLARE OR REPLACE session.MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
        "DECLARE OR REPLACE VARIABLE T STRING DEFAULT CONCAT(session.MART_PREFIX, '.t');\n"
        "SELECT * FROM IDENTIFIER(session.T);"
    )
    assert result.score == 1.0


@pytest.mark.parametrize(
    "sql",
    [
        # A WORK dataset: a temporary view, read by a bare name.
        "CREATE OR REPLACE TEMP VIEW a AS SELECT 1 AS x;\nSELECT * FROM a;",
        # A CTE, read_files, Databricks' own catalog, a file read in place.
        "WITH x AS (SELECT 1 AS a) SELECT * FROM x;",
        "SELECT * FROM read_files('/Volumes/c/s/v/in/', format => 'csv');",
        "SELECT * FROM system.information_schema.tables;",
        "SELECT * FROM delta.`/Volumes/c/s/v/t`;",
        "VACUUM delta.`/Volumes/c/s/v/t`;",
        # A schema or a catalog is no table.
        "CREATE SCHEMA IF NOT EXISTS cat.mart;\nUSE CATALOG main;",
    ],
)
def test_what_is_no_permanent_table_is_skipped(sql):
    result = _score(sql)
    assert (result.skipped, result.passed) == (True, True)
    assert result.details == "no permanent table named"


# ── what fails, and says so ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql, problem",
    [
        (
            "CREATE TABLE main.sales.orders AS SELECT * FROM IDENTIFIER(session.X);",
            "hard-coded table name `main.sales.orders`",
        ),
        ("SELECT * FROM mart.orders;", "hard-coded table name `mart.orders`"),
        (
            "CREATE OR REPLACE TEMP VIEW v AS SELECT * FROM mart.raw;",
            "hard-coded table name `mart.raw`",
        ),
        (
            "CREATE OR REPLACE TABLE orders AS SELECT 1 AS a;",
            "`orders` is written as a permanent table by a literal name",
        ),
        ("INSERT INTO orders SELECT 1 AS a;", "`orders` is written as a permanent table"),
        (
            "SELECT * FROM IDENTIFIER('main.sales.orders');",
            "IDENTIFIER('main.sales.orders') hard-codes the name",
        ),
        (
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`session.ORDERS` is not declared before it is used",
        ),
        (
            "SELECT * FROM IDENTIFIER(session.ORDERS);\n"
            "DECLARE OR REPLACE VARIABLE ORDERS STRING DEFAULT CONCAT(session.P, '.o');",
            "`session.ORDERS` is not declared before it is used",
        ),
        (
            "DECLARE OR REPLACE VARIABLE ORDERS STRING DEFAULT 'main.sales.orders';\n"
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`ORDERS` holds a literal name: build it from its library's prefix",
        ),
        (
            "DECLARE OR REPLACE VARIABLE ORDERS STRING;\n"
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`ORDERS` is never given a value",
        ),
        (
            "DECLARE OR REPLACE VARIABLE ORDERS STRING "
            "DEFAULT CONCAT(session.MART_PREFIX, '.orders');\n"
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`ORDERS` is built from `session.MART_PREFIX`, which is not declared",
        ),
        # Databricks keeps a stored view's text, which may not read a session
        # variable ("the body of a persisted view").
        (
            "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
            "DECLARE OR REPLACE VARIABLE A STRING DEFAULT CONCAT(session.MART_PREFIX, '.a');\n"
            "DECLARE OR REPLACE VARIABLE B STRING DEFAULT CONCAT(session.MART_PREFIX, '.b');\n"
            "CREATE OR REPLACE VIEW IDENTIFIER(session.A) AS SELECT * FROM IDENTIFIER(session.B);",
            "a stored view's query reads IDENTIFIER(session.B)",
        ),
        # OPTIMIZE and VACUUM, which sqlglot reads only as commands.
        ("OPTIMIZE main.mart.orders FULL;", "hard-coded table name `main.mart.orders`"),
        ("VACUUM TABLE mart.orders RETAIN 168 HOURS;", "hard-coded table name `mart.orders`"),
    ],
)
def test_a_name_that_skips_the_variables_fails_and_is_named(sql, problem):
    result = _score(sql)
    assert result.passed is False and result.score < 1.0
    assert problem in result.details


def test_the_score_is_the_share_of_references_that_pass():
    result = _score(GOOD + "\nINSERT INTO mart.audit SELECT * FROM IDENTIFIER(session.SALES_CLEAN);")
    assert result.score == pytest.approx(4 / 5)
    assert result.details.startswith("4/5 permanent table reference(s)")


def test_details_list_five_problems_then_count_the_rest():
    sql = "\n".join(f"SELECT * FROM mart.t{i};" for i in range(7))
    result = _score(sql)
    assert result.details.count("hard-coded table name") == 5
    assert result.details.endswith("; and 2 more")


def test_details_name_a_repeated_problem_once():
    # One table hard-coded six times still leaves room for the other one.
    sql = "\n".join(["SELECT * FROM mart.orders;"] * 6 + ["SELECT * FROM mart.audit;"])
    result = _score(sql)
    assert result.score == 0.0
    assert result.details.startswith("0/7 permanent table reference(s)")
    assert result.details.count("`mart.orders`") == 1
    assert "IDENTIFIER(session.<VAR>) (6 places)" in result.details
    assert "`mart.audit`" in result.details


def test_an_unparsed_block_is_reported_and_left_to_target_syntax():
    result = _score("SELECT * FROM (((", "SELECT * FROM mart.orders;")
    assert result.passed is False
    assert result.details.endswith("1 block(s) not parsed (see target_syntax)")


def test_only_the_translation_s_sql_blocks_are_read():
    response = (
        "## Analysis\n\n"
        f"{_FENCE}sql\nSELECT * FROM mart.illustration;\n{_FENCE}\n\n"
        f"## Translation\n\n{_FENCE}sql\n{GOOD}\n{_FENCE}\n"
    )
    result = SessionVariablesMetric(output_language="Spark SQL").evaluate(_run(response))
    assert result.score == 1.0


def test_responses_are_scored_together():
    result = SessionVariablesMetric(output_language="Spark SQL").evaluate(
        _run(_response(GOOD), _response("SELECT * FROM mart.orders;"))
    )
    assert result.details.startswith("3/4 permanent table reference(s)")


# ── where it runs ────────────────────────────────────────────────────────────


def test_it_is_part_of_the_spark_sql_suite_only():
    sparksql = [m.name for m in default_metrics(output_language="Spark SQL")]
    pyspark = [m.name for m in default_metrics(output_language="PySpark")]
    assert sparksql[sparksql.index("target_syntax") + 1] == "session_variables"
    assert "session_variables" not in pyspark


def test_a_pyspark_run_is_skipped():
    result = SessionVariablesMetric(output_language="PySpark").evaluate(
        _run(_response("df = spark.table('mart.orders')", fence="python"))
    )
    assert (result.skipped, result.details) == (True, "not a Spark SQL run (PySpark)")


def test_the_threshold_reads_config(monkeypatch):
    import app_config

    def get(section, key, default=None):
        if (section, key) == ("validation", "session_variables_threshold"):
            return 0.5
        return default

    monkeypatch.setattr(app_config, "get_value", get)
    assert SessionVariablesMetric().threshold == 0.5
    assert SessionVariablesMetric(threshold=0.9).threshold == 0.9


def test_a_failing_answer_tells_the_retry_what_to_fix():
    # What the pipeline sends back before a retry names the hard-coded table.
    from pipeline.engine import SasLLMPipeline

    result = Evaluator(output_language="Spark SQL").evaluate(
        _run(_response("CREATE TABLE main.mart.orders AS SELECT 1 AS a;"))
    )
    note = SasLLMPipeline._validation_feedback_message(result).content
    assert "**session_variables**" in note
    assert "hard-coded table name `main.mart.orders`" in note


def test_dataset_fidelity_counts_a_table_named_through_a_variable():
    # A variable prefixed with its library (two libraries share the member
    # name) leaves the member only as the CONCAT's quoted suffix.
    from validation.metrics import _mentions

    response = (
        "DECLARE OR REPLACE VARIABLE MART_CUSTOMERS STRING "
        "DEFAULT CONCAT(session.MART_PREFIX, '.customers');"
    )
    assert _mentions("mart.customers", response)
    assert not _mentions("mart.orders", response)


def test_check_table_names_reads_blocks_in_order():
    found = check_table_names(
        [
            "SELECT * FROM IDENTIFIER(session.T);",
            "DECLARE OR REPLACE VARIABLE T STRING DEFAULT CONCAT(session.P, '.t');",
        ]
    )
    assert (found.checked, len(found.problems), found.unparsed) == (1, 1, 0)


# ── the instructions practise the rule ──────────────────────────────────────

_SPARKSQL_INSTRUCTIONS = (
    pathlib.Path(__file__).resolve().parents[1] / "prompt_builder/instructions/sparksql"
)


def _instruction_sql() -> list[tuple[str, str]]:
    """``(file#n, body)`` for every Spark SQL block the bundled instructions
    show: what the model copies its naming from."""
    found: list[tuple[str, str]] = []
    for path in sorted(_SPARKSQL_INSTRUCTIONS.glob("*.md")):
        blocks = _fenced_blocks(path.read_text(encoding="utf-8"))
        found += [
            (f"{path.stem}#{n}", body)
            for n, (info, body) in enumerate(blocks)
            if SPARKSQL.owns_fence(info)
        ]
    return found


@pytest.mark.parametrize(
    "sql", [pytest.param(body, id=label) for label, body in _instruction_sql()]
)
def test_every_instruction_sql_block_passes_what_it_teaches(sql):
    assert SPARKSQL.check_syntax(sql) is None
    found = check_table_names([sql])
    assert (found.problems, found.unparsed) == ([], 0)


def test_the_instruction_examples_name_permanent_tables():
    # Not vacuous: the rule's own example and the worked example each name
    # permanent tables, through variables.
    checked = {
        label: check_table_names([body]).checked for label, body in _instruction_sql()
    }
    assert checked["table_names#0"] == 2
    assert max(n for label, n in checked.items() if label.startswith("examples#")) == 3
