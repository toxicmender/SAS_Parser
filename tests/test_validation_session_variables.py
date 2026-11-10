"""Spark SQL names its permanent tables through session variables.

The Spark SQL instructions ask for every table outside WORK to be named
through Databricks SQL session variables — a ``<LIBREF>_PREFIX`` per SAS
library, a variable per table built from it, and ``IDENTIFIER(session.<VAR>)``
wherever the table is read or written — so its catalog and schema are set in
one place. ``session_variables`` is the check that generated code does.
"""

from __future__ import annotations

import logging
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
        "SELECT COUNT(*) FROM IDENTIFIER(session.B);"
    )
    assert (result.score, result.details.split(" ")[0]) == (1.0, "1/1")


@pytest.mark.parametrize(
    "keyword",
    # VAR is VARIABLE's synonym in Spark's grammar (`variable: VARIABLE | VAR`).
    ["", "VARIABLE ", "VAR "],
)
def test_a_qualified_declaration_is_read(keyword):
    # sqlglot reads `DECLARE ... session.x` only as a command; the check reads
    # it as the plain DECLARE it is once the qualifier is blanked out.
    result = _score(
        f"DECLARE OR REPLACE {keyword}session.MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"
        f"DECLARE OR REPLACE {keyword}session.T STRING DEFAULT CONCAT(session.MART_PREFIX, '.t');\n"
        "SELECT * FROM IDENTIFIER(session.T);"
    )
    assert (result.score, result.details.split(" ")[0]) == (1.0, "1/1")


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


# ── statements the rule names ───────────────────────────────────────────────

_VARS = "DECLARE OR REPLACE VARIABLE LIB_PREFIX STRING DEFAULT 'catalog_name.lib';\n" + "".join(
    f"DECLARE OR REPLACE VARIABLE {v} STRING DEFAULT CONCAT(session.LIB_PREFIX, '.{v.lower()}');\n"
    for v in ("OLD", "NEW", "V", "B")
)


@pytest.mark.parametrize(
    "sql, problems",
    [
        # Statements sqlglot reads only as commands still name tables.
        (
            "CREATE TABLE main.mart.y DEEP CLONE main.stage.x;",
            [
                "hard-coded table name `main.mart.y`: declare it as a session variable",
                "hard-coded table name `main.stage.x`: build the statement",
            ],
        ),
        ("MSCK REPAIR TABLE main.mart.t;", ["hard-coded table name `main.mart.t`"]),
        ("OPTIMIZE `main`.`mart`.`orders` FULL;", ["hard-coded table name `main.mart.orders`: build"]),
        # The places the rule says take no IDENTIFIER clause.
        (_VARS + "OPTIMIZE IDENTIFIER(session.OLD) FULL;", ["OPTIMIZE takes no IDENTIFIER"]),
        (_VARS + "VACUUM IDENTIFIER(session.OLD);", ["VACUUM takes no IDENTIFIER"]),
        (
            _VARS + "ALTER TABLE IDENTIFIER(session.OLD) RENAME TO IDENTIFIER(session.NEW);",
            ["the new name of RENAME TO takes no IDENTIFIER"],
        ),
        (
            _VARS + "ALTER TABLE IDENTIFIER(session.OLD) RENAME TO new_t;",
            ["`new_t` is written as a permanent table by a literal name: build the statement"],
        ),
        (
            _VARS + "CREATE TABLE IDENTIFIER(session.NEW) SHALLOW CLONE IDENTIFIER(session.OLD);",
            ["a CLONE source takes no IDENTIFIER"],
        ),
        (
            _VARS + "CREATE TABLE IDENTIFIER(session.NEW) DEEP CLONE IDENTIFIER(session.OLD);",
            ["a CLONE source takes no IDENTIFIER"],
        ),
        # A permanent table named like a WORK dataset is still permanent.
        (
            "CREATE OR REPLACE TEMP VIEW orders AS SELECT 1 AS a;\n"
            "CREATE OR REPLACE TABLE orders AS SELECT * FROM orders;",
            ["`orders` is written as a permanent table by a literal name"],
        ),
        # Bodies Databricks persists may not read a session variable.
        (
            _VARS + "ALTER VIEW IDENTIFIER(session.V) AS SELECT * FROM IDENTIFIER(session.B);",
            ["a stored view's query reads IDENTIFIER(session.B)"],
        ),
        (
            _VARS + "CREATE OR REPLACE FUNCTION f() RETURNS TABLE (a INT) "
            "RETURN SELECT * FROM IDENTIFIER(session.B);",
            ["a SQL function's body reads IDENTIFIER(session.B)"],
        ),
        # ... so a name they hard-code is built with EXECUTE IMMEDIATE, not
        # IDENTIFIER, which the next attempt would fail on.
        (
            "CREATE OR REPLACE FUNCTION f() RETURNS TABLE (a INT) RETURN SELECT * FROM main.mart.t;",
            ["hard-coded table name `main.mart.t`: build the statement"],
        ),
        (
            _VARS + "CREATE OR REPLACE VIEW IDENTIFIER(session.V) AS SELECT * FROM main.mart.src;",
            ["hard-coded table name `main.mart.src`: build the statement"],
        ),
        # The tables a variable's value reads.
        (
            "DECLARE OR REPLACE VARIABLE MAX_DT DATE;\n"
            "SET VAR MAX_DT = (SELECT MAX(dt) FROM main.mart.t);",
            ["hard-coded table name `main.mart.t`"],
        ),
        (
            "DECLARE OR REPLACE VARIABLE N INT DEFAULT (SELECT COUNT(*) FROM main.mart.t);",
            ["hard-coded table name `main.mart.t`"],
        ),
        (
            "DECLARE OR REPLACE VARIABLE N INT;\n"
            "SET VAR N = (SELECT COUNT(*) FROM IDENTIFIER(session.T));",
            ["`session.T` is not declared"],
        ),
        # SET without VAR on a variable is rejected by Spark, so it gives none.
        (
            _VARS + "DECLARE OR REPLACE VARIABLE T STRING;\n"
            "SET T = CONCAT(session.LIB_PREFIX, '.t');\n"
            "SELECT * FROM IDENTIFIER(session.T);",
            ["`SET T = ...` without VAR", "`T` is never given a value"],
        ),
        # A write inside a scripting block.
        (
            "DECLARE OR REPLACE VARIABLE i INT DEFAULT 0;\n"
            "WHILE i < 3 DO\n  INSERT INTO perm_t SELECT 1;\n  SET VAR i = i + 1;\nEND WHILE;",
            ["`perm_t` is written as a permanent table by a literal name"],
        ),
        # A permanent view named like a WORK dataset is still a stored view.
        (
            _VARS + "CREATE OR REPLACE TEMP VIEW v AS SELECT 1 AS a;\n"
            "CREATE OR REPLACE VIEW v AS SELECT * FROM IDENTIFIER(session.B);",
            [
                "`v` is written as a permanent table by a literal name",
                "a stored view's query reads IDENTIFIER(session.B)",
            ],
        ),
        # A WORK view's variable given a permanent table's name later.
        (
            "DECLARE OR REPLACE VARIABLE OUT_VIEW STRING DEFAULT 'region_out';\n"
            "CREATE OR REPLACE TEMP VIEW IDENTIFIER(session.OUT_VIEW) AS SELECT 1 AS a;\n"
            "SET VAR OUT_VIEW = 'region_final';\n"
            "INSERT INTO IDENTIFIER(session.OUT_VIEW) SELECT * FROM region_out;",
            ["`OUT_VIEW` holds a literal name"],
        ),
    ],
)
def test_every_statement_the_rule_names_is_judged(sql, problems):
    found = check_table_names([sql])
    assert len(found.problems) == len(problems)
    for problem, expected in zip(found.problems, problems):
        assert problem.startswith(expected)
    assert found.checked >= len(problems)


@pytest.mark.parametrize(
    "sql",
    [
        # A temporary view's or function's body may read a session variable.
        _VARS + "CREATE OR REPLACE TEMP VIEW t AS SELECT * FROM IDENTIFIER(session.B);\n"
        "ALTER VIEW t AS SELECT * FROM IDENTIFIER(session.V);",
        _VARS + "CREATE TEMPORARY FUNCTION f() RETURNS TABLE (a INT) "
        "RETURN SELECT * FROM IDENTIFIER(session.B);",
        # Dropping a WORK dataset is no permanent write.
        "CREATE OR REPLACE TEMP VIEW orders AS SELECT 1 AS a;\nDROP TABLE IF EXISTS orders;",
        # A function's name is no table, and dropping one reads none.
        "DROP FUNCTION IF EXISTS main.mart.f;",
        # A path read in place is a file.
        "VACUUM delta.`/Volumes/c/s/v/t`;",
        # The forms the rule asks for.
        _VARS + "ALTER TABLE IDENTIFIER(session.OLD) CLUSTER BY (a);\n"
        "DECLARE OR REPLACE VARIABLE STMT STRING;\n"
        "SET VAR STMT = CONCAT('OPTIMIZE ', session.OLD, ' FULL');\n"
        "EXECUTE IMMEDIATE STMT;",
        # A variable read from a table through its own variable.
        _VARS + "DECLARE OR REPLACE VARIABLE N INT;\n"
        "SET VAR N = (SELECT COUNT(*) FROM IDENTIFIER(session.OLD));",
        # A WORK dataset made inside a scripting block, and a MERGE's actions.
        "DECLARE OR REPLACE VARIABLE i INT DEFAULT 0;\n"
        "WHILE i < 3 DO\n  CREATE OR REPLACE TEMP VIEW step AS SELECT i AS n;\n"
        "  SET VAR i = i + 1;\nEND WHILE;",
        _VARS + "MERGE INTO IDENTIFIER(session.OLD) AS t USING IDENTIFIER(session.NEW) AS s "
        "ON t.k = s.k WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *;",
        # A WORK dataset named through a variable, as macros.md resolves
        # `work.&prefix._out`: a temporary view, not a permanent table.
        "DECLARE OR REPLACE VARIABLE OUT_VIEW STRING DEFAULT 'region_out';\n"
        "CREATE OR REPLACE TEMP VIEW IDENTIFIER(session.OUT_VIEW) AS SELECT 1 AS a;\n"
        "SELECT * FROM IDENTIFIER(session.OUT_VIEW);\n"
        "ALTER VIEW IDENTIFIER(session.OUT_VIEW) AS SELECT 2 AS a;\n"
        "DROP VIEW IF EXISTS IDENTIFIER(session.OUT_VIEW);",
        # Spark's global temporary views.
        "SELECT * FROM global_temp.v;",
        # A WORK view named through more than one variable, and a Spark
        # setting: no permanent table either.
        "DECLARE OR REPLACE VARIABLE REGION STRING DEFAULT 'east';\n"
        "CREATE OR REPLACE TEMP VIEW IDENTIFIER(session.REGION || '_out') AS SELECT 1 AS a;",
        "SET ansi_mode = true;",
    ],
)
def test_what_the_rule_allows_passes(sql):
    assert check_table_names([sql]).problems == []


# ── edges of the reading ─────────────────────────────────────────────────────

_PREFIX = "DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';\n"


@pytest.mark.parametrize(
    "clause",
    [
        "IDENTIFIER(CONCAT(session.MART_PREFIX, '.orders'))",
        "IDENTIFIER(session.MART_PREFIX || '.orders')",
    ],
)
def test_a_name_built_inline_from_a_declared_prefix_passes(clause):
    found = check_table_names([_PREFIX + f"SELECT * FROM {clause};"])
    assert (found.checked, found.problems) == (1, [])


@pytest.mark.parametrize(
    "sql, problem",
    [
        # A column, not a session variable: the name is spelled out.
        (
            "SELECT * FROM IDENTIFIER(mart.orders);",
            "IDENTIFIER(mart.orders) hard-codes the name",
        ),
        (
            _PREFIX + "SELECT * FROM IDENTIFIER(session.MART_PREFIX ||);",
            "IDENTIFIER(session.MART_PREFIX ||) could not be read",
        ),
        # `DECLARE ... session.X` is a command to sqlglot, read by hand.
        (
            "DECLARE OR REPLACE VARIABLE session.ORDERS STRING;\n"
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`ORDERS` is never given a value",
        ),
        (
            "DECLARE OR REPLACE VARIABLE session.ORDERS STRING DEFAULT CONCAT(session.P, ;\n"
            "SELECT * FROM IDENTIFIER(session.ORDERS);",
            "`ORDERS`'s DEFAULT could not be read",
        ),
        # Neither a Spark setting nor a SET VAR on nothing declares a variable.
        (
            "SET spark.sql.shuffle.partitions = 8;\nSELECT * FROM IDENTIFIER(session.ORDERS);",
            "`session.ORDERS` is not declared",
        ),
        (
            "SET VAR ORDERS = 'c.mart.orders';\nSELECT * FROM IDENTIFIER(session.ORDERS);",
            "`session.ORDERS` is not declared",
        ),
    ],
)
def test_what_the_check_cannot_accept_is_named(sql, problem):
    found = check_table_names([sql])
    assert found.checked == 1
    assert len(found.problems) == 1 and found.problems[0].startswith(problem)


@pytest.mark.parametrize(
    "sql",
    [
        "DECLARE (x INT);",  # no name the check can read
        "OPTIMIZE .;",
        ";",
        "-- a comment and nothing else",
    ],
)
def test_what_names_no_table_is_left_alone(sql):
    assert check_table_names([sql]) == check_table_names([])


def test_problems_are_logged_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="validation.table_names")
    check_table_names(["SELECT * FROM mart.orders;"])
    assert "check_table_names: ['hard-coded table name `mart.orders`" in caplog.text


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


# ── end to end ───────────────────────────────────────────────────────────────

_CASE = pathlib.Path(__file__).resolve().parents[1] / "validation/cases/permanent_library.json"

_HARD_CODED = """\
CREATE OR REPLACE TABLE main.mart.orders_clean AS
SELECT o.*, amount - cost AS margin FROM main.stage.orders o WHERE amount > 0;

CREATE OR REPLACE TEMP VIEW region_margin AS
SELECT region, SUM(margin) AS total_margin FROM main.mart.orders_clean GROUP BY region;

INSERT INTO main.mart.margin_history SELECT * FROM region_margin;
"""

_THROUGH_VARIABLES = """\
DECLARE OR REPLACE VARIABLE STAGE_PREFIX STRING DEFAULT 'catalog_name.stage';
DECLARE OR REPLACE VARIABLE MART_PREFIX STRING DEFAULT 'catalog_name.mart';
DECLARE OR REPLACE VARIABLE ORDERS STRING DEFAULT CONCAT(session.STAGE_PREFIX, '.orders');
DECLARE OR REPLACE VARIABLE ORDERS_CLEAN STRING DEFAULT CONCAT(session.MART_PREFIX, '.orders_clean');
DECLARE OR REPLACE VARIABLE MARGIN_HISTORY STRING DEFAULT CONCAT(session.MART_PREFIX, '.margin_history');

CREATE OR REPLACE TABLE IDENTIFIER(session.ORDERS_CLEAN) AS
SELECT o.*, amount - cost AS margin FROM IDENTIFIER(session.ORDERS) o WHERE amount > 0;

CREATE OR REPLACE TEMP VIEW region_margin AS
SELECT region, SUM(margin) AS total_margin
FROM IDENTIFIER(session.ORDERS_CLEAN) GROUP BY region;

INSERT INTO IDENTIFIER(session.MARGIN_HISTORY) SELECT * FROM region_margin;
"""

_RULE = "Table names through session variables"


class _Scripted:
    """Chat-model stub: gives its answers in order and records each prompt."""

    def __init__(self, *answers: str) -> None:
        self.answers = answers
        self.prompts: list[list] = []

    def invoke(self, messages, config=None):
        from langchain_core.messages import AIMessage

        self.prompts.append(list(messages))
        return AIMessage(self.answers[len(self.prompts) - 1])


def _spark_sql_pipeline(llm: _Scripted, *, retries: int = 0):
    from llm_client import LLMClientConfig
    from pipeline import SasLLMPipeline
    from pipeline.setup import PromptingSetup, ValidationSetup
    from prompt_builder.user_instructions import UserInstructionSet
    from validation import LiveValidator

    return SasLLMPipeline(
        llm_config=LLMClientConfig(model="unused-because-llm-injected"),
        output_language="Spark SQL",
        prompting=PromptingSetup(
            structured_output=False,
            user_instructions=UserInstructionSet.from_dir(
                str(_SPARKSQL_INSTRUCTIONS.parent)
            ),
        ),
        validation=ValidationSetup(
            validator=LiveValidator(output_language="Spark SQL"), retries=retries
        ),
        llm=llm,
    )


def _text(prompt: list) -> str:
    return "\n".join(str(message.content) for message in prompt)


def test_a_hard_coded_answer_is_retried_with_the_tables_named():
    from langchain_core.messages import SystemMessage

    from validation.dataset import load_cases

    case = load_cases(_CASE)[0]
    llm = _Scripted(_response(_HARD_CODED), _response(_THROUGH_VARIABLES))
    results = _spark_sql_pipeline(llm, retries=1).run_text(
        case.sas_source, source_id=f"{case.case_id}.sas"
    )

    # The item names permanent tables, so it was shown the rule and the
    # worked example.
    assert _RULE in _text(llm.prompts[0])
    assert "Permanent library tables through session variables" in _text(llm.prompts[0])
    # The first answer failed, and the retry was told which tables to fix.
    assert len(llm.prompts) == 2
    notes = [
        str(m.content)
        for m in llm.prompts[1]
        if isinstance(m, SystemMessage) and "**session_variables**" in str(m.content)
    ]
    assert len(notes) == 1
    for table in ("main.stage.orders", "main.mart.orders_clean", "main.mart.margin_history"):
        assert f"hard-coded table name `{table}`" in notes[0]
    # The answer kept is the one that names every table through a variable.
    (result,) = results
    verdict = {m["metric"]: m for m in result["validation"]["metrics"]}
    assert result["validation"]["passed"] is True
    assert verdict["session_variables"]["details"].startswith(
        "4/4 permanent table reference(s) go through declared session variables"
    )
    assert verdict["target_syntax"]["score"] == 1.0
    assert verdict["dataset_fidelity"]["score"] == 1.0


def test_a_work_only_item_is_not_shown_the_rule():
    llm = _Scripted(
        _response("CREATE OR REPLACE TEMP VIEW b AS SELECT * FROM a WHERE x > 0;")
    )
    (result,) = _spark_sql_pipeline(llm).run_text(
        "data work.b;\n  set work.a;\n  where x > 0;\nrun;\n", source_id="work_only.sas"
    )
    assert _RULE not in _text(llm.prompts[0])
    verdict = {m["metric"]: m for m in result["validation"]["metrics"]}
    assert verdict["session_variables"]["skipped"] is True
