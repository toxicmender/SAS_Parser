# Target languages

The converter supports exactly two output targets.

| Target | Accepted aliases | Notebook code | Syntax checker |
| --- | --- | --- | --- |
| PySpark | pyspark, python, python3, py, sparkpython | Python | Python AST |
| Spark SQL | sparksql, sql, databrickssql, ansisql | SQL | sqlglot Databricks dialect |

resolve_target_language returns a canonical target object or raises
UnknownTargetLanguage. Unsupported target names are rejected at the CLI,
SharePoint request, notebook, validation, complexity, and XREF boundaries.

Spark SQL parsing and post-conversion XREF rewriting always use the Databricks
dialect. sqlglot is a required dependency, so SQL validation never degrades to
structural quote or bracket checks.

The Spark SQL check masks IDENTIFIER clauses before it parses
(`mask_identifier_clauses`). Databricks takes `IDENTIFIER(<constant string>)`
wherever a name goes — the subject of CREATE, ALTER and DROP, the target of
INSERT, MERGE, UPDATE, DELETE and COPY INTO, and anything a query names — and
the generated code names its permanent tables that way, through session
variables. sqlglot reads the clause in a query but not as the target of
CREATE TABLE, INSERT INTO, COPY INTO or DROP TABLE, so each clause is replaced
by a placeholder name for the parse; `restore_identifier_clauses` puts them
back for a caller that rewrites the SQL (`xref.rewrite`).
