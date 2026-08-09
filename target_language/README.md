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
