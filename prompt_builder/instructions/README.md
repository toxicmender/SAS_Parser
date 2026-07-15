# Bundled operator instructions

Starter operator-instruction sets, organised by target output language. Load
the whole tree with:

```python
from prompt_builder.user_instructions import UserInstructionSet
ins = UserInstructionSet.from_dir("prompt_builder/instructions")
```

or via config.json:

```json
{ "user_instructions": { "dir": "prompt_builder/instructions" } }
```

## Layout convention

`from_dir` reads every `*.md` under this directory and scopes each file by
its **first path component**:

```
instructions/
  sparksql/     -> every section scoped [lang: sparksql]
    overview.md
    constructs.md
    datetime.md
    macros.md
    examples.md
  _common/      -> language-agnostic (the leading _ opts out of lang scoping)
  <root>.md     -> language-agnostic
```

Add a new target by creating a sibling directory (`pyspark/`, `snowflake/`,
…). The run's `output_language` selects the matching slice at build time, so
several languages coexist here without leaking into each other.

## Section directives

Standard `user_instructions` heading directives apply — `## [when: proc:sql]
…`, `## [topic] …`, `## [example: …] …` — and stack with the implicit
language scope. A section may override the directory's language with its own
`## [lang: …] …`. See `prompt_builder/README.md` for the full grammar.

The SparkSQL set is synthesised from the [Spark SQL
reference](https://spark.apache.org/docs/latest/sql-ref.html) and the
[PySpark SQL API docs](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/index.html);
review and adapt it to your project's conventions before relying on it.
