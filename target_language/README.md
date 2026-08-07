# target_language

What the run translates *into*, as one resolved object instead of a string
four layers each interpret on their own.

Dependency-free (stdlib only) and imported by `pipeline`, `prompt_builder`,
`validation`, and `complexity` — which is why it is its own package rather
than living inside any of them.

## Why

`output_language` used to be free-form text passed separately to whatever
happened to need it, and every consumer guessed differently:

- the system prompt named it and nothing checked the answer came back in it;
- `pipeline.notebook` mapped it to a kernel, falling back to python3 for
  anything it did not recognise;
- `validation`'s syntax metric ignored it and checked **Python**, so a correct
  Spark SQL run scored `0.0` with *"no fenced Python code blocks"* — and with
  `validation_retries` on, drove retries that could never pass;
- the two entry points disagreed on the default (`SasLLMPipeline` said
  PySpark, the CLI said SparkSQL), and `complexity.target` was a third,
  uncoordinated knob.

Resolving once and passing the object down removes the class of bug.

## Use

```python
from target_language import resolve_target_language

target = resolve_target_language("spark sql")
target.display_name        # "Spark SQL" — the only spelling that reaches a prompt
target.default_fence       # "sql"       — the tag translated code must carry
target.cell_language       # "sql"       — notebook cell / schema `language`
target.comment_prefix      # "--"        — line-comment token for the prompt
target.owns_fence("sql")   # True; owns_fence("python") is False
target.check_syntax(sql)   # None when it parses, else an error message
```

`comment_prefix` exists so the system prompt can ask for a
`-- NOT CONVERTIBLE TO Spark SQL: <reason>` marker without hard-coding one
target's comment syntax — the templates interpolate `{comment_prefix}` and
the same sentence renders as `#` for PySpark and `//` for Spark Scala.

Normally you do not call this directly: `SasLLMPipeline` resolves at
construction and exposes the result as `.target_language`, and everything
downstream should take it from there.

## Targets

| Target | Folds from | Fences | Kernel | Comment | Complexity profile |
|---|---|---|---|---|---|
| PySpark | `pyspark`, `python`, `python3`, `py` | `python`, `py`, `pyspark` | python3 | `#` | `pyspark` |
| Spark SQL | `sparksql`, `spark sql`, `sql`, `databrickssql` | `sql`, `sparksql` | sql | `--` | `sparksql` |
| Spark Scala | `sparkscala`, `scala` | `scala` | scala | `//` | `pyspark` |

Spelling is folded case-, space-, hyphen-, and underscore-insensitively — the
same rule `prompt_builder` matches `[lang: ...]` directive tokens with, which
is why that rule lives here and is re-exported there rather than duplicated.

An unrecognised name raises `UnknownTargetLanguage` listing the known targets.
`allow_unknown=True` keeps the old lenient behaviour — the name still reaches
the prompt, everything structural borrows PySpark's — and warns. `pipeline`
rejects; `pipeline.notebook` allows, because by the time a notebook is being
written the translation has already been paid for and losing it is worse than
writing it under the wrong kernel.

## Syntax checking

`check_syntax` returns an error message or `None`. Python uses `ast.parse`.
Spark SQL uses **sqlglot** (`pip install 'sas-parser[sql]'`) and, without it,
falls back to a structural check that flags only unbalanced brackets and
quotes — deliberately conservative, because a false failure here spends an
LLM call re-answering a correct item. `checker_name` reports which ran, and
the metric puts it in its `details`. Spark Scala has no checker, so
`checks_syntax` is `False` and the metric skips rather than passing everything.

Note that syntax and language are separate questions: sqlglot happily parses
plenty of non-SQL, so `python_syntax` alone will not catch a Python answer to
a Spark SQL prompt. `validation`'s `language_compliance` is what does — the
two metrics are complementary by design.
