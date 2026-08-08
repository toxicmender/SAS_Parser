## [when: proc:format] [kind: PROC_STEP] PROC FORMAT
`PROC FORMAT` defines named value mappings; it produces no data of its own.
Translate each `VALUE` / `INVALUE` block into a lookup the later queries can
join or a `CASE` they can inline — a small mapping is clearer inlined, a large
or reused one belongs in a view:

```sql
CREATE OR REPLACE TEMP VIEW fmt_region AS
SELECT * FROM VALUES ('N', 'North'), ('S', 'South') AS t(code, label);
```

Keep the `OTHER=` branch as the `ELSE` (or as the `COALESCE` default on a left
join). A range-based format (`low-<100 = 'small'`) becomes a `CASE` with the
same boundaries — ⚠️ mind SAS's exclusive `<` markers, which sit on a
different side of the boundary than a naive `BETWEEN`.

⚠️ A `PROC FORMAT` with `CNTLIN=`/`CNTLOUT=` builds the format from a dataset;
translate that dataset into the lookup view directly.

## [when: function:put, function:putc, function:putn] [kind: DATA_STEP, PROC_STEP] PUT with a user-defined format
⚠️ **`PUT(x, myfmt.)` is not a cast.** Where the format is user-defined
(anything declared by a `PROC FORMAT` in this codebase, conventionally
`$name.` or `name.`), `PUT` *applies the mapping* — translating it to
`CAST(x AS STRING)` silently replaces labels with raw values, and every
downstream comparison against a label then fails to match.

Translate it as the mapping it is: inline the `CASE`, or left-join the format
view and select its label. `INPUT(s, myinfmt.)` is the same in reverse — a
reverse lookup, not a numeric parse.

Only a *built-in* format is a formatting call: `PUT(n, best12.)` ->
`CAST(n AS STRING)`, `PUT(n, comma10.2)` -> `format_number(n, 2)`,
`PUT(d, yymmdd10.)` -> `DATE_FORMAT(d, 'yyyy-MM-dd')`. If you cannot tell
whether a format is built-in or user-defined, say so under Risks rather than
guessing — the two translations produce different data.
