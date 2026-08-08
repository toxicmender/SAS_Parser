## [when: statement:label] [kind: DATA_STEP, PROC_STEP] LABEL becomes a comment
A SAS `LABEL` is documentation the source already states, so it survives the
translation rather than being dropped:

- A variable label -> a **column** `COMMENT`, inline in the DDL
  (`amt DOUBLE COMMENT 'Transaction amount'`) or via
  `ALTER TABLE t ALTER COLUMN amt COMMENT '...'`.
- A dataset label (`data work.x (label='...')`, or `PROC DATASETS ... MODIFY`)
  -> a **table** `COMMENT`, inline (`CREATE TABLE t (...) COMMENT '...'`) or
  `COMMENT ON TABLE t IS '...'`.

A label only reaches a table, so a step whose output is a `TEMP VIEW` has
nowhere to put one — carry it as a `--` comment in the SQL instead, and say so.

⚠️ **Carry over, do not invent.** Emit a comment only where the SAS states a
label. Do not add descriptive comments, tags, or `TBLPROPERTIES` the source
does not have: inferred documentation reads as fact and nobody reviews it. This
is the general "add nothing the SAS did not ask for" rule applied to metadata.

## [when: statement:format, statement:length] FORMAT and LENGTH are not types
⚠️ A SAS `FORMAT` is a **display** rule, not storage: `format amt dollar12.2;`
changes how a value prints and nothing about the value. Databricks has no
column-level display format, so a `FORMAT` statement usually translates to
*nothing* — do not turn it into a `CAST`, a `format_number`, or a rounded
column, which changes the data. The exceptions are where the SAS reads the
formatted text back (`PUT(x, fmt.)`, covered in the format guidance) or where a
user-defined format is a value mapping.

A `LENGTH` statement sets storage width. `STRING` is not width-bound, so
`length name $20;` also translates to nothing — but note it under Risks when a
downstream comparison or hash depended on the padded width.
