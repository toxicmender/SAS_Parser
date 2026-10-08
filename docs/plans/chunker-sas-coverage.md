# Plan: close the chunker's SAS coverage gaps (breaking changes allowed)

Status: Phases 0–5 implemented. `tests/test_sas_coverage.py` holds 183 probes:
the 168 from the coverage review plus 15 precision probes (`X01`–`X15`). Today
178 pass and 5 are expected failures, each naming the phase below that fixes
it. Phases 6–9 are not started.

## Context

The coverage review ran 168 SAS probes: 112 pass (67%); 56 gaps fall into 10
root causes. Re-reading the module for this plan found a second class of
problem the probes under-sampled: **false reads and writes.**

- **Commented-out code is treated as live.** `* set lib.old;` inside a DATA
  step makes the step read `lib.old`. `* out=c;` inside a PROC makes it write
  `work.c`.
- **Comments in macro bodies are treated as live.** `%* old: data lib.tmp
  set lib.x;` is read as a DATA statement writing three datasets.
- **`%put … from stage;`** in a macro body becomes a body read of `work.stage`.
- **`referenced_datasets` is noisy.** It picks up `%put` text and variables
  named `out`, `data` or `from`.

These trace to four structural causes:

1. **No statement context.** `_metadata_for` (`chunker/metadata.py`) runs ~40
   regexes over a region's whole text. Comments, datalines, foreign code,
   `%put` text and variable names are all scanned as code. Operand lists are
   cut by lookahead regexes, which lose the whole input on `end=`, `point=`
   or `where=(…)`.
2. **Two copies of every pattern.** `_io_for` (steps) and `_macro_body_io`
   (macro bodies) each hold their own set (`_SET_RE`, `_BODY_SET_RE`, …), so
   every fix has to be made twice and they have drifted.
3. **PROCs know three options.** Only `data=`, `out=` and `outdata=` are
   recognised, with no per-PROC meaning, so `base=` and `cntlin=` are missed
   and `COPY out=tgt` reads as a dataset.
4. **The scanner's only terminator is `;`, and every DATA/PROC keyword at
   statement start opens a step.** This holds even inside datalines, SUBMIT
   blocks, DS2 programs and run-group PROCs, and `%str(;)` splits statements.

**Outcome:**

- The scanner knows what each statement is.
- Dataset references are extracted per statement into typed records, with one
  extractor shared by steps and macro bodies.
- Each PROC's options have defined meanings.
- One SQL grammar serves PROC SQL, FedSQL and pass-through.

**Targets:**

- Every probe in `tests/test_sas_coverage.py` passes: the 168 from the review
  and the 15 precision probes.
- Full `chunk_text` is no more than 20% slower on the 900 KB benchmark files
  (5.4–5.8 s at Phase 0).

## Breaking changes

| Change | Affected | Migration |
|---|---|---|
| `SasChunkMetadata.dataset_refs: tuple[SasDatasetRef, ...]` becomes the stored source; `input_datasets`, `output_datasets`, `referenced_datasets`, `referenced_librefs`, `body_literal_*`, `body_param_*` become read-only computed views | ~10 test constructors; every `model_copy(update={"input_datasets": …})` site | JSON keys unchanged (computed fields serialise). A `model_validator(mode="before")` turns legacy lists into refs, so old JSON still loads. A guard test fails on any `update=` that names a view |
| Update-in-place (SQL INSERT/UPDATE/DELETE/ALTER, `APPEND base=`, `MODIFY`) is role `UPDATE`, so the table appears in both input and output lists | Batching: the modifying step now depends on the table's earlier producer | Intended; batch diffs are reviewed in verification |
| `referenced_datasets`/`referenced_librefs` come only from real dataset positions | Fewer, correct entries | — |
| Chunk boundaries | Run-group PROCs (DATASETS, REG, …) keep statements after `run;` until `quit;`. DS2 programs stay inside PROC DS2. Datalines/SUBMIT text never opens a step. `%*` becomes COMMENT_BLOCK. `%symdel`, `%syslput`, … change MACRO_CALL → GLOBAL_STATEMENT. New global statements are recognised | Snapshot-style tests updated |
| `_Unit.is_comment` → `_Unit.role` (`UnitRole`) | `chunker/` internals only (12 uses) | — |
| `_io_for`/`_macro_body_io` replaced by `chunker/statements.py` | 14 direct test calls | Tests move to the new API |
| `PathLocation.FILEREF`; `%include` records every file; `PathSpec` can yield several values per statement | `paths.py`, `xref/pre.py` | `xref.pre` iterates the values |

## Phase 0: regression net first (S) — done

- `tests/test_sas_coverage.py` holds the probe table and the harness, with one
  parametrised case per probe.
- A probe failing today carries `gap=<reason>`, naming the phase that fixes
  it, and runs as `xfail(strict=True)`. Each phase removes the gaps it closes.
  With `strict=True`, an accidental pass or a regression fails the run.
- The 15 precision probes `X01`–`X15` assert that names never appear:
  - commented-out `set`/`out=`/SQL lines in a DATA step, a PROC and PROC SQL;
  - `*` and `%*` comments inside a macro body;
  - `%put … from x` in open code and in a macro body; a `%let` value;
  - DATA step `out = x*2;` and `if data = y`; PROC `var from to;`;
  - a PHREG programming statement `out = x + 1;`;
  - datalines holding `set lib.z`; Python SUBMIT code `out = df.merge(x)`;
  - SQL inside a string literal (passes today; kept as a guard).
- `scripts/bench_chunker.py` times full chunking, best of 3.
  - Corpora: two 900 KB jobs (macro calls with and without semicolons), and
    5,000 back-to-back calls.
  - At Phase 0: 5.75 s, 5.42 s and 0.52 s.
  - Machine drift between sessions is about 15%, so the gate compares the two
    commits in the same session.

## Phase 1: scanner, statement roles and block structure (M) — done

Done as planned, with these differences:

- **Masking came forward from Phase 3.** Metadata reads `_Region.code_text`,
  the region with every non-CODE unit blanked, so comments count nowhere
  either. That flipped X01–X05 as well as X13 and X14. Phase 3 still moves
  `_metadata_for` onto units. One consequence: SAS runs a macro call written
  inside a `*` comment, but the chunker no longer records it.
- **Unclosed macro quoting stays linear.** `_macro_quote_end` records the
  match of every parenthesis it walks past, so a `%str(` that never closes is
  walked once. 20,000 such lines take 2.3 s, against 67 s for a walk per
  function and 2.0 s at Phase 0.
- **Small additions:**
  - the classifier knows `cas`, `caslib`, `resetline`, `sysecho` and `%inc`;
  - `RUN_GROUP_PROCS` adds catmod, document, gslide, gradar and gbarline;
  - a DATA or PROC keyword used as a variable (`data = 1;`) opens no step;
  - `global_statement_keyword` handles numbered statements, so `title2` is
    `title` (it was `None`).
- **Performance.** Full `chunk_text` on the 900 KB files is about 3% slower
  (A/B against Phase 0, same session). The event regex is now an alternation
  of literals, which the engine scans 2.5× faster than the old character
  class.
- **Batch diff.** On a 187-file corpus (every probe, both reference examples,
  two multi-step jobs), batches went from 25 to 23 and singletons from 142 to
  131. Every change follows from a chunk fix:
  - global statements (`%let`, `signon`, `rsubmit`) join the step they
    precede;
  - run-group PROCs and DS2/Python blocks are whole chunks instead of
    fragments;
  - a DATA step that the false reads in its datalines held in a batch now
    stands alone.
- **Reference examples.** Both give the same complexity reports and hydration
  plans as at Phase 0.

Files: `chunker/scanner.py`, `chunker/chunker.py` (`_scan_units`,
`_group_regions`, `_collect_block`), `chunker/keywords.py`.

1. **Unit roles.** `UnitRole` {CODE, COMMENT, DATALINES, FOREIGN} replaces
   `is_comment`. `%* …;` is a COMMENT.
2. **Macro quoting.** `_SCAN_EVENT_RE` adds `%(?:nr)?b?(?:str|quote)\s*\(`.
   Its body is skipped with paren depth, `%x` escapes and quotes, so a `;`
   inside is not a terminator. `_sanitise` also blanks those semicolons in
   `mt` (`cf` keeps them).
3. **Datalines.** After `datalines|cards|lines|parmcards` (with or without 4),
   the data up to the terminator line becomes one DATALINES unit. The
   terminator is the first line holding `;`, or a `;;;;` line for the 4-forms.
4. **SUBMIT blocks.** After `submit …;`, everything up to the `endsubmit;`
   line is one FOREIGN unit, with no quote tracking inside. This covers PROC
   PYTHON, LUA, GROOVY, IML and FCMP.
5. **RUN CANCEL.** `run cancel;` closes a step. The `run`/`quit` check in
   `_collect_block` becomes a regex.
6. **Run-group PROCs.** `keywords.RUN_GROUP_PROCS` lists sql, fedsql, ds2,
   datasets, catalog, plot, reg, glm, anova, arima, iml, optmodel, cas,
   casutil, model, gplot, gchart, gmap, g3d, gcontour, greplay, pmenu and
   trantab. For these:
   - `run;` does not close the PROC; `quit;` does;
   - a following step header closes it without UNCLOSED_* once a RUN was seen;
   - `_collect_block` takes the PROC name from the opening statement.
7. **PROC DS2.** Nested `data|thread|package` headers do not close it.
8. **Non-code units.** A unit that isn't CODE never opens or closes a block
   and is never classified. This also applies to `_split_after_calls`.
9. **Classifier.** New `_CLS_GLOBAL_MISC_RE` maps these to GLOBAL_STATEMENT:
   endsas, dm, sasfile, lock, goptions, axisN, symbolN, legendN, patternN,
   missing, page, skip, catname, signon, signoff, rsubmit, endrsubmit,
   rdisplay, rget, listtask and killtask.
   - Macro statements (`%symdel`, `%syslput`, `%sysrput`, `%sysmacdelete`,
     `%sysmstoreclear`, `%window`, `%display`, `%input`, `%keydef`, `%copy`)
     also become GLOBAL_STATEMENT.
   - `_CLS_MACROCALL_RE` excludes `_MACRO_LANGUAGE_WORDS`.
   - The new words are added to `SAS_GLOBAL_STATEMENT_TOKENS`.

Probes flipped: L03, L08, L11, L15, L16, M16, G11, G12, G15, G16, G17, A06,
E04, X01–X05, X13 and X14, plus the structural half of E01 (its I/O is
Phase 6).

## Phase 2: dataset reference model, behaviour-preserving (M) — done

Done as planned, with these differences:

- **The record.** `SasDatasetRef` is a frozen, slotted dataclass, not a
  pydantic model, and `dataset_refs` is a tuple. Pydantic still validates and
  serialises it as a field. A model cost three tracked objects per record and
  showed up as collector time. The tuple can only be replaced, never edited,
  so the views are computed once per tuple and cached outside the fields.
- **`referenced_datasets` and `referenced_librefs` stay stored until Phase
  3.** They hold the spellings a separate raw scan finds (and `%LET` values
  that look like datasets), which no reference records yet. Phase 3, which
  removes that scan, makes them views; a `%LET` value then needs a
  reference that neither reads nor writes.
- **Rewrites are methods.** `SasChunkMetadata.map_dataset_names(rename)`
  renames or drops references, and `add_dataset_refs` appends them.
  `model_copy` raises on an `update` that names a view, rather than silently
  dropping it. No sort key is needed: merging is a source-order union.
- **Constructors.** A `before` validator turns the legacy lists into
  references, so old JSON and every test constructor work unchanged. The five
  sites pyright checks were moved to `dataset_refs` anyway.
- **Behaviour.**
  - Chunks, metadata and batches are identical to Phase 1 on the 187-file
    corpus and a 4,800-chunk job, also with forced splits.
  - The one change is planned: a split child keeps its region's dataset
    order instead of an alphabetical one, so its last output is again
    `_LAST_`.
- **Performance.**
  - Neutral, after two fixes: `_libname_tables` returns at once when no
    database LIBNAME is in force, and `_resolved_meta` reads names from the
    references instead of the views.
  - Chunking the 900 KB files takes what it did at Phase 0, about 3% faster
    than Phase 1. Batching is level with Phase 1.

Files: `chunker/models.py`, `chunker/metadata.py`, `chunker/batcher.py`.

- **`DatasetRole`:** READ, WRITE, UPDATE (read and rewritten in place), DROP.
- **`SasDatasetRef`** (frozen, sortable by `_dataset_ref_sort_key`), with
  fields:
  - `name` — canonical form (`work.x`, `lib.x`, `'/path'`, `&ref`, or the
    pattern `lib.pre:`);
  - `raw`, `role`, `via` (`set`, `merge`, `data`, `output`, `hash`, `data=`,
    `out=`, `base=`, `from`, `join`, `create`, `insert`, `ods_output`,
    `macro_call`, …);
  - `in_macro_body`, `param`, `param_pos`, `pattern`.
- **Computed views**, all in first-seen order:

  | View | Built from |
  |---|---|
  | `input_datasets` | READ and UPDATE refs, excluding macro-body refs |
  | `output_datasets` | WRITE and UPDATE refs, excluding macro-body refs |
  | `body_literal_inputs`, `body_literal_outputs` | macro-body refs that are not parameters |
  | `body_param_inputs`, `body_param_outputs` | `{"param", "pos"}` from parameter refs |
  | `referenced_datasets` | raw and canonical names |
  | `referenced_librefs` | the libref of each name, plus `defines_librefs` |
  | `dropped_datasets` (new) | DROP refs |

  `step_name` comes from the DATA header ref.
- **One rewrite helper.** `map_dataset_names(meta, fn)` replaces each list-level
  rewrite:
  - `_resolved_meta` and `_CANONICAL_DS_FIELDS`;
  - batcher `_resolve_implicit_datasets` (`_last_`, which keeps relying on
    output order) and `replace_dataset_names`;
  - `_resolve_macro_body`, which appends refs with `via="macro_call"`.
- **Merging.** `_merge_meta` gets a `list[SasDatasetRef]` branch (union in
  source order).
- **Plumbing only.** In this phase refs are still built from the current
  `_io_for`/`_macro_body_io` output. The suite stays green apart from
  constructor sites, which proves the plumbing before extraction changes.

## Phase 3: per-statement extraction (L), new `chunker/statements.py` — done

Done as planned, with these differences:

- **No separate tokenizer.** `_operands` reads an operand list straight off the
  sanitised statement: names, quoted paths and name literals, `(…)` option
  groups skipped whole, macro calls skipped, numbered ranges (zero padding
  kept) and prefix lists. It stops at a `name=` option or a `/`. `Statement`
  carries `mt`, `cf`, `keyword`, `context`, `proc` and `in_macro`. The
  statement core strips `%IF … %THEN`, `%ELSE`, labels, and in a DATA step
  `IF … THEN`, `ELSE`, `WHEN (…)` and `OTHERWISE`.
- **`_metadata_for(region)`** takes the region and reads its units. Only
  DATA_STEP, PROC_STEP and MACRO_DEFINITION regions are read for datasets, as
  before; the other kinds hold no statement that reads or writes.
- **Hash `.output(dataset:)` writes.** PROC SQL keeps its four regexes, now
  per statement, until Phase 5. A statement shaped like an assignment is
  never read for PROC options, which also flipped X12, planned for Phase 4.
- **`referenced_datasets` holds canonical names only,** one per reference of
  any role, a parameter as `&param`. The raw spellings live in
  `SasDatasetRef.raw`. `referenced_librefs` is their librefs plus
  `defines_librefs`. `unresolved_dataset_refs` is every name with a `&`.
- **`DatasetRole.MENTION`** is the role for a `%LET` value written like a
  dataset, with `via="%let"`. `_resolved_meta` derives these again on every
  run, so the corpus-level run replaces the file-level answer. Legacy JSON's
  `referenced_datasets` loads as MENTION references.
- **Macro bodies hold step fragments.** A body statement outside any step it
  opens may complete its caller's step (`%macro sets; set a b; %mend;`), so
  it is read by its keyword: a DATA step statement, a PROC SQL clause, or PROC
  options. The old whole-text scan caught these, and losing them would have
  cost batching edges. For the same reason, a call of another macro inside a
  body is read for its `data=` / `out=` arguments: the batcher binds only the
  parameters of the macro a job calls.
- **Behaviour,** against Phase 2 on the 187-file corpus, the 4,800-chunk job
  and the two reference examples:
  - Recovered: SET/MERGE operands after options, UPDATE's transaction, MODIFY's
    transaction, numbered ranges, prefix lists, and `h.output` (WRITE, not
    READ).
  - Removed:
    - `work.view` (`/ view=`), `work.data` (`data;`) and `work.n` (`'x'n`);
    - formats read as librefs (`yymmddn8.`);
    - `%PUT` and `%LET` text, and assignment targets;
    - a bogus libref `&d` from `&d._2`.
  - Batches: D05, D06, D07 and D29 now join the files that write their
    recovered inputs; 131 singletons become 127.
  - Both reference examples are unchanged.
  - **Gone until a later phase:** names that only the old raw scan saw.
    These are SQL `UPDATE` (Phase 5), FedSQL (Phase 5), DS2 and IML
    (Phase 6), and DATASETS `MODIFY` (Phase 4). They are no longer in
    `referenced_datasets`.
- **Performance.** With the extraction limited to step and macro regions,
  `scripts/bench_chunker.py` measures within 0–4% of Phase 0 in one session.

- **`Statement`** fields: `text`, `mt`, `cf` (slices of the region's single
  sanitised text), `keyword`, `context`, `offset`.
- **`statements_of(units)`** yields CODE units only and tracks context: OPEN,
  DATA, `PROC(name)`, nested inside `%MACRO`. It uses the same header and
  terminator rules as Phase 1, including run groups. Step regions and macro
  bodies share it, which removes the duplicated pipeline.
- **Tokenizer `_tokens(mt)`:**
  - names (`DS_REF_TOKEN`, name literals, quoted paths);
  - `=`, `/`, `,`;
  - a balanced `(…)` group as one token;
  - `{…}` (DS2 SQL).
- **DATA-context extractors:**
  - **`data` header:** operands become WRITE; `/ view= …` options are dropped;
    `_null_` is ignored.
  - **`set` / `merge`:** each operand is a name plus an optional options
    group. The list ends at any `name=` option (`end=`, `nobs=`, `point=`,
    `key=`, `indsname=`, `open=`, `cur…`), `/`, or the statement end.
  - **`update` / `modify`:** master and transaction; the MODIFY master is
    UPDATE.
  - **`output`:** every operand.
  - **Hash objects:** `declare|dcl|_new_ hash|hiter (dataset:)` is READ;
    `.output(dataset:)` is WRITE.
  - **Lists:** numbered ranges `ds1-ds3` are expanded (capped at 1,000).
    Prefix lists `lib.pre:` become pattern refs.
- **Macro bodies.** The same extractors run, then `_classify_ref` marks each
  ref as parameter, literal or macro variable. Refs built from several
  parameters are still dropped, as today.
- **`_metadata_for(units, kind)`.** The signature changes: the region's
  `mt`/`cf` are built once with non-CODE units blanked. The remaining
  whole-text scans (functions, CALL routines, macro variables, labels,
  options, symput) run on that masked text, so comments, datalines and
  foreign code count nowhere. (The masking itself, `_Region.code_text`,
  landed in Phase 1.)
- **`referenced_datasets` and `referenced_librefs` become views** (carried over
  from Phase 2): the raw and canonical names of every reference, and their
  librefs plus `defines_librefs`. A `%LET` value that looks like a dataset
  becomes a reference with a role that neither reads nor writes.
- **Deleted:** `_DATASET_RE`, `_SET_RE`/`_MERGE_RE`/`_UPDATE_RE`/`_MODIFY_RE`/
  `_OUTPUT_DS_RE`, all 12 `_BODY_*` patterns, `_io_for`, `_macro_body_io`,
  `_multi_ds`.

Probes flipped: D02–D07, D09, D10, D13, D17, D21, D29, D30, and the
precision probes X06–X12.

## Phase 4: PROC option roles and PROC statements (M) — done

Done as planned, with these differences:

- **Two tables, in `keywords.py`.**
  - `PROC_OPTION_DEFAULTS` holds for every PROC: `data=` read, `out=` and
    `outdata=` write, and the modelling options (`outest=`, `outstat=`,
    `outmodel=`, `outtree=`, `outseed=`, `testout=` write; `inmodel=`,
    `inest=`, `testdata=`, `classdata=` read).
  - `PROC_OPTION_ROLES` adds a PROC's own options or overrides a default.
    Roles: read, write, update, `libref`, `fileref`, and `package` (FCMP's
    `lib.member.package`). An option neither table lists names nothing.
  - `seed=` reads a dataset only in FASTCLUS; elsewhere it is a random seed
    (`seed=&seed` in SURVEYSELECT). `score=` reads only in PROC SCORE.
- **Where options are read.** Anywhere in the PROC statement and in `output`,
  `score`, `baseline` and `forecast`. In any other statement, only after its
  `/` (`tables g / out=f`, `model … / outroc=r`), so `label out = 'x';` and
  `where data = x;` name nothing. Never in an assignment, nor inside a
  dataset's own `(…)` options.
- **One reader per PROC step** (`statements._ProcStep`), because what one
  statement names can depend on another: DATASETS's `lib=`, COPY's `select`.
  - DATASETS also handles `exchange` (UPDATE both). CHANGE reads, drops and
    writes.
  - UPLOAD and DOWNLOAD with `inlib=`/`outlib=` copy like PROC COPY.
- **ODS OUTPUT.** `resolve_ods_outputs` runs between macro-variable
  resolution and database librefs, so names resolve where the statement
  stands and the PROC's database tables see them. It moves a request to the
  next top-level PROC_STEP of the same file and to that region's split
  slices. A request cancelled by `ods output close|clear` stays on its
  statement as a MENTION; one never taken stays a WRITE there.
- **Probe expectations.** P08 and P10 now list the APPEND base among the
  inputs. This is the UPDATE role the breaking-changes table set; the probes
  predate it.
- **Behaviour,** against Phase 3:
  - In the 187-file corpus, 22 chunks' references change, every one through
    an option role, a DATASETS statement, a library copy, ODS OUTPUT or the
    PRINTTO path. `work.tgt` (COPY) and `work.resp` (HTTP) are gone.
  - Batches: 23 → 24, and 127 → 118 singletons. Each merge is a dataset the
    roles now see: CHANGE writes what REG reads, APPEND reads its base,
    COMPARE reads its inputs, LOGISTIC reads its `inmodel=`, and CHANGE
    writes a member another file reads.
  - The 4,800-chunk job and both reference examples are unchanged; only the
    run date stamped in the reports differs.
- **Performance.** `scripts/bench_chunker.py` measures within 0–9% of Phase 0
  in one session; batching is level with Phase 3.

- **`keywords.PROC_OPTION_ROLES`** (pure data). The default for every PROC is
  `data=` read, `out=`/`outdata=` write. Per-PROC entries:

  | PROC | Options |
  |---|---|
  | APPEND | `base=` update; `data=`/`new=` read |
  | COMPARE | `base=`/`compare=` read; `out=`/`outstats=` write |
  | SORT | `dupout=`/`uniqueout=` write; with no `out=`, `data=` is update |
  | FORMAT | `cntlin=` read; `cntlout=` write; `library=` is a catalog |
  | COPY | `in=`/`out=` are librefs |
  | HTTP / JSON | `out=`/`in=`/`headerout=` are filerefs |
  | REG, LOGISTIC, CORR, … | `outest`, `outmodel`, `outp`/`outs`/`outk`/`outh`, `outstat`, `outtree`, `outseed`, `testout`: write. `inmodel`, `inest`, `testdata`, `seed`, `score`, `classdata`: read |
  | UPLOAD / DOWNLOAD | `inlib=`/`outlib=` are librefs |
  | FCMP | `outlib=` (3-level) writes `lib.ds` |

- **Where options are read.** On the PROC statement, and on statements that
  carry options (`output`, `score`, `tables`, `append`, `copy`, `modify`,
  `ods`). A statement shaped like an assignment (`name = …`) is never read,
  so PHREG, NLMIXED, FCMP and IML programs can't produce bogus writes.
- **Statement handlers:**
  - DATASETS: `lib=` sets the default library. `append` is base update plus
    data read; `delete` drops; `change a=b` drops `a` and writes `b`; `copy` +
    `select`/`exclude` as for COPY; `modify` and `age` update; `kill` drops
    the pattern `lib.:`.
  - COPY: members read from `in.` and written to `out.`; with no
    `select`/`exclude`, the pattern `in.:` / `out.:`.
  - CONTENTS: `data=lib._all_` is the pattern `lib.:`.
- **ODS OUTPUT.** Inside a PROC, its datasets are WRITE refs of that PROC.
  In open code they are recorded on the global chunk, then a new
  `resolve_ods_outputs` (run from `resolve_references`) moves them to the next
  PROC_STEP. This stops at `ods output close|clear`, and running the pass
  twice changes nothing.
- **PRINTTO.** A `PathSpec` for PRINTTO `log=`/`print=`.

Probes flipped: P03, P08, P09, P10, P12, P15, P16, P18–P21, P27, P30, P32,
P33, P35, P36, P38.

## Phase 5: one SQL grammar (M), new `chunker/sql.py` — done

Done as planned, with these differences:

- **`sql.SqlStatement(text, dialect)`** owns the walker and
  `split_table_name`; `passthrough` imports both. NATIVE is the old walker
  unchanged, and `tests/test_passthrough.py` passes as it did.
  - `reads()` and `writes()` give NATIVE's raw names.
  - `refs()` gives SAS's `(name, role, clause)` in source order.
- **SAS dialect.** It reads `cf`, where strings are intact. A quoted string is
  a literal anywhere but a table's place, where it is a path or a name
  literal. `%name(…)` there is a macro call, skipped. `DROP TABLE a, b` drops
  a list (SAS only: NATIVE keeps its old reading).
- **Callers.** PROC SQL, PROC FEDSQL and a macro body's SQL fragments.
  `statements._SQL_WRITE_RE` and `_SQL_READ_RE` are deleted. DS2's `{…}`
  waits for Phase 6, which reads DS2's SET.
- **Probe expectations.** S06 and S07 now list the INSERT target among the
  inputs. This is the UPDATE role the breaking-changes table set; the probes
  predate it.
- **Behaviour,** against Phase 4:
  - In the 187-file corpus, comma joins recovered `lib.b`, `lib.c3` and
    `work.lookup`. FedSQL reads and writes. DELETE, UPDATE, ALTER and INSERT
    are UPDATE refs, LIKE reads, and DROP is a DROP ref.
  - Batches: 118 → 112 singletons. A comma-joined lookup now joins the step
    that builds it, the in-place rewrites of `lib.t` chain, and the FedSQL
    step joins the files that write its input.
  - The 4,800-chunk job and both reference examples are unchanged.
- **Performance.** `scripts/bench_chunker.py` measures within 3–11% of
  Phase 0 in one session. Batching is within noise of Phase 4.

- **Move the walker.** `_NativeTables` moves from `passthrough.py` to
  `sql.SqlStatement(text, dialect)`.
  - `NATIVE` reproduces today's behaviour: `--` comments, `@dblink`, CTEs,
    DUAL, and `name(` read as a table function.
  - `SAS` reads `name(…)` after a table as dataset options, excludes
    `dictionary.*`, and accepts quoted paths and name literals.
- **SAS statement roles:**
  - FROM/JOIN items, comma lists included: READ;
  - `create table|view`: WRITE; `create … like t`: reads `t`;
  - `insert into` (VALUES/SET/SELECT), `update`, `delete from`, `alter
    table`: UPDATE;
  - `drop table|view`: DROP; `create index`: none;
  - `select … into :mv` is unchanged.
- **Callers.** PROC SQL, FEDSQL and DS2 `{…}` statements use it. The
  `_SQL_CREATE_RE`/`_FROM_RE`/`_JOIN_RE`/`_INTO_RE` patterns are deleted.
  `passthrough` uses `NATIVE`, so one grammar has one owner.

Probes flipped: S02, S03, S08, S09, S13, S15, E02.

## Phase 6: embedded languages (S)

- **PROC DS2:**
  - DS2 `data out / overwrite=yes;` → WRITE `out`;
  - `set`/`merge` operands, including `set {select …}` through `sql.py`, are
    READ.
- **PROC IML:** `use`/`edit` are READ/UPDATE; `create … from` and `append`
  are WRITE.
- **SUBMIT code** (Python, Lua, Groovy, R) stays opaque from Phase 1. This is
  documented.

Probes flipped: E01 (I/O), E03.

## Phase 7: %INCLUDE, filerefs, macro signatures (M)

- **Multi-value path specs.** `PathSpec` gets a multi-value mode. The include
  spec matches the whole `%include` statement and yields one `SasPathRef` per
  quoted file. `xref/pre.py` rewrites each value through
  `spec.value_spans(match)`.
- **Fileref references.** `%include fref(m1 m2);`, `%include fref;` and
  `infile/file fref` produce refs with `location=FILEREF` and
  `binds=fref`.
- **New `resolve_filerefs` pass** in `resolve_references`:
  - keeps a running map of fileref → path from FILENAME refs, ended by
    `clear` or a rebind (the `resolve_db_librefs` pattern);
  - a FILEREF ref takes the FILENAME's location and path: `dir/member.sas`
    for a directory, or the file itself;
  - `includes` is derived afterwards.
- **Macro signatures.** A new `macro_vars.macro_signature(text)` uses
  balanced-paren extraction and `_split_call_args`. It replaces
  `_MACRO_SIG_RE`, `_ARG_SPLIT_RE` and `_parse_macro_params`, and is used by
  `_MacroDef.of` too.

Probes flipped: G20, G21, M23, plus a new probe for INFILE through a fileref.

## Phase 8: batcher and consumers (S)

- **Dataset flow (`_dataset_flow`, `_build_indices`).** Pattern refs match
  producers sharing their libref and prefix: a sorted name index with bisect,
  then the nearest preceding producer per name. UPDATE refs are handled
  already by "strictly preceding producer". DROP refs are not producers.
- **Complexity report.** The Datasets section adds "Updated in place" and
  "Deleted" lines. The LLM prompt context is unchanged, because UPDATE refs
  already appear in both lists.
- **Validation metrics.** Unaffected; they read the lists.

## Phase 9: documentation (S)

- `chunker/README.md`:
  - layout rows for `statements.py` and `sql.py`;
  - the chunking model: unit roles, run groups, datalines/SUBMIT, macro
    quoting;
  - the metadata section: `dataset_refs`, roles and views;
  - new invariants: only CODE statements are scanned; one SQL grammar; one
    extractor for steps and macro bodies; per-PROC option roles live in
    `keywords.py`.
- `Architecture.md`: the same, plus a migration note listing the breaking
  changes.

## Verification (every phase, then end to end)

- `uv run pytest tests/`: the only allowed failures are the 6 `prompt_builder`
  ones that also fail on `main`, plus the Delta errors this sandbox can't
  avoid. The coverage suite's strict xfail marks shrink phase by phase, to
  zero.
- `uvx ruff@0.15.20 check .` and the pyright ratchet.
- Timing against the Phase 0 baseline after phases 1, 3 and 5. Gate: full
  `chunk_text` on the 900 KB file at most 20% slower.
- **Batch diff.** Batch structure on a synthetic multi-file corpus, before and
  after. Every change is listed in the commit message and justified (UPDATE
  roles, recovered SET inputs, fewer false edges).
- **The two reference examples** end to end (`MANUAL_EXAMPLE` in
  `tests/test_macro_var_datasets.py`, `EXAMPLE` in `tests/test_passthrough.py`):
  `python -m complexity <dir> --hydration` and
  `python -m data_hydration <dir> --dry-run`, with unchanged results.

## Delivery

- One commit per phase (0–9), each green on its own.
- A phase that uncovers a design problem stops for review rather than widening
  its scope.

## Out of scope (stated in docs)

- Separate remote WORK for code inside RSUBMIT. Those statements are
  recognised, but `work.x` inside RSUBMIT is not told apart from local WORK.
- CAS action I/O (`table.loadTable`, `casOut=`).
- Code generated at run time (CALL EXECUTE text, `dosubl`) beyond the macro
  names invoked.
- `%IF … %THEN %call(…)` followed by another statement in the same unit.
- A full SAS grammar. The statement-scanner design stays.
