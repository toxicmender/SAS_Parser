"""Shared prompt constants — importable without langchain installed.

Both templates are filled by ``.format()`` at pipeline construction (never
f-stringed at import time) from the run's resolved
:class:`~target_language.TargetLanguage`: ``{output_language}`` is its display
name, ``{fence_info}`` the Markdown fence tag its code blocks must carry, and
``{cell_language}`` the value a structured cell's ``language`` field takes.
Naming all three is what makes the target enforceable downstream — the
validation suite scores the emitted fences against exactly these.
"""

# The target-exclusivity clause both templates share: naming the target once
# is not enough on its own, because a model that knows PySpark well will drift
# into it when the SAS construct maps more naturally there.
# How the target is *carried* differs between the two templates (a fence tag
# vs a schema field), so that sentence lives in each of them; only the
# exclusivity rule is shared.
_LANGUAGE_RULE = """\
## Target language (strict)
Translate into {output_language} and nothing else. Do not emit any other
target language as runnable code — not as an alternative, not as a fallback,
and not "for comparison" — and never present the original SAS as the
translation. If a SAS construct has no clean {output_language} equivalent,
say so in the risks and give the closest {output_language} form; do not
switch languages to make it fit. There are exactly two exceptions. The
`NOT CONVERTIBLE` marker described below, whose suggested equivalent is
always commented out and therefore never runs. And a
`## Target language for THIS item` section in the batch context: that is the
run naming a different target for one item because {output_language} cannot
express what the item contains — when it appears, it wins over this section,
for that item only.
"""

# The preamble both templates open with — same wording, so a run reads the
# same whether or not the model can be asked for a schema.
_PREAMBLE = """\
You are an expert SAS-to-{output_language} migration assistant.
You will be given either a single semantic chunk of Base SAS source code,
or a dependency batch containing several chunks (possibly from different
source files) that must be translated together because they share
dataset, macro, or macro-variable dependencies.
"""

_SYSTEM_PROMPT_TEMPLATE = (
    _PREAMBLE
    + "\n"
    + _LANGUAGE_RULE
    + """
Structure every response with these Markdown sections, in order:

## Analysis
Identify the SAS construct(s) and their purpose. Before writing any code,
reason step by step through whatever bears on correctness for this item:
execution order, PDV vs DAG semantics, macro expansion timing, and any
hazard flags surfaced in the context below (SYMPUT scope hazards, %ABORT,
computed %GOTO).

## Mapping
For each construct, its {output_language} equivalent and any semantic
difference (date epoch offsets, MERGE vs join defaults, PDV vs DAG
execution, macro expansion, PROC step equivalents, etc.).

## Translation
The {output_language} code, in fenced blocks tagged ```{fence_info} — that
tag is how the translation is told apart from an illustrative snippet, so
every block of translated code carries it and nothing else does. When
translating a batch, preserve execution order across member chunks/files and
make cross-file/cross-chunk dependencies explicit.

Preserve what the SAS source specifies: table and column names, join types,
filter predicates, grouping, ordering, and any de-duplication it asks for.
Renaming or dropping one of those is a decision to report under Risks, never
a silent one. Equally, add nothing it does not ask for \u2014 no extra
aggregations, statistics, orderings, or physical-layout directives.

An input dataset that no earlier step produces is an external dependency:
name it as one rather than inventing a definition for it.

## Risks
Flag every P0 silent-error risk with a \u26a0\ufe0f marker. If a translation is
ambiguous or unsafe, say so explicitly rather than guessing.

Where a construct genuinely has no {output_language} equivalent, mark the
spot in the code with a comment reading
`{comment_prefix} NOT CONVERTIBLE TO {output_language}: <reason>`, follow it
with the closest equivalent **commented out**, and explain it here. That
marker is the one place another language may appear, and only ever as a
comment - never as code.

Reason as thoroughly as the item requires in Analysis and Mapping; keep
Translation and Risks concise.
"""
)

# Structured-output system prompt — used when the pipeline asks for a
# TranslationDocument instead of free-form Markdown. Same reasoning demands as
# above; the four sections become schema fields, and the Translation section
# becomes the ordered `cells` list a notebook runs top to bottom.
_STRUCTURED_SYSTEM_PROMPT_TEMPLATE = (
    _PREAMBLE
    + "\n"
    + _LANGUAGE_RULE
    + """
Answer with the structured document you have been given a schema for. Fill
every field:

- `analysis`: identify the SAS construct(s) and their purpose. Before writing
  any code, reason step by step through whatever bears on correctness for this
  item: execution order, PDV vs DAG semantics, macro expansion timing, and any
  hazard flag surfaced in the context below (SYMPUT scope hazards, %ABORT,
  computed %GOTO).
- `mapping`: one entry per construct — its {output_language} equivalent and any
  semantic difference (date epoch offsets, MERGE vs join defaults, PDV vs DAG
  execution, macro expansion, PROC step equivalents, etc.).
- `cells`: the {output_language} code, split into the cells a notebook should
  run in order. Each code cell must be complete, runnable {output_language}
  with `language` set to '{cell_language}' — never a Markdown fence, never a
  prose paragraph; put prose in a markdown cell
  or in the cell's `comment`. When translating a batch, preserve execution order
  across member chunks/files and make cross-file/cross-chunk dependencies
  explicit. When the batch has several members, set every cell's `chunk_id` to
  the member id it implements, exactly as listed under '## Batch members' — it
  routes the cell into its source file's notebook (a cell serving several
  members carries the id of the one whose step it completes).
  Preserve what the SAS source specifies: table and column names, join types,
  filter predicates, grouping, ordering, and any de-duplication it asks for —
  renaming or dropping one of those belongs in `risks`, never silent. Equally,
  add nothing it does not ask for: no extra aggregations, statistics,
  orderings, or physical-layout directives. An input dataset no earlier step
  produces is an external dependency; name it as one rather than inventing a
  definition. Where a construct genuinely has no {output_language} equivalent,
  mark the spot with a
  `{comment_prefix} NOT CONVERTIBLE TO {output_language}: <reason>` comment and
  follow it with the closest equivalent **commented out** — that marker is the
  one place another language may appear, and only ever as a comment.
- `risks`: every risk worth flagging, worst first, P0 for a silent-error risk.
  If a translation is ambiguous or unsafe, say so explicitly rather than
  guessing. Include every `NOT CONVERTIBLE` marker left in the cells.

Reason as thoroughly as the item requires in `analysis` and `mapping`; keep
the cells and `risks` concise.
"""
)

# Singleton-chunk context (SasChunk items in all_ordered_items).
_BATCH_MEMBER_TEMPLATE = """\
### {chunk_id}  [{kind}]  ({source_id}, lines {start_line}-{end_line})
Title: {title}
```sas
{text}
```
"""

_BATCH_CONTEXT_TEMPLATE = """\
## Batch context
- Batch id          : {batch_id}
- This item         : {index}/{total_items}
- Cross-file batch  : {is_cross_file}
- Source files      : {source_files}
- Member chunks     : {chunk_count} (lines {start_line}-{end_line})
- Grouping reason   : {reason}
- Datasets (in)     : {input_datasets}
- Datasets (out)    : {output_datasets}
- Macros (required) : {required_macros}
- Macros (defined)  : {defined_macros}
- Librefs (required): {required_librefs}
- Autocall macros   : {standard_autocall_macros}
- Macrovars (req)   : {required_macrovars}
- Macrovars (prod)  : {produced_macrovars}
- PROCs run         : {proc_names}
- DATA-step stmts   : {data_step_statements}
- SAS functions     : {sas_functions}
- CALL routines     : {call_routines}
- Component objects : {component_objects}
- Global stmt kws   : {global_statement_keywords}
- ⚠️ SYMPUT hazard  : {symput_hazard}
- ⚠️ Contains ABORT: {contains_abort}
- ⚠️ Computed GOTO : {contains_computed_goto}
- Diagnostics       : {diagnostics}
{target_directive}
## Batch members
{members}
"""

#: Injected into the batch context only when an item's target differs from the
#: run's — see :func:`complexity.fallback.choose_target`.
#:
#: It lives *here*, in the per-item message, and not in the system prompt: the
#: system block is built once and cached (Architecture.md invariant 6), so a
#: target that varied per item would miss the prompt cache on every one. That
#: makes this ephemeral context in the sense of invariant 5 — prompted, never
#: persisted.
#:
#: The reason is stated, not just the instruction. A model told to write PySpark
#: directly under a system prompt demanding Spark SQL will otherwise try to
#: reconcile the two, and reconciling usually means emitting SQL anyway.
_TARGET_OVERRIDE_TEMPLATE = """
## Target language for THIS item: {output_language}

This item uses {reasons}, which {run_language} cannot express — so translate
**this item only** into {output_language}, overriding the target named in your
instructions. Later items return to {run_language}.

- Fenced blocks: ```{fence_info}
- Structured cells: `language` = '{cell_language}'
- Comments: {comment_prefix}

Do not translate the surrounding logic into {output_language} beyond what this
item contains, and do not apologise for the switch or explain it in prose — the
notebook records it.
"""
