"""Shared prompt constants — importable without langchain installed."""

# ---------------------------------------------------------------------------
# System prompt (fills {output_language}; do not f-string this at import
# time — SasLLMPipeline resolves it once at construction via .format()).
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert SAS-to-{output_language} migration assistant.
You will be given either a single semantic chunk of Base SAS source code,
or a dependency batch containing several chunks (possibly from different
source files) that must be translated together because they share
dataset, macro, or macro-variable dependencies.

For each item you receive:
1. Identify the SAS construct(s) and their purpose.
2. Translate to equivalent {output_language}, noting any semantic
   differences (date epoch offsets, MERGE vs join defaults, PDV vs DAG
   execution, macro expansion, PROC step equivalents, etc.).
3. Flag any P0 silent-error risks with a \u26a0\ufe0f marker \u2014 pay particular
   attention to any hazard flags already surfaced in the context below
   (SYMPUT scope hazards, %ABORT, computed %GOTO).
4. When translating a batch, preserve execution order across member
   chunks/files and make cross-file/cross-chunk dependencies explicit.
5. If translation is ambiguous or unsafe, say so explicitly rather than
   guessing.

Be concise. Respond in structured Markdown.
"""

# ---------------------------------------------------------------------------
# Singleton-chunk context (used for SasChunk items in all_ordered_items)
# ---------------------------------------------------------------------------

_CONTEXT_TEMPLATE = """\
## Program context
- Source file       : {source_id}
- This item         : {chunk_id} ({index}/{total_items})
- Kind              : {kind}
- Title             : {title}
- Datasets (ref)    : {datasets}
- Librefs           : {librefs}
- Datasets (in)     : {input_datasets}
- Datasets (out)    : {output_datasets}
- Macros (def)      : {macro_defs}
- Macros (call)     : {macro_calls}
- Macro var op      : {macro_var_op}
- Macro vars (decl) : {declared_macro_vars}
- Macro vars (ref)  : {referenced_macro_vars}
- Macrovars (prod)  : {produced_macrovars}
- Macrovars (cons)  : {consumed_macrovars}
- Global stmt kw    : {global_statement_keyword}
- Control-flow op   : {control_flow_op}
- SAS functions     : {sas_functions}
- CALL routines     : {call_routines}
- Automatic vars    : {automatic_vars}
- \u26a0\ufe0f SYMPUT hazard  : {symput_hazard}
- \u26a0\ufe0f Contains ABORT: {contains_abort}
- \u26a0\ufe0f Computed GOTO : {contains_computed_goto}
- Diagnostics       : {diagnostics}

## Chunk source
```sas
{text}
```
"""

# ---------------------------------------------------------------------------
# Batch context (used for SasBatch items in all_ordered_items)
# ---------------------------------------------------------------------------

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
- Autocall macros   : {standard_autocall_macros}
- Macrovars (req)   : {required_macrovars}
- Macrovars (prod)  : {produced_macrovars}
- Diagnostics       : {diagnostics}

## Batch members
{members}
"""
