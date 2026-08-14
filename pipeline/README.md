# pipeline

SAS → target-language LLM pipeline: feeds the chunker/batcher's work items,
in dependency order, through a LangChain/LangGraph chat model with per-run
conversational memory, and renders the responses as runnable notebooks.
This is the integration layer — the only package that imports `chunker`,
`memory`, `llm_client`, and `prompt_builder` together.

For the whole-system view see the repository
[Architecture.md](../Architecture.md); for the parsing/batching layers see
the [chunker README](../chunker/README.md).

## Target output language

`output_language` is resolved once, at construction, through the
[`target_language`](../target_language/__init__.py) registry, and the resolved
object drives every stage that has an opinion about the language: the system
prompt (which names the target, the fence tag its code must carry, and forbids
any other), the `[lang: ...]` instruction axis, the notebook kernel and cell
tags, and what `validation` checks the emitted code parses as.

```python
pipe = SasLLMPipeline(output_language="spark sql")
pipe.output_language        # "Spark SQL" — canonical, whatever you typed
pipe.target_language        # the resolved TargetLanguage; pass it downstream
```

Known targets are **PySpark** and **Spark SQL**; spelling is
folded (`SparkSQL`, `spark sql`, `spark_sql`, and `sql` are one target). An
unrecognised name raises `UnknownTargetLanguage` instead of silently behaving
like a Python run. Omit the argument to take config.json
`pipeline.output_language`, then the code default.

Asking for a language is not the same as getting one, so the request is
enforced on the way back: `validation`'s `language_compliance` scores the
fraction of translated blocks actually in the target, and with
`validation_retries > 0` an off-target answer is rolled back and re-prompted
with a note naming the offending fence tag.

### When Spark SQL cannot express an item

Some SAS has no Spark SQL answer at all — a `%MACRO` definition, `CALL EXECUTE`,
`PROC FCMP`. Asked for SQL anyway, a model produces something plausible, and the
failure is silent: `target_syntax` parses it happily because it *is* valid SQL,
just not equivalent SAS.

So a Spark SQL run resolves the target **per item**. When an item's constructs
rate `HARD` or `MANUAL` against Spark SQL and better against PySpark
(`complexity.fallback`, Architecture.md invariant 15), that item is translated
into PySpark instead. That covers the macro facility (`%MACRO`, `CALL EXECUTE`,
`PROC FCMP`, …) and the DATA step's procedural core — an iterative `DO` loop and
`LINK`/`RETURN`, both found by scanning source rather than from chunk metadata.
When it fires:

- the override is announced in the item's **batch context**, never the system
  prompt — the system block is the cached prefix, and varying it per item would
  miss the prompt cache every time;
- the output carries `target_language` and `fallback_reasons`;
- the notebook is hosted by the Python kernel and its SQL cells get the `%sql`
  magic, because a SQL kernel cannot run Python;
- the item is validated against *its* target, so a correct PySpark fallback is
  not scored 0.0 for containing no SQL;
- the item's notebook header says which construct forced it.

An all-SQL run is unaffected — same prompts, same notebook, byte for byte. Set
config.json `pipeline.sql_fallback` to `false` to keep every item on Spark SQL
and see the failures instead.

## Quick start

```python
from pipeline import SasLLMPipeline

pipe    = SasLLMPipeline()  # model="gpt-5.4" by default
outputs = pipe.run_files(["macros.sas", "etl.sas", "reports.sas"])

# what the run cost, as the gateway reported it
print(pipe.token_usage.summary())
```

Transport and memory are configured as two objects, and only as two
objects — there is no per-knob spelling on the constructor:

```python
from llm_client import LLMClientConfig
from pipeline import MemorySetup, SasLLMPipeline

pipe = SasLLMPipeline(
    llm_config=LLMClientConfig(model="claude-sonnet-4-5", max_retries=5),
    memory_setup=MemorySetup(task_id="migration-2026"),
)
```

`token_usage` is cumulative over the pipeline's lifetime, not per `run_*`
call: a caller attributing one run's spend snapshots it before and subtracts
after (`llm_client.TokenUsage` supports `-`). It stays at zero when the
gateway reports no usage block.

## Package layout

| File | Role |
|------|------|
| `engine.py` | `SasLLMPipeline`: the LangGraph `StateGraph` wiring, memory/validation integration, resume and fork, opt-in Anthropic prompt caching. |
| `setup.py` | `MemorySetup`: the pipeline's memory wiring (store hub, task policy, thread memory, extractor, chat identity), with the cross-injection logic in `build()`. |
| `run_ledger.py` | `RunLedger`: the KV-side run bookkeeping — per-item outcome facts and stored verdicts, resume (what to skip, what to redo, the rewind), and the fact-copying half of fork. Never calls an LLM. |
| `prompting.py` | Item → retrieval query / construct keys / scope tokens mapping (the sole SAS-metadata → `prompt_builder` bridge) and batch prompt formatting. The `## Batch context` block reports the batch's rollups — datasets, macros, PROCs run, DATA-step statements, functions, routines, component objects, global statements — so the facts the model is shown cover the keys its guidance was selected on. `_attribution_for_item` derives the same keys per member, which lets a multi-member batch's guidance name the step each section serves. |
| `constants.py` | Prompt templates — the Markdown-sections system prompt and its structured-output counterpart (importable without langchain installed). |
| `response_models.py` | `TranslationDocument` (+ `TranslationCell`, `MappingEntry`, `RiskNote`): the structured answer the pipeline asks for, and `to_markdown()`, which renders it back to the four `##` sections that get persisted and scored. Pydantic only. |
| `notebook.py` | Renders pipeline outputs as nbformat v4.5 notebooks — one `.ipynb` per SAS source file plus `_cross_file.ipynb` — from a `TranslationDocument`, or by parsing the Markdown response when there is none. Stdlib + `response_models`. |

`SasLLMPipeline` is resolved lazily from the package root, so importing
`pipeline.notebook` or `pipeline.response_models` never pulls in
langchain/langgraph.

## Pipeline and memory

`SasLLMPipeline` compiles a one-node LangGraph `StateGraph(MessagesState)`. The
model node loads the thread's history from `KVChatMessageHistory`, runs
`_trim | prompt | LLMClient`, and persists exactly the prompted message plus the
response in one bulk `add_messages` write (trimming only limits what is
*prompted*; storage keeps every turn).

Prompted-history trimming has two modes: the default `window_k` recency window,
or — when a `memory.relevance.RelevantHistorySelector` is passed as
`history_selector` — relevance-based selection (see the [memory README](../memory/README.md)).
All items of one `run_file` / `run_text` / `run_files` call share one thread
(`thread_id = "run::<source ids>"`), so the LLM sees the run's accumulated
context batch by batch. Those calls send `SasBatch` objects only:
`coalesce_into_batches` first merges the run's standalone singleton chunks into
`merged-NNN` batches (capped at `max_merged_chunks`), so the model is never
prompted with a bare `SasChunk`. Coalescing is **token-budgeted call
packing, on by default**: adjacent items — small dependency batches
included — share one LLM call as `packed-NNN` batches while the estimated
prompt cost (counted with the pipeline's tokenizer, the same encoding as
`max_input_tokens`) stays under `max_merged_tokens` (constructor argument,
else config.json `pipeline.max_merged_tokens`, else a derived default —
0.8 × the input-token headroom when `max_input_tokens` is set, ~64k tokens
otherwise; pass `0` to disable). The global-context batch never packs. The mapping is deterministic in the
budgets, so resume and `fork_run` reproduce the same batch ids — but packed
ids differ from unpacked ones, so resume only matches runs made under the
same budget. `llm_client.LLMClient` owns model
construction (temperature, output-token cap, an optional proactive rate
limiter — on for the `from_ai_gateway` credential path) and invocation
(input-token budget, transient-error retry that honors a gateway
`Retry-After`); an injected `llm` still gets the retry / budget layers.

## Structured output → notebooks

With `structured_output` on (constructor argument, else config.json
`pipeline.structured_output`, default on) the model is asked for a
`TranslationDocument` rather than free-form Markdown, so `notebook.py` knows
which cells are runnable code instead of inferring it from fences. The turn
*persisted* to memory is still `to_markdown()` — the same four sections, code
in fenced blocks — with the document carried on the AI message's
`additional_kwargs["translation_document"]` and surfaced as `document` in each
output dict. That keeps conversation memory, resume, and every `validation`
metric unchanged; storing the raw content instead would store an empty turn
whenever the answer rides in a tool call. A model or gateway that cannot honour
the schema degrades to prose (detected at construction, or demoted once
mid-run) and the notebook is built by parsing the Markdown.

Multi-source batches are **split back per source file** when the document's
cells are cleanly attributed: the structured prompt asks the model to tag
every cell with the `chunk_id` of the batch member it implements, and each
output dict carries a `chunk_sources` map (`chunk_id` → source file) the
renderer routes by. The split is all-or-nothing per item — any untagged or
unresolvable *code* cell (or a Markdown-fallback item, which has no
attribution) sends the whole item to the shared `_cross_file.ipynb` with
pointer cells in each participating notebook, the pre-split behavior. The
per-item header, Analysis/Mapping, untagged prose, and Risks are duplicated
into every participating notebook so each one stands alone.

```python
from pipeline.notebook import write_notebooks

outputs = pipe.run_files(sas_files)
write_notebooks(outputs, "out", output_language=pipe.target_language)
# out/<source>.ipynb per file; out/_cross_file.ipynb only for items whose
# cells could not be attributed per source
```

## Load-bearing invariants

- **The LangGraph graph is compiled *without* a checkpointer, on purpose.**
  Durable per-thread persistence lives in the KV `msg::` row schema that
  `snapshot()`, `prune_before()`, and `list_threads()` depend on. A
  `BaseCheckpointSaver` would store full state blobs per turn (O(n²) growth in
  the Delta table) and duplicate the canonical store. One graph invocation is
  one conversational turn — the node prompts with, and persists, exactly the
  last state message.
- **Ephemeral context is prompted but never persisted** — reference guidance,
  the rolling summary, and short-term thread notes ride outside the `msg::`
  history. See Architecture.md invariants 5–6 for the full statement.

## Logging

f-string messages everywhere (never lazy `%`-style). Logger names follow
modules: `pipeline.engine`, `pipeline.prompting`, `pipeline.notebook`.
