# pipeline

SAS → target-language LLM pipeline: feeds the chunker/batcher's work items,
in dependency order, through a LangChain/LangGraph chat model with per-run
conversational memory, and renders the responses as runnable notebooks.
This is the integration layer — the only package that imports `chunker`,
`memory`, `llm_client`, and `prompt_builder` together.

For the whole-system view see the repository
[Architecture.md](../Architecture.md); for the parsing/batching layers see
the [chunker README](../chunker/README.md).

## Quick start

```python
from pipeline import SasLLMPipeline

pipe    = SasLLMPipeline()  # model="gpt-5.4" by default
outputs = pipe.run_files(["macros.sas", "etl.sas", "reports.sas"])

# what the run cost, as the gateway reported it
print(pipe.token_usage.summary())
```

`token_usage` is cumulative over the pipeline's lifetime, not per `run_*`
call: a caller attributing one run's spend snapshots it before and subtracts
after (`llm_client.TokenUsage` supports `-`). It stays at zero when the
gateway reports no usage block.

## Package layout

| File | Role |
|------|------|
| `engine.py` | `SasLLMPipeline`: the LangGraph `StateGraph` wiring, memory/validation integration, resume and fork, opt-in Anthropic prompt caching. |
| `prompting.py` | Item → retrieval query / construct keys / scope tokens mapping (the sole SAS-metadata → `prompt_builder` bridge) and chunk/batch prompt formatting. |
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
prompted with a bare `SasChunk`. The mapping is deterministic, so resume and
`fork_run` reproduce the same batch ids. `llm_client.LLMClient` owns model
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

```python
from pipeline.notebook import write_notebooks

outputs = pipe.run_files(sas_files)
write_notebooks(outputs, "out", output_language="PySpark")
# out/<source>.ipynb per file, out/_cross_file.ipynb for cross-file batches
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
