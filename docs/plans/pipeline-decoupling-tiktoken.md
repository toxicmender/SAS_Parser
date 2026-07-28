# Plan: pipeline decoupling, tiktoken-by-default, and default-workflow simplification

Status: all phases (1–5, including the Phase 4 default flip) implemented on this branch: packing is on by default with the derived budget, max_merged_tokens=0 disables it. §3 still recommends validating answer quality against a real model on the sample corpus and tuning the default budget from the results. Scope of this document:

1. Decoupling — what is coupled today and why it hurts.
2. Abstracting the pipeline out of the `chunker` package.
3. tiktoken as the default encoding / token estimator, with `gpt-5.4` as the
   default model.
4. Simplification driven by the current default workflow (`demo_run.py local`).
5. Fewer LLM calls: token-budgeted packing of singletons and adjacent small
   batches.
6. A complete notebook per converted SAS source file.

## 1. Evaluation of the current state

### 1.1 What is already decoupled (and must stay that way)

The repo's import direction is deliberately strict (see Architecture.md):
`chunker.pipeline` is the *sole* module that imports `memory.*`, `llm_client`,
and `prompt_builder`; those three never import `chunker` or each other. The
SAS-metadata → `(query, constructs)` mapping lives in the pipeline precisely so
`prompt_builder` needs no `chunker` import. `chunker/__init__.py` lazily loads
`SasLLMPipeline` via `__getattr__`, so importing the chunker alone never pulls
langchain/langgraph. That design is sound; the problems are placement and
accretion, not direction.

### 1.2 What is coupled, concretely

- **Placement**: the orchestration layer lives *inside* the parsing package.
  `chunker/` contains both a dependency-light SAS scanner/batcher and a
  ~2,000-line LangGraph application (`pipeline.py`), its prompt templates
  (`pipeline_constants.py`), its response schema (`response_models.py`), and
  its deliverable renderer (`notebook.py`). "chunker" no longer names what the
  package does, and the lazy-import trick in `__init__.py` exists only to work
  around the placement.
- **God constructor**: `SasLLMPipeline.__init__` takes ~35 keyword arguments
  spanning five concerns — LLM transport (10 args that are really
  `LLMClientConfig` fields re-declared), chunk/batch knobs, memory wiring
  (7 args), validation, and prompting. Every new feature has widened it.
- **`pipeline.py` mixes five responsibilities** in one module/class:
  1. item → retrieval query/constructs/meta-flags mapping (module functions);
  2. chunk/batch prompt formatting (`_format_*_message` + templates);
  3. LLM orchestration (LangGraph graph, trimming, structured-output
     degradation, prompt caching);
  4. memory wiring (MemoryHub/TaskPolicy/ThreadMemory/extractor/summarizer
     auto-construction and cross-injection);
  5. run control (resume, fork, run facts, validation-driven retry/rewind).
- **A stray upward dependency**: `_load_sharepoint_databricks_mapping` reaches
  into `app_config.sharepoint` from the pipeline constructor. Operational I/O
  (reading a CSV off SharePoint) does not belong in library construction; it
  belongs with the caller (`demo_run.py`).

### 1.3 Token estimation today (the tiktoken gap)

Token counting exists in three disconnected tiers:

| Where | What | Default |
|---|---|---|
| `llm_client.count_tokens` | `max_input_tokens` budget | model-native `get_num_tokens_from_messages`; non-GPT models get `tiktoken_model_name="gpt-5"` proxy; any failure → `chars//4` |
| `memory.turns.approx_token_count` | summarizer trigger, history `max_tokens` packing | `chars//4`, overridable per component via `token_counter=` |
| `chunker` / `prompt_builder` budgets | chunk/instruction sizing | **word** counts, not tokens |

The concrete bug-shaped gap for `gpt-5.4`: `_TIKTOKEN_COUNTED_PREFIXES`
includes `"gpt-5"`, so `"gpt-5.4"` is assumed to count itself and **no**
`tiktoken_model_name` stand-in is set. But tiktoken's model table keys on
`"gpt-5-"` (dash) prefixes; `encoding_for_model("gpt-5.4")` raises, and
langchain-openai either falls back to an old vocabulary or the whole counter
falls through to `chars//4`. Today a Claude model gets *better* token counting
through this client than `gpt-5.4` does. (Verify against the locked
tiktoken/langchain-openai versions, but the fix below is right either way:
resolve the encoding explicitly, never by model-name lookup.)

Also: the default model everywhere is `claude-sonnet-4-5`
(`LLMClientConfig.model`, `SasLLMPipeline(model=...)`), and tiktoken is only a
transitive dependency (via langchain-openai), not a declared one.

### 1.4 The default workflow, and what it makes dead weight

The default workflow (`demo_run.py local`, and `sharepoint` mode) is:

```
PromptBuilder.from_reference_dir + .sas files
  → SasLLMPipeline.run_files → chunk → MultiFileBatcher
  → coalesce_into_batches → per-batch LLM calls on one thread
  → LiveValidator inline scoring → notebooks + Markdown report
```

Because `_items_as_batches` **always** coalesces (`coalesce_into_batches`
wraps every singleton chunk into a `merged-NNN` batch, even at
`max_chunks=1`), the pipeline's item type is effectively `SasBatch` only.
Yet the whole file is written against `SasBatch | SasChunk`:
`_query_for_item`, `_identifier_sets`, `_constructs_for_item`,
`_kinds_for_item`, `_meta_flags_for_item`, `_format_chunk_message`,
`_diagnostics_for_chunk`-as-item, and a dozen
`item.batch_id if isinstance(item, SasBatch) else item.chunk_id` ternaries —
all to support a chunk-shaped code path that `run_file`/`run_text`/`run_files`
can no longer reach.

### 1.5 LLM-call anatomy: where calls are spent, where they can be saved

One run makes exactly one LLM call per coalesced item (plus opt-in extras:
validation retries, summarizer folds, extractor classifications). The
coalescing layer (`chunker.batcher.coalesce_into_batches`) currently reduces
calls only within one narrow case:

- **Singletons**: each maximal run of *consecutive* independent singleton
  chunks is packed into `merged-NNN` batches — but capped by **member count**
  (`max_merged_chunks`, default 8), blind to size. Eight one-line `%let`
  chunks and eight 600-word DATA steps hit the same cap, so small chunks
  under-fill their requests while large ones can overshoot
  `max_input_tokens`.
- **Real dependency batches are never packed.** Every `SasBatch` from the
  batcher passes through as its own LLM call *and* acts as a flush barrier
  for the pending singleton run on either side of it. A corpus that batches
  into many small two-chunk dependency batches gets no call reduction at
  all — and each call re-pays the fixed per-call overhead (system prompt —
  uncached on non-Anthropic models — plus history window and retrieved
  guidance), which for small items dominates the item text itself.

So the two levers are: make singleton merging **token-budgeted** instead of
count-budgeted, and extend packing to **adjacent small batches** under the
same budget. Both are call-level packing concerns and belong in the
coalescing layer — the batcher's dependency grouping (union-find, weak-edge
resolution, batch identity) must not change: batch membership and reason
strings are pinned by tests, and complexity scoring and validation read the
same units.

## 2. Plan

Five phases, each independently shippable and separately committed
(conventional commits, one logical change per commit). Phase 1 is pure code
motion; phases 2–5 change behavior in narrow, stated ways. See §3 for
sequencing rationale and cross-phase interactions.

### Phase 1 — move the pipeline out of `chunker` (code motion only)

Create a top-level `pipeline/` package:

```
pipeline/
  __init__.py        re-exports SasLLMPipeline, TranslationDocument, notebook API
  engine.py          SasLLMPipeline (from chunker/pipeline.py, orchestration half)
  prompting.py       item → query/constructs/kinds/meta-flags mapping +
                     _format_chunk_message/_format_batch_message (module functions)
  constants.py       from chunker/pipeline_constants.py
  response_models.py from chunker/response_models.py
  notebook.py        from chunker/notebook.py (renders pipeline output, not SAS)
```

- `chunker/` keeps: `models`, `keywords`, `scanner`, `metadata`, `chunker`,
  `batcher`, `_repl` — a pure SAS→chunks/batches library. The
  `__getattr__` lazy-import hack in `chunker/__init__.py` is deleted.
- **Back-compat shims** for one release: `chunker/pipeline.py`,
  `chunker/response_models.py`, `chunker/notebook.py` become 3-line modules
  re-exporting from `pipeline.*` with a `DeprecationWarning`; the
  `__getattr__` in `chunker/__init__.py` keeps resolving `SasLLMPipeline` (and
  the notebook/response names) with the same warning. Known consumers to
  migrate in the same PR: `demo_run.py`, `validation/runner.py`,
  `validation/live.py` (docstring refs), `tests/test_pipeline*.py`,
  `tests/test_notebook.py`, Architecture.md, READMEs.
- Import direction after the move: `pipeline` sits above `chunker`, `memory`,
  `llm_client`, `prompt_builder` and is the only package importing all four;
  `validation` sits above `pipeline`. `chunker` drops to a leaf beside
  `prompt_builder`. Nothing else changes.
- Logger names follow the modules (`pipeline.engine`, `pipeline.prompting`);
  update the Conventions list in Architecture.md.
- Verification: the existing pipeline/notebook test suites, plus the
  documented snapshot-diff practice for the untouched chunker/batcher.

### Phase 2 — tiktoken by default, `gpt-5.4` as the default model

New leaf module `llm_client/tokens.py` (imports nothing from the repo):

- `default_encoding(model: str) -> tiktoken.Encoding`: explicit resolution —
  `gpt-5*`/`gpt-4o*` → `o200k_base`, older GPT names via
  `tiktoken.encoding_for_model`, **everything unknown (incl. `gpt-5.4`,
  `claude-*`, `gemini-*`) → `o200k_base`** — never a bare model-name lookup
  that can raise mid-run.
- `count_text(text, model=...) -> int` and
  `count_messages(messages, model=...) -> int` (per-message overhead constant,
  matching the ChatML ~3-tokens-per-message accounting), with a module-level
  encoding cache.
- Offline degradation stays: tiktoken loads `o200k_base` from its cache or the
  network; on failure, fall back to `chars//4` with the existing one-time
  WARNING. (This keeps the "tests run without network" property.)

Wire it as the default:

- `llm_client.client`: replace the `_TIKTOKEN_COUNTED_PREFIXES` /
  `_TIKTOKEN_PROXY_MODEL` special-casing — `count_tokens` uses
  `tokens.count_messages(model=config.model)` unless a custom
  `token_counter` is configured; drop the `tiktoken_model_name` kwarg on
  `ChatOpenAI` (no longer needed once we own counting). Fixes the `gpt-5.4`
  gap in §1.3.
- `LLMClientConfig.model` default: `"claude-sonnet-4-5"` → `"gpt-5.4"`.
  Same for the `SasLLMPipeline(model=...)` default and the config.json
  comments / README examples. Prompt caching stays Anthropic-gated and
  therefore off by default — unchanged semantics, now just inert for the
  default model.
- `memory.turns.approx_token_count` callers: `RollingSummarizer` and
  `RelevantHistorySelector` keep their `token_counter=None` signature, but
  `None` now resolves to the tiktoken counter (falling back to `chars//4`
  offline). The pipeline passes its model name down so all tiers count under
  one vocabulary.
- `pyproject.toml`: declare `tiktoken` explicitly (it is already installed
  transitively; pin `>=` the version whose `o200k_base` ships).
- **Deliberately out of scope**: converting the chunker's `min_words` /
  `max_words` and prompt_builder word budgets to token budgets. That moves
  chunk boundaries, which moves batch membership, which is pinned by tests and
  by the snapshot-diff discipline. If wanted later, it is its own phase with
  recalibrated defaults (~1.3 tokens/word for code) and a config switch.

Tests: unit-test `default_encoding` resolution (incl. `gpt-5.4` → `o200k_base`
and the offline fallback), and that `LLMClient.count_tokens` on a fake model
no longer warns/falls back for `gpt-5.4`.

### Phase 3 — simplification around the default workflow

In rough order of value:

1. **Normalize to `SasBatch` at the boundary.** `_process` and everything
   below it takes `Sequence[SasBatch]` only; `_items_as_batches` already
   guarantees it. Delete the `SasChunk` half of the item-shaped unions:
   `_identifier_sets` collapses into `_constructs_for_item` reading batch
   aggregates, `_format_chunk_message` + `_CONTEXT_TEMPLATE` go away (batch
   template covers the single-member case), and every
   `isinstance(item, SasBatch)` ternary becomes `item.batch_id`. The public
   `run_*` outputs keep their shape (`is_batch` stays, now always `True`;
   note it as deprecated in the docstring).
2. **Group the constructor.** Keep the existing kwargs working, but make the
   canonical form
   `SasLLMPipeline(llm_config: LLMClientConfig | None, memory_config: MemorySetup | None, ...)`:
   the ten transport kwargs collapse into an optional `LLMClientConfig`
   (already the internal representation — today they are unpacked only to be
   repacked), and the seven memory kwargs into a small `MemorySetup` dataclass
   whose wiring logic (`store injection`, extractor implies thread-memory,
   policy snapshot) moves out of `__init__` into `MemorySetup.build()`. The
   legacy kwargs stay accepted for one release, forwarded into the configs.
3. **Move SharePoint mapping I/O out of the constructor.**
   `databricks_mapping_sharepoint` is removed from the pipeline; `demo_run.py`
   (the only caller that can configure SharePoint anyway) loads the CSV via a
   helper relocated to `chunker.batcher` / `app_config.sharepoint` and passes
   the plain `databricks_mapping` dict. The pipeline loses its only
   `app_config.sharepoint` import.
4. **Split run control from orchestration.** Resume/fork/run-fact/rewind logic
   (`_resume_state`, `_rewind_for_resume`, `fork_run`, `_record_*_fact`,
   `get_run_facts`, `get_validation_facts`) moves to `pipeline/run_ledger.py`
   as a `RunLedger(memory)` collaborator; `engine.py` keeps thin delegating
   methods so the public API is unchanged. `_answer_item`'s validation-retry
   loop stays in the engine (it owns the graph call).

### Phase 4 — token-budgeted call packing (fewer LLM calls)

Builds on Phase 2 (the tiktoken counter) and Phase 3.1 (batch-only path).
Generalize `coalesce_into_batches` from "count-capped singleton runs" to a
single token-budgeted packing pass over the ordered items:

**Mechanics.**

- One walk in corpus order, accumulating a window of adjacent items
  (singletons *and* real batches). The window is emitted as one unit when
  adding the next item would exceed either budget:
  - `max_merged_tokens` — estimated prompt cost of the combined unit;
  - `max_merged_chunks` — total member chunks (kept as a secondary cap so a
    pathological corpus of tiny chunks still bounds prompt complexity).
- A window containing a single real batch passes it through **unchanged**
  (same `batch_id`, same reason); a window containing a single singleton
  chunk keeps today's one-member `merged-NNN` wrap (the "always a SasBatch"
  invariant) — so the no-packing case degenerates to today's behavior.
  A multi-item window becomes a synthetic `packed-NNN` batch whose
  aggregates are recomputed the way `_merge_singletons_into_batch` already
  does (outputs of earlier members consumed by later members become internal;
  external I/O, required macros/macrovars/librefs unioned), with reason
  `"packed N item(s) (~T tokens) to reduce LLM calls"`.
- **Ordering is preserved by construction**: only adjacent items pack, member
  chunks keep corpus order inside the prompt, so producers still precede
  consumers — both across packed units and within one. Packing two batches
  where one depends on the other is not just safe but beneficial: the model
  sees the producing and consuming steps in one context.
- **Exclusions**: the `is_global_context` batch (emitted first, context for
  the whole corpus) stays unpacked — it anchors the thread's history and its
  identity is load-bearing for relevance selection. Oversized-split
  parent/child singletons pack like any other chunk (their text redundancy is
  already intentional).

**Token estimation.** The batcher cannot know the pipeline's prompt templates
(import direction: `chunker` sits below `pipeline`), so the cost function is
injected: `coalesce_into_batches(items, *, max_chunks, max_tokens=None,
item_cost=None)` where `item_cost: Callable[[SasBatch | SasChunk], int]`.

- Default (no injection): tiktoken-free estimate — `chars//4` over member
  texts plus a per-member constant for the metadata/template overhead — so
  the batcher stays dependency-light and offline.
- The pipeline injects a precise counter built on Phase 2's
  `llm_client/tokens.py`: member text tokens + measured per-member and
  per-batch template overheads (constants measured once from the templates,
  not by formatting every candidate window repeatedly).

**Budget resolution** (new constructor arg + config key
`pipeline.max_merged_tokens`):

1. explicit argument, else
2. config.json, else
3. derived from `max_input_tokens` when set: `(max_input_tokens − system
   prompt − guidance budget − history-window headroom) × 0.8`, else
4. a conservative hard default (~6,000 tokens — large enough that today's
   8-member merges of typical chunks still form, small enough to keep answer
   quality per item; tune against the validation suite).

**Consequences to accept and document:**

- *Item identity changes.* `packed-NNN` ids replace some `merged-NNN`/real
  batch ids, so a `resume=True` against a thread written before this change
  (or under a different budget) finds no matching run facts and regenerates —
  same caveat `max_merged_chunks` already carries; packing stays
  deterministic in `(items, budgets)`, so resume within one configuration is
  unaffected.
- *Coarser verdicts and notebooks.* Validation scores per item, so packed
  items get one verdict covering more code; `chunker.notebook` routing is
  unchanged (a packed batch spanning files lands in `_cross_file.ipynb` like
  any cross-file batch, single-file packs render into that file's notebook).
- *Answer-quality tradeoff.* Larger prompts mean the model divides attention
  across more steps per call. Mitigation: the budget default is deliberately
  well under `max_input_tokens`, and the validation suite (response coverage,
  dataset fidelity) is the regression gate — run it before/after on the
  sample corpus and tune the default budget from that.
- *Tests.* Coalesce-output tests that pin `merged-NNN` grouping under the
  default config need updating in the same commit; the batcher's own
  grouping/reason-string tests are untouched.

Rollout: land the mechanism with `max_merged_tokens=None` meaning "packing of
real batches off, singleton merging token-aware only" for one commit, then
flip the derived default on in a separate commit — so a regression bisects to
the flip, not the mechanism.

### Phase 5 — one complete notebook per SAS source file

**Current state.** `chunker.notebook` already writes one `.ipynb` per source
file *for single-source batches*. The gap is multi-file batches: their whole
output lands in a shared `_cross_file.ipynb`, and each participating file's
notebook gets only a pointer cell ("run it there"). So a source file whose
steps were pulled into cross-file batches has an incomplete notebook. Phase 4
makes this worse: packed batches span files more often, so ever more of the
translation would drain into `_cross_file.ipynb`. The requirement — a
corresponding, complete notebook for every converted SAS source — needs the
batch output to be split back per source file.

**Why splitting is currently impossible.** The unit of attribution exists on
the *input* side — every member chunk carries `chunk_id` + `source_id`, and
the batch prompt template already shows both per member — but not on the
*output* side: `TranslationCell` has `kind/source/language/comment` and no
link to the chunk it translates. A cross-file document's cells therefore
cannot be routed to files without guessing.

**Plan: per-cell chunk attribution through the structured schema.**

1. `response_models.TranslationCell` gains `chunk_id: str | None = None`
   (optional — old stored documents and lenient models keep validating), and
   `_STRUCTURED_SYSTEM_PROMPT_TEMPLATE` instructs the model to stamp every
   cell with the member `chunk_id` it implements (the ids are already in the
   batch context block it reads). Single-member batches need no tag.
2. The pipeline output dict gains `chunk_sources: {chunk_id: source_id}`
   (additive; built from the batch members) so the renderer can map a cell's
   `chunk_id` to a notebook without re-deriving anything.
3. `notebooks_from_outputs` routing changes for multi-source items:
   - Cells whose `chunk_id` resolves to a source file go to **that file's
     notebook**, in output order (outputs are already consumed in dependency
     order, so per-file cell order stays dependency-respecting).
   - The item header cell is emitted into every participating notebook
     (annotated with the sibling files), so each reader still sees the
     step's provenance and validation verdict.
   - Document-level Analysis/Mapping/Risks cells go to every participating
     notebook — they are markdown, cheap, and each file's notebook should
     stand alone.
   - **Degradation is all-or-nothing per item**: an item splits only when
     *every* code cell's `chunk_id` resolves; if any code cell is untagged or
     unresolvable — or the item came through the markdown-fallback path,
     which has no attribution — the **whole item** falls back to today's
     behavior (`_cross_file.ipynb` + pointer cells). A per-cell fallback
     would scatter one item's translation across per-file notebooks *and*
     `_cross_file.ipynb`, which is worse than either whole. Nothing gets
     worse than the status quo; `_cross_file.ipynb` becomes the exception
     path rather than the rule, and disappears entirely on a run where every
     cross-file document tags cleanly.
4. Ordering caveat, stated in the notebook docstring: when file A's and
   file B's steps interleave through shared batches, running notebook A top
   to bottom before notebook B is only safe if A's steps don't depend on B's
   later outputs; the existing corpus-order guarantee (producers precede
   consumers *across* the run) is preserved per notebook, and genuinely
   interleaved dependencies keep their pointer/`_cross_file` treatment if
   untagged. The per-file header annotations make the cross-file coupling
   visible either way.

**Verification.** `tests/test_notebook.py` gains: tagged cross-file document
splits per file; untagged falls back to `_cross_file`; mixed tagging routes
what it can; nbformat schema still validates. Prompt-side, the validation
suite gates that asking for `chunk_id` tags doesn't degrade translation
quality; if a target model tags unreliably, the feature degrades per item,
not per run.

Dependency: none on Phases 1–3 (works against today's layout); strongly
recommended to land with or before Phase 4's default flip, so packing does
not visibly regress the per-file deliverable.

Invariants that must survive every phase (Architecture.md §Load-bearing):
no LangGraph checkpointer; ephemeral guidance/summary/notes never persisted;
`chat::` key schema untouched; in-memory mode stays Spark-free. The batcher's
**dependency grouping** — union-find, weak-edge resolution, batch membership,
reason strings, corpus ordering — is untouched by every phase; Phase 4's one
stated exception is the *coalescing* layer above it (`coalesce_into_batches`
and the synthetic merged/packed batch ids and reasons it emits), which was
always call-level packaging, not dependency discovery.

## 3. Sequencing rationale and cross-phase interactions

**Two viable tracks.** The commit sequence below is hygiene-first
(1→2→3→4→5): the package move lands before the feature work so Phases 4–5
edit files in their final location instead of being churned by a later move.
If reducing LLM calls and the per-file notebook deliverable are urgent, a
value-first track works too: Phase 2 → 4 → 5 first (4 needs 2's counter;
5 is independent), deferring 1 and 3 — at the cost of the move later touching
freshly-changed files. Recommendation: hygiene-first unless there is schedule
pressure; Phase 1 is cheap and makes everything after it cleaner to review.

**Interactions to keep in view:**

- *Packing grows the history, not just the prompt* (4 ↔ default trimming).
  `window_k` counts turn *pairs*; packed items make each pair bigger, so the
  default 6-pair window can grow materially in tokens. Phase 4's budget
  derivation accounts for it via the history-headroom term, and Phase 2's
  counter makes `RelevantHistorySelector(max_tokens=...)` a real
  (token-accurate) alternative when window growth becomes a problem — worth
  revisiting the default trimming mode after Phase 4 is measured, not before.
- *Per-call overhead economics changed by the model default* (2 ↔ 4).
  With `gpt-5.4`, Anthropic `cache_control` is inert, but OpenAI-compatible
  endpoints typically apply automatic server-side prompt caching to a stable
  prefix — partially discounting the repeated system prompt. Phase 4's win is
  therefore mostly the *history + guidance* re-send and request latency, not
  the system prompt; the call-reduction estimate should be validated against
  real gateway usage numbers (the `TokenUsage` cache fields already report
  this) rather than assumed.
- *Bigger items stress structured output* (4 ↔ 5). Packed items mean longer
  `TranslationDocument`s with more cells to tag; if tagging reliability drops
  with size, Phase 5's all-or-nothing fallback sends more items to
  `_cross_file.ipynb`, silently undoing the per-file deliverable. This is the
  concrete reason Phase 5 lands *before* Phase 4's default flip and both are
  gated on the validation suite plus a manual notebook-output check on the
  sample corpus.
- *One vocabulary end to end* (2 ↔ 4). The packing cost function, the input
  budget, the summarizer trigger, and history packing must all count with the
  same encoding, or the packing headroom math in Phase 4's budget derivation
  is fiction. Phase 2 is what makes that true; do not land Phase 4 without it.

## 4. Suggested commit sequence

1. `refactor(pipeline): move pipeline/notebook/response models out of chunker into a pipeline package` (Phase 1)
2. `feat(llm_client): tiktoken-backed token counting by default, gpt-5.4 default model` (Phase 2)
3. `refactor(pipeline): batch-only processing path` (Phase 3.1)
4. `refactor(pipeline): grouped constructor configs; SharePoint mapping load moves to demo_run` (Phase 3.2–3.3)
5. `refactor(pipeline): extract RunLedger` (Phase 3.4)
6. `feat(chunker): token-budgeted coalescing of singletons and adjacent small batches` (Phase 4, mechanism)
7. `feat(notebook): route cross-file batch cells to per-source notebooks via chunk_id attribution` (Phase 5)
8. `feat(pipeline): enable token-budgeted call packing by default` (Phase 4, flip — after Phase 5)
