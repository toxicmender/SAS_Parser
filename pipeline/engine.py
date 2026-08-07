"""Glue layer: SAS chunker/batcher -> LangChain chat-memory threads -> LLM.

This package is the sole integration point between the chunker/batcher stack
and the ``memory`` / ``llm_client`` / ``prompt_builder`` packages, so those
stay independently usable. The item → prompt mapping and message formatting
live in :mod:`pipeline.prompting`; this module owns orchestration.

Logger name: ``pipeline.engine``.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence
from typing import Any

import app_config
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda

# Some langgraph releases build their checkpoint serde with langchain-core's
# default `allowed_objects`, which langchain-core >= 1.3.3 answers with a
# LangChainPendingDeprecationWarning ("The default value of `allowed_objects`
# will change in a future version...") at import time. The deserialisation
# happens inside langgraph — no call site in this repo can pass the explicit
# value — so exactly that message is silenced around the import. The locked
# versions pass allowed_objects="core" themselves and emit nothing.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*allowed_objects.*",
        category=PendingDeprecationWarning,
    )
    from langgraph.graph import START, MessagesState, StateGraph

from chunker.batcher import (
    MultiFileBatcher,
    SasChunkBatcher,
    coalesce_into_batches,
)
from chunker.chunker import SasSemanticChunker
from chunker.models import (
    SasBatch,
    SasBatchResult,
    SasChunkResult,
    SasCorpus,
    SasDiagnostic,
)
from llm_client import LLMClient, LLMClientConfig, TokenUsage, tokens
from memory.extractor import MemoryExtractor
from memory.policy import TaskPolicy
from memory.thread_mem import ThreadMemory
from prompt_builder import PromptBuilder, UserInstructionSet
from target_language import TargetLanguage, resolve_target_language

from .constants import (
    _STRUCTURED_SYSTEM_PROMPT_TEMPLATE,
    _SYSTEM_PROMPT_TEMPLATE,
)
from .prompting import (
    _attribution_for_item,
    _constructs_for_item,
    _format_batch_message,
    _kinds_for_item,
    _meta_flags_for_item,
    _query_for_item,
    prompt_cost_estimator,
)
from .response_models import TranslationDocument
from .run_ledger import DOCUMENT_KEY, RunLedger, document_of
from .setup import ChunkingSetup, MemorySetup, PromptingSetup, ValidationSetup

logger = logging.getLogger(__name__)

# Token-budgeted call packing defaults (see _resolve_packing_budget). The
# hard default assumes a modern long-context model (the gpt-5.4 default) and
# packs aggressively; tighten it via pipeline.max_merged_tokens if per-item
# answer quality drops. The headrooms mirror what shares the request with
# the item text — retrieved guidance (prompt_builder's max_instruction_words
# default, ~1.3 tokens/word) and the window_k history.
_DEFAULT_MAX_MERGED_TOKENS = 64_000
_PACKING_GUIDANCE_HEADROOM_TOKENS = 2_000
_PACKING_HISTORY_HEADROOM_TOKENS = 4_000


def _is_anthropic_model(model: str) -> bool:
    """True when *model* resolves to an Anthropic chat model — a bare
    ``claude-*`` name or an explicit LangChain ``anthropic:`` provider
    prefix. Gates provider-specific request features (prompt caching)."""
    return model.startswith("claude") or model.startswith("anthropic:")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SasLLMPipeline:
    """
    End-to-end pipeline: SAS source(s) -> semantic chunks -> dependency
    batches -> LLM responses, with a memory.store-backed chat-memory
    thread per run.

    All batches and singleton chunks produced for a single ``run_file`` /
    ``run_text`` / ``run_files`` call are fed, in dependency-respecting
    corpus order, into **one thread** — so the LLM sees the whole run's
    accumulated context, batch by batch, exactly like a single
    conversation about one migration job. Call with an explicit
    ``thread_id`` to resume or fork that conversation later.

    Every processed item is a :class:`SasBatch` — singletons arrive wrapped
    by ``coalesce_into_batches`` — so an output dict no longer carries an
    ``is_batch`` flag or a ``kind``. Both were constants (``True`` and
    ``None``), and a field that cannot vary tells a reader nothing while
    inviting them to branch on it.

    Parameters
    ----------
    llm_config : LLMClientConfig | None
        Every LLM transport setting, as one
        :class:`llm_client.LLMClientConfig` used as-is: the model, the
        endpoint and its credential, temperature, timeouts, retry budget,
        the input-token budget, the rate limiter, and the request-body
        extras. Its ``model`` is the pipeline's model. ``None`` (default)
        builds ``LLMClientConfig()``, which resolves everything from
        config.json and the environment.
    memory_setup : MemorySetup | None
        All memory wiring, as one :class:`pipeline.setup.MemorySetup`: the
        store hub, the long-term policy, the thread notes, the extractor,
        the chat identity, the Delta/Spark target, and the history policy
        below. Its ``build()`` produces what the pipeline holds. ``None``
        (default) builds ``MemorySetup()`` — an in-memory store, no Spark,
        no JVM.
    chunking : ChunkingSetup | None
        How the source is split and grouped (:class:`pipeline.setup.ChunkingSetup`).
    prompting : PromptingSetup | None
        What the model is asked, and how (:class:`pipeline.setup.PromptingSetup`).
    validation : ValidationSetup | None
        Inline validation and its retry budget
        (:class:`pipeline.setup.ValidationSetup`).
    output_language : str | None
        Target language to translate into. Resolved once, here, through the
        ``target_language`` registry: the name is folded (``"spark sql"``,
        ``"SparkSQL"``, and ``"Spark SQL"`` are one target) and canonicalised,
        and the resolved :class:`~target_language.TargetLanguage` then drives
        the system prompt, the ``[lang: ...]`` instruction axis, the notebook
        kernel and fence tags, and what the validation suite checks the
        emitted code against. An unrecognised name raises
        :class:`~target_language.UnknownTargetLanguage` rather than degrading
        into a Python run that only looks right. ``None`` (default) defers to
        config.json ``pipeline.output_language``, then
        ``target_language.DEFAULT_OUTPUT_LANGUAGE`` (``"SparkSQL"``).
    llm : Any | None
        Pre-built LangChain chat model to use instead of constructing one
        from ``model`` via :class:`llm_client.LLMClient`.  Useful for
        injecting a fake or pre-configured client (e.g. in tests).  The
        retry and input-token-budget layers still wrap an injected model;
        the construction-time knobs (``temperature``, ``base_url``,
        ``api_key``, ``url_headers``, ``gateway_version``, ``timeout``,
        ``model_kwargs``, ``llm_kwargs``, ``requests_per_second``) do not
        apply to it.

    Group fields
    ------------
    There is deliberately **one spelling per knob**: these are fields on the
    five objects above, never keyword arguments here as well. Two places to
    set one thing means a rejection branch to stop them being set in both,
    which is what this constructor used to carry.

    memory_setup.window_k : int | None
        Rolling-window size in (human, AI) turn-pairs kept in context per
        LLM call. ``None`` disables trimming (full history every call).
        Ignored when ``history_selector`` is set.
    memory_setup.history_selector : RelevantHistorySelector | None
        Relevance-based history selection: per LLM call, prompt only the
        turn pairs most relevant to the current batch/chunk message
        (BM25 + optional FAISS dense retrieval, RRF-fused) instead of the
        recency window. ``None`` (default) keeps ``window_k`` behaviour.
    memory_setup.summarizer : RollingSummarizer | None
        Rolling thread summarization (``memory.summarize``): turns older
        than the summarizer's recency tail are folded into one running
        summary per thread, prepended to every prompt as a SystemMessage
        after trimming/selection. Like reference guidance, the summary is
        **prompted but never persisted** to the ``msg::`` history — it
        lives in the KV layer and is re-derivable from the full stored
        thread. A summarizer constructed without a ``store`` is given this
        pipeline's ``memory.kv``. ``None`` (default) disables compression.
    chunking.min_words, max_words : int | None
        Forwarded to :class:`SasSemanticChunker`. ``None`` (default) lets
        the chunker read ``sas_chunker.*`` from config.json (see the
        ``app_config`` package), falling back to 300/700.
    chunking.include_options_chunks, include_comment_chunks : bool
        Forwarded to the batchers.
    chunking.max_merged_chunks : int
        Every LLM call is made per :class:`SasBatch`: before a run, the
        batcher's ordered items are coalesced so each dependency batch stays
        one call and each maximal run of consecutive independent singleton
        chunks is packed into synthetic ``merged-NNN`` batches of at most this
        many members (see :func:`chunker.batcher.coalesce_into_batches`).
        Larger values mean fewer, larger requests; the cap keeps a merged
        prompt from blowing ``max_input_tokens``. Must be ``>= 1`` (``1``
        wraps each singleton as its own batch without merging). Default ``8``.
    chunking.max_merged_tokens : int | None
        Token-budgeted call packing: when set, coalescing accumulates
        *adjacent* items — singletons **and** real dependency batches — into
        one LLM call while the estimated prompt cost stays under this budget
        (and total members under ``max_merged_chunks``), emitting multi-item
        windows as synthetic ``packed-NNN`` batches (see
        :func:`chunker.batcher.coalesce_into_batches`). Cost is estimated
        with this pipeline's tokenizer
        (:func:`pipeline.prompting.prompt_cost_estimator`), so the budget
        and ``max_input_tokens`` count under one vocabulary. The
        global-context batch never packs, and item identity changes versus
        unpacked runs (``packed-NNN`` ids), so resume only matches runs made
        under the same budget. ``None`` (default) defers to config.json
        ``pipeline.max_merged_tokens``, then to a **derived default — packing
        is on by default**: with ``max_input_tokens`` set,
        ``0.8 × (max_input_tokens − system prompt − guidance/history
        headroom)`` (a budget too small to pack disables it); otherwise
        ~64,000 tokens. Pass ``0`` (or set the config key to ``0``) to turn
        packing off and keep the original count-capped singleton merging
        only. Must be ``>= 1``, or ``0`` for off.
    chunking.databricks_mapping : dict[str, str] | None
        SAS→Databricks dataset-name mapping forwarded to the batchers
        (see :func:`chunker.batcher.replace_dataset_names`): batch and
        chunk metadata dataset names — and ``%let`` values holding a
        dataset reference — are rewritten to Unity Catalog
        ``catalog.schema.table`` names before prompting.  Default:
        ``None`` (no renaming).
    prompting.system_prompt : str | None
        Override the default prompt from ``pipeline.constants``.
    prompting.structured_output : bool | None
        Ask the model for a
        :class:`~pipeline.response_models.TranslationDocument` instead of
        free-form Markdown, so the notebook renderer knows exactly which cells
        are runnable code (see ``pipeline.notebook``). ``None`` (default) defers
        to config.json ``pipeline.structured_output``, then ``True``. Either
        way the *persisted* turn and the item's ``response`` are Markdown, so
        conversation memory, resume, and every validation metric are unchanged;
        a gateway that rejects the schema degrades to the model's prose with a
        warning rather than failing the run.
    prompting.prompt_caching : bool | None
        Anthropic prompt caching for the system prompt: when enabled and
        ``model`` is an Anthropic model, the system prompt is sent as a
        content block carrying a ``cache_control`` breakpoint, so every
        item after a run's first reads it from the provider cache (~10%
        of input cost) instead of re-paying full price. ``None``
        (default) defers to config.json ``llm_client.prompt_caching``,
        then ``False``. Prompts shorter than the model's minimum
        cacheable prefix (~1024 tokens on ``claude-sonnet-4-5``; larger
        on newer models) are silently not cached — harmless. Ignored for
        non-Anthropic models.

        Requesting it is safe even against a gateway that does not
        support it. ``cache_control`` is an Anthropic-native content-part
        key, and whether it survives the trip through the gateway's
        OpenAI-compatible API is a property of the gateway, not the
        model. So :class:`llm_client.LLMClient` settles it by asking:
        it sends the breakpoint, and if the endpoint rejects it, strips
        it, re-sends, and drops it from every later call (one WARNING,
        one failed request per process). The run then simply pays full
        price for the system prompt.
    prompting.prompt_builder : PromptBuilder | None
        Reference-PDF guidance source. When set, each item's prompt gains a
        block of instruction chunks relevant to that item's constructs
        (retrieved from the reference corpus). The guidance is **ephemeral**:
        it is prompted but never stored in the thread's history — see the
        load-bearing invariant on this in Architecture.md. ``None`` (default)
        disables guidance injection entirely.
    prompting.user_instructions : str | UserInstructionSet | None
        Operator-supplied project rules (see
        ``prompt_builder/user_instructions.py`` for the heading/directive
        syntax). ``None`` (default) falls back to the standing instructions
        file named by config.json ``user_instructions.path``, when set and
        present. With a ``prompt_builder``, the rules are folded into it
        (replacing any set it already carries — the pipeline-level argument
        wins, with a WARNING); without one, a corpus-less
        :class:`PromptBuilder` is built so instruction injection works with
        no reference PDFs at all. Selected rules render in a
        ``## Project instructions`` block and are ephemeral like all
        guidance: prompted, never persisted.
    validation.validator : Any | None
        Optional inline validator (``validation.live.LiveValidator`` —
        duck-typed, so this package imports nothing from ``validation``,
        which itself imports this one). When set, each item is scored the
        moment its response returns and the verdict is written to this run's
        conversation memory, beside its run fact (see
        :meth:`get_validation_facts`). With ``validation_retries == 0``
        (default) validation is observe-only: a failing or erroring
        validation never retries the item or aborts the run. ``None``
        (default) disables inline validation entirely.
    validation.retries : int
        How many times to *re-generate* an item that fails inline validation
        before accepting its answer (``0``, default, keeps the observe-only
        policy — score and store, never act). Requires a ``validator``.
        On a failing verdict the just-produced turn is rolled back and the
        item is re-prompted with a corrective note naming the metrics that
        fell short (ephemeral, like reference guidance — prompted, never
        persisted), then re-scored; the loop stops as soon as an attempt
        passes or the budget is exhausted, and the final attempt's turn and
        verdict are what persist. This same switch also makes **resume**
        validation-aware: an item whose stored verdict failed no longer
        counts as done, so a resumed run rewinds to the earliest unsatisfied
        item and regenerates from there.
    """

    def __init__(
        self,
        *,
        llm_config: LLMClientConfig | None = None,
        memory_setup: MemorySetup | None = None,
        chunking: ChunkingSetup | None = None,
        prompting: PromptingSetup | None = None,
        validation: ValidationSetup | None = None,
        output_language: str | None = None,
        llm: Any | None = None,
    ) -> None:
        # Each group defaults to its own all-defaults instance, so the
        # no-argument constructor still works and every knob has exactly one
        # place to be set.
        if llm_config is None:
            llm_config = LLMClientConfig()
        if memory_setup is None:
            memory_setup = MemorySetup()
        if chunking is None:
            chunking = ChunkingSetup()
        if prompting is None:
            prompting = PromptingSetup()
        if validation is None:
            validation = ValidationSetup()

        # Unpacked once, here, so the body below reads exactly as it did
        # before the grouping and no use site has to spell out its group.
        window_k = memory_setup.window_k
        history_selector = memory_setup.history_selector
        summarizer = memory_setup.summarizer
        min_words = chunking.min_words
        max_words = chunking.max_words
        include_options_chunks = chunking.include_options_chunks
        include_comment_chunks = chunking.include_comment_chunks
        max_merged_chunks = chunking.max_merged_chunks
        max_merged_tokens = chunking.max_merged_tokens
        databricks_mapping = chunking.databricks_mapping
        system_prompt = prompting.system_prompt
        structured_output = prompting.structured_output
        prompt_caching = prompting.prompt_caching
        prompt_builder = prompting.prompt_builder
        user_instructions = prompting.user_instructions
        validator = validation.validator
        validation_retries = validation.retries

        if validation_retries < 0:
            raise ValueError(
                f"ValidationSetup.retries must be >= 0, got {validation_retries}"
            )
        # Resolved once, before anything reads it: the prompt, the guidance
        # selector, the notebook writer, and the validator all take the target
        # from here, so they cannot each interpret the caller's spelling
        # differently (raises on an unknown name — see the argument docs).
        target = resolve_target_language(output_language)
        model = llm_config.model
        # A validator built for another target fails every item on
        # `language_compliance` — and with validation_retries on, burns the
        # whole retry budget doing it. Cheap to detect, so say so up front
        # rather than letting a run's verdicts explain it item by item.
        validator_target = getattr(validator, "target_language", None)
        if validator_target is not None and validator_target.key != target.key:
            logger.warning(
                f"SasLLMPipeline: the validator scores against "
                f"{validator_target.display_name} but this run translates into "
                f"{target.display_name}; build it with "
                f"LiveValidator(output_language={target.display_name!r}) or "
                f"every item will fail on language"
            )
        if validation_retries and validator is None:
            logger.warning(
                f"SasLLMPipeline: validation_retries={validation_retries} has no "
                "effect without a validator; validation-driven retry/resume "
                "stays disabled"
            )
        if user_instructions is None:
            # A standing instructions file (config.json user_instructions.path)
            # applies whenever no explicit set is passed.
            user_instructions = UserInstructionSet.from_config()
        if user_instructions is not None:
            if prompt_builder is None:
                logger.info(
                    "SasLLMPipeline: no prompt_builder given; building a "
                    "corpus-less PromptBuilder for the user instructions"
                )
                prompt_builder = PromptBuilder(
                    [],
                    user_instructions=user_instructions,
                    output_language=target.display_name,
                )
            else:
                if prompt_builder.user_instructions is not None:
                    logger.warning(
                        "SasLLMPipeline: replacing the PromptBuilder's "
                        "existing user instructions with the pipeline-level "
                        "set (the user_instructions argument wins)"
                    )
                prompt_builder = prompt_builder.with_user_instructions(
                    user_instructions
                )
        if prompt_caching is None:
            prompt_caching = bool(
                app_config.llm_client_value("prompt_caching", False)
            )
        self._prompt_caching = prompt_caching and _is_anthropic_model(model)
        if prompt_caching and not self._prompt_caching:
            logger.warning(
                f"SasLLMPipeline: prompt_caching requested but model "
                f"{model!r} is not an Anthropic model; caching stays off"
            )
        logger.info(
            f"SasLLMPipeline.__init__  model={model}  "
            f"output_language={target.display_name}  "
            f"window_k={window_k}  guidance={'on' if prompt_builder else 'off'}  "
            f"prompt_caching={'on' if self._prompt_caching else 'off'}"
        )
        if max_merged_chunks < 1:
            raise ValueError(
                f"max_merged_chunks must be >= 1, got {max_merged_chunks}"
            )
        if max_merged_tokens is None:
            max_merged_tokens = app_config.get_typed_value(
                "pipeline", "max_merged_tokens", int, None
            )
        if max_merged_tokens is not None and max_merged_tokens < 0:
            raise ValueError(
                f"max_merged_tokens must be >= 1, or 0 to disable packing, "
                f"got {max_merged_tokens}"
            )
        self.model = model
        self.window_k = window_k
        self._max_merged_chunks = max_merged_chunks
        # None here means "derive a default"; the budget is settled below,
        # once the system prompt exists to be measured (0 = packing off).
        self._requested_merged_tokens = max_merged_tokens
        self._target_language = target
        self._output_language = target.display_name
        self._history_selector = history_selector
        self._prompt_builder = prompt_builder
        self._validator = validator
        # Validation-driven retry/resume is active only with both a validator
        # and a positive budget; otherwise validation stays observe-only.
        self._validation_retries = (
            validation_retries if validator is not None else 0
        )

        # SAS→Databricks dataset renaming, forwarded to both batchers, which
        # apply it as a post-pass after grouping. Sourcing one is the caller's
        # job: xref.sourcing.mappings() reads the SharePoint XREF list and
        # load_databricks_mapping_sharepoint() a CSV in the library; both
        # return this shape, so merging them is one line there.
        self.databricks_mapping = databricks_mapping or None

        self.chunker = SasSemanticChunker(min_words=min_words, max_words=max_words)
        self.batcher = SasChunkBatcher(
            include_options_chunks=include_options_chunks,
            databricks_mapping=self.databricks_mapping,
        )
        self.multi_batcher = MultiFileBatcher(
            include_options_chunks=include_options_chunks,
            include_comment_chunks=include_comment_chunks,
            databricks_mapping=self.databricks_mapping,
        )

        if structured_output is None:
            structured_output = bool(
                app_config.get_typed_value(
                    "pipeline", "structured_output", bool, True
                )
            )
        self._requested_structured_output = structured_output

        # All memory wiring — hub default, store injection, extractor
        # implications, chat identity — lives in MemorySetup.build(). One
        # pipeline instance is one *chat*: the span of a thread this object
        # writes (Task -> Thread -> Chat -> Message); the id is per-instance,
        # so resuming a thread from a new pipeline opens a new chat on the
        # same thread — exactly the boundary the policy snapshot below is
        # taken at.
        built = memory_setup.build()
        self._memory = built.hub
        self._ledger = RunLedger(
            self._memory, validation_retries=self._validation_retries
        )
        self.chat_id = built.chat_id
        self.task_id = built.task_id
        self._task_policy = built.task_policy
        self._thread_memory = built.thread_memory
        self._memory_context = built.context
        self._memory_extractor = built.extractor

        self._summarizer = summarizer
        if summarizer is not None and summarizer.store is None:
            # Summaries persist beside the thread they compress, so
            # snapshot()/restore() carry them along with the history.
            summarizer.store = self._memory.kv

        # llm_client owns construction (temperature, endpoint overrides,
        # output cap, rate limiter) and invocation (transient-error retry,
        # input-token budget). An injected chat model replaces only the
        # construction half; retry and budget still apply.
        self._llm_client = LLMClient(llm_config, llm=llm)

        # Which system prompt to send depends on whether this chat model can
        # actually be asked for a schema, so the prompt is settled only now
        # that the client exists. A model that cannot (an injected stub, an
        # older integration) falls back to prompting for Markdown, which the
        # notebook renderer parses — the run proceeds either way.
        self._structured_output = (
            self._requested_structured_output
            and self._llm_client.supports_structured_output(TranslationDocument)
        )
        if self._requested_structured_output and not self._structured_output:
            logger.warning(
                f"SasLLMPipeline: structured output requested but model "
                f"{model!r} does not support it; prompting for Markdown instead"
            )
        template = (
            _STRUCTURED_SYSTEM_PROMPT_TEMPLATE
            if self._structured_output
            else _SYSTEM_PROMPT_TEMPLATE
        )
        # Both templates are filled with every target fact; each uses the
        # subset it needs (the Markdown one names the fence tag, the
        # structured one the cell `language` value), and .format() ignores the
        # rest.
        self._system_prompt = system_prompt or template.format(
            output_language=target.display_name,
            fence_info=target.default_fence,
            cell_language=target.cell_language,
            comment_prefix=target.comment_prefix,
        )
        # The long-term task policy rides INSIDE the cached system block: it
        # is the same text for every thread and every item of the run, so it
        # costs one cache write and is then served from cache. Short-term
        # thread notes deliberately do not (they change per thread and would
        # miss the cache on each one) — they are prompted after the
        # breakpoint, through the ephemeral `instructions` channel.
        # Snapshotted here, at construction: a policy edited mid-run would
        # invalidate the cached prefix on the next item.
        policy_block = self._memory_context.system_suffix
        # Snapshotted with the text, not read live: the fingerprint has to
        # describe what this pipeline actually prompted, and the policy object
        # may be edited afterwards (by an operator, or by an approved
        # extraction) without this run's prompts changing.
        self._policy_fingerprint = self._memory_context.policy_fingerprint
        if policy_block:
            self._system_prompt = f"{self._system_prompt}\n\n{policy_block}"
            logger.info(
                f"SasLLMPipeline: task policy folded into the system prompt  "
                f"task='{self.task_id}'  "
                f"fingerprint={self._memory_context.policy_fingerprint}"
            )
        # Token-budgeted call packing is settled only now: the derived
        # default needs the *final* system prompt (policy included) measured
        # under this model's encoding, and the client's resolved
        # max_input_tokens. The cost function is built once per pipeline and
        # counts with the same encoding the input budget uses
        # (llm_client.tokens), so the two budgets are commensurable.
        self._max_merged_tokens = self._resolve_packing_budget(
            self._requested_merged_tokens,
            self._llm_client.config.max_input_tokens,
            self._system_prompt,
            model,
        )
        self._item_cost = (
            prompt_cost_estimator(model)
            if self._max_merged_tokens is not None
            else None
        )
        if self._max_merged_tokens is not None:
            logger.info(
                f"SasLLMPipeline: token-budgeted call packing on  "
                f"max_merged_tokens={self._max_merged_tokens}"
            )
        # With prompt caching on, the system prompt travels as a content
        # block with a cache_control breakpoint (a concrete SystemMessage,
        # exempt from template interpolation); Anthropic then serves the
        # prompt prefix up to that block from cache on every item after the
        # run's first. The breakpoint sits on the system block — history
        # varies per item under trimming/selection, so it is the one
        # stable prefix.
        if self._prompt_caching:
            system_entry: Any = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": self._system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
        else:
            system_entry = ("system", self._system_prompt)
        prompt = ChatPromptTemplate.from_messages(
            [
                system_entry,
                MessagesPlaceholder("history"),
                # Ephemeral per-item reference guidance: 0 or 1 SystemMessage,
                # prompted but never persisted (see _call_model).
                MessagesPlaceholder("instructions"),
                ("human", "{input}"),
            ]
        )
        # Kept on the instance for introspection (tests assert the
        # cache_control block; the chain below captures it by closure).
        self._prompt = prompt

        def _trim(inputs: dict[str, Any]) -> dict[str, Any]:
            history: list[BaseMessage] = inputs.get("history", [])
            instructions = inputs.get("instructions", [])
            # The rolling summary (if any) is prepended AFTER trimming or
            # selection: it is not a turn, must never be dropped by the
            # window, and must not participate in relevance scoring.
            summary = inputs.get("summary")
            prefix: list[BaseMessage] = [summary] if summary is not None else []
            if self._history_selector is not None:
                selected = self._history_selector.select(history, inputs["input"])
                logger.debug(
                    f"_trim: relevance selector kept {len(selected)}/{len(history)} message(s)"
                )
                return {
                    "input": inputs["input"],
                    "history": prefix + selected,
                    "instructions": instructions,
                }
            k = self.window_k
            if k is not None and len(history) > k * 2:
                dropped = len(history) - k * 2
                logger.debug(
                    f"_trim: dropping {dropped} old message(s), window_k={k}"
                )
                history = history[-(k * 2) :]
            return {
                "input": inputs["input"],
                "history": prefix + history,
                "instructions": instructions,
            }

        prompt_chain = RunnableLambda(_trim) | prompt
        chain = prompt_chain | self._llm_client.as_runnable()
        # Structured mode asks for a TranslationDocument; the envelope
        # (raw/parsed/parsing_error) is unpacked in _structured_response so a
        # schema the gateway will not honour degrades to the prose in `raw`
        # instead of failing the run.
        structured_chain = (
            prompt_chain
            | self._llm_client.as_structured_runnable(TranslationDocument)
            if self._structured_output
            else None
        )

        def _call_model(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, list[BaseMessage]]:
            # One graph invocation == one conversational turn: only the LAST
            # state message is prompted, and exactly that message plus the
            # response is persisted (the store never records an unshown message).
            # Reference guidance is prompted via `instructions` but is NOT part
            # of the persisted turn — it is re-derivable, would bloat the store,
            # and would pollute relevance-based history selection.
            # "configurable" is NotRequired on RunnableConfig; the graph is
            # always invoked with one, and thread_id below is genuinely required.
            configurable = config.get("configurable", {})
            thread_id = configurable["thread_id"]
            instructions = configurable.get("instructions", [])
            # Short-term thread notes join the same ephemeral channel as the
            # reference guidance: prompted after the cache breakpoint, never
            # persisted to the msg:: history, never scored by the relevance
            # selector. They go last so they read as the most local
            # instruction the model was given.
            instructions = [
                *instructions,
                *self._memory_context.thread_messages(thread_id),
            ]
            history = self._memory.get_thread(thread_id, chat_id=self.chat_id)
            input_message = state["messages"][-1]
            history_messages = history.messages
            # The rolling summary is ephemeral like the guidance: prompted
            # (prepended in _trim), never persisted to the msg:: history.
            summary = (
                self._summarizer.refresh(thread_id, history_messages)
                if self._summarizer is not None
                else None
            )
            payload = {
                "input": input_message.content,
                "history": history_messages,
                "instructions": instructions,
                "summary": summary,
            }
            if structured_chain is not None and self._structured_output:
                try:
                    response = self._structured_response(
                        structured_chain.invoke(payload), thread_id
                    )
                except Exception:
                    # The gateway rejected the schema request itself (as
                    # opposed to answering it badly, which arrives as
                    # parsing_error). Demote once, for the rest of the run, and
                    # re-send unstructured — the same one-shot degradation
                    # llm_client applies to a refused cache_control breakpoint.
                    logger.warning(
                        f"_call_model: structured request failed  "
                        f"thread='{thread_id}'; disabling structured output for "
                        "this pipeline and re-sending as Markdown",
                        exc_info=True,
                    )
                    self._structured_output = False
                    response = chain.invoke(payload)
            else:
                response = chain.invoke(payload)
            history.add_messages([input_message, response])
            return {"messages": [response]}

        # Compiled WITHOUT a checkpointer on purpose: durable per-thread
        # persistence lives in the KV-backed chat history above, keeping the
        # msg:: row schema canonical instead of duplicating state in blobs.
        logger.debug("SasLLMPipeline: compiling LangGraph state graph")
        builder = StateGraph(MessagesState)
        builder.add_node("model", _call_model)
        builder.add_edge(START, "model")
        self._graph = builder.compile()
        logger.debug("SasLLMPipeline: ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_file(
        self, path: str, *, thread_id: str | None = None, resume: bool = False
    ) -> list[dict[str, Any]]:
        """Chunk + batch the SAS file at *path*, run every item through the LLM.

        With ``resume=True``, items whose run fact already reads ``ok`` on
        this thread are skipped (their stored responses — and any inline
        validation verdict recorded for them — are returned), so a crashed
        run picks up where it stopped instead of replaying — and
        re-appending — completed turns.
        """
        logger.info(f"run_file: '{path}'  resume={resume}")
        result = self.chunker.chunk_file(path)
        batch_result = self.batcher.batch(result)
        tid = thread_id or self._default_thread_id([result.source_id or path])
        return self._process(
            self._items_as_batches(batch_result),
            result.diagnostics,
            thread_id=tid,
            resume=resume,
        )

    def run_text(
        self,
        source: str,
        *,
        source_id: str | None = None,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        """Chunk + batch the SAS *source* string, run every item through the LLM.

        ``resume`` behaves as in :meth:`run_file`.
        """
        label = source_id or "<inline>"
        logger.info(f"run_text: source_id='{label}'  chars={len(source)}  resume={resume}")
        result = self.chunker.chunk_text(source, source_id=source_id)
        batch_result = self.batcher.batch(result)
        tid = thread_id or self._default_thread_id([result.source_id or label])
        return self._process(
            self._items_as_batches(batch_result),
            result.diagnostics,
            thread_id=tid,
            resume=resume,
        )

    def run_files(
        self,
        paths: list[str],
        *,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Chunk every file in *paths*, resolve cross-file dependency batches
        via :class:`MultiFileBatcher`, and run every batch/singleton
        through the LLM on **one shared thread** for the whole corpus.

        ``resume`` behaves as in :meth:`run_file`.
        """
        logger.info(f"run_files: {len(paths)} file(s)  resume={resume}")
        return self._run_corpus(
            [self.chunker.chunk_file(p) for p in paths],
            thread_id=thread_id,
            resume=resume,
        )

    def run_texts(
        self,
        sources: list[tuple[str, str]],
        *,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        """
        :meth:`run_files` for a corpus that is already in memory: *sources* is
        a list of ``(source_id, text)`` pairs.

        The same corpus-wide batching, on one shared thread — this differs from
        :meth:`run_files` only in where the text came from. It exists because a
        remotely hosted corpus has no local paths: ``conversion.sources`` hands
        back a drive-relative path and the file's text, and that path is the
        *source id* the reports, notebooks and run facts are named by. Staging
        the corpus to a temporary directory just to have paths to pass would
        name every source after a directory that no longer exists.

        ``resume`` behaves as in :meth:`run_file`.
        """
        logger.info(f"run_texts: {len(sources)} source(s)  resume={resume}")
        return self._run_corpus(
            [
                self.chunker.chunk_text(text, source_id=source_id)
                for source_id, text in sources
            ],
            thread_id=thread_id,
            resume=resume,
        )

    def _run_corpus(
        self,
        file_results: list[SasChunkResult],
        *,
        thread_id: str | None,
        resume: bool,
    ) -> list[dict[str, Any]]:
        """Batch an already-chunked corpus and run every item on one thread.

        The shared tail of :meth:`run_files` and :meth:`run_texts`: the two
        differ only in how they produced *file_results*, and cross-file edge
        resolution must not be able to drift between them.
        """
        corpus = SasCorpus(file_results=file_results)
        multi_result = self.multi_batcher.batch(corpus)
        tid = thread_id or self._default_thread_id(corpus.source_ids)
        return self._process(
            self._items_as_batches(multi_result),
            corpus.all_diagnostics,
            thread_id=tid,
            resume=resume,
        )

    def fork_run(
        self,
        src_thread_id: str,
        dst_thread_id: str,
        *,
        upto_items: int | None = None,
    ) -> int:
        """Fork a run's conversation at item boundary *upto_items*.

        Copies the first ``upto_items`` (human, AI) turn pairs of
        *src_thread_id* — every pair when ``None`` — plus their ``ok`` run
        facts onto the empty thread *dst_thread_id*. Rerunning the same
        source with ``thread_id=dst_thread_id, resume=True`` then skips
        the copied items and continues from item ``upto_items + 1`` on the
        forked history: rewind, edit, re-run — without a checkpointer.
        Returns the number of messages copied.
        """
        copied = self._ledger.fork(
            src_thread_id, dst_thread_id, upto_items=upto_items
        )
        # Short-term notes travel with the branch: an exception the source
        # conversation was granted still holds on a rewind of it, and losing
        # it would silently change what the fork is allowed to do. (The
        # rolling summary is not copied — it is re-derivable, and
        # RollingSummarizer rebuilds it for the shortened thread.)
        notes_copied = (
            self._thread_memory.fork(src_thread_id, dst_thread_id)
            if self._thread_memory is not None
            else 0
        )
        logger.info(
            f"fork_run: '{src_thread_id}' -> '{dst_thread_id}'  "
            f"upto_items={upto_items}  messages_copied={copied}  "
            f"notes_copied={notes_copied}"
        )
        return copied

    def get_thread_messages(self, thread_id: str) -> list[BaseMessage]:
        """Return the raw message history stored for *thread_id*."""
        logger.debug(f"get_thread_messages: thread_id='{thread_id}'")
        msgs = self._memory.get_thread(thread_id).messages
        logger.debug(
            f"get_thread_messages: thread_id='{thread_id}'  messages={len(msgs)}"
        )
        return msgs

    # ---- Memory surfaces ---------------------------------------------------

    def get_chats(self, thread_id: str) -> list[dict[str, Any]]:
        """The chats recorded on *thread_id*, oldest first.

        One chat is one pipeline instance's span of the thread, so this is
        the run-level view of a resumed or forked conversation: which
        instance wrote which stretch of messages, and when.
        """
        return self._memory.chats(thread_id)

    def remember(
        self,
        thread_id: str,
        text: str,
        *,
        kind: str = "note",
        source: str | None = None,
        ttl_s: float | None = None,
    ) -> Any:
        """Add a short-term note to *thread_id* (see ``memory.thread_mem``).

        Raises ``RuntimeError`` when the pipeline was built without a
        ``thread_memory`` — silently dropping an instruction an operator
        just gave is the worse failure.
        """
        if self._thread_memory is None:
            raise RuntimeError(
                "SasLLMPipeline has no thread_memory; construct it with "
                "thread_memory=ThreadMemory() to record conversation-scoped "
                "instructions"
            )
        return self._thread_memory.add(
            thread_id, text, kind=kind, source=source, ttl_s=ttl_s
        )

    @property
    def target_language(self) -> TargetLanguage:
        """The resolved target this run translates into.

        Callers that need the target *after* construction — writing notebooks
        (``pipeline.notebook``), building a validator, picking a complexity
        profile — should read it from here rather than re-resolving the string
        they passed in, so every stage agrees on one object.
        """
        return self._target_language

    @property
    def output_language(self) -> str:
        """The resolved target's canonical name (``"Spark SQL"``, ...)."""
        return self._output_language

    @property
    def task_policy(self) -> TaskPolicy | None:
        """The long-term policy this pipeline prompted, if any.

        Editing it does **not** change the running pipeline: the policy text
        is snapshotted into the (cached) system prompt at construction. Build
        a new pipeline — a new chat — to pick up an edit.
        """
        return self._task_policy

    @property
    def thread_memory(self) -> ThreadMemory | None:
        """The short-term note store this pipeline reads each turn, if any."""
        return self._thread_memory

    @property
    def memory_extractor(self) -> MemoryExtractor | None:
        """The extractor observing accepted turns, if any."""
        return self._memory_extractor

    @property
    def policy_fingerprint(self) -> str | None:
        """Content hash of the prompted task policy, ``None`` without one.

        The policy counterpart of :meth:`instructions_fingerprint`: recorded
        alongside validation results so runs under different standing
        instructions are never compared as equals. Fixed at construction
        along with the prompt it describes — a policy edited mid-life does
        not change it, because it did not change what was prompted.
        """
        return self._policy_fingerprint or None

    def snapshot(self) -> dict[str, Any]:
        """
        Export the entire persistence-layer store (all threads + kv).
        Delegates straight to :meth:`MemoryHub.snapshot` — pipeline
        does not re-implement export logic that memory.store already owns.
        """
        logger.info("snapshot: delegating to MemoryHub")
        return self._memory.snapshot()

    def get_run_facts(self, thread_id: str) -> list[dict[str, Any]]:
        """Per-item outcome records written during a run, in item order.

        One record per processed item, stored in the KV layer under
        ``run::{thread_id}::item::{item_id}`` as the run progresses —
        durable evidence of *which items completed* (status, index,
        timing) that later batches, later runs, and resume query without
        replaying the message history. Delegates to :class:`RunLedger`.
        """
        return self._ledger.run_facts(thread_id)

    def get_validation_facts(self, thread_id: str) -> list[dict[str, Any]]:
        """Per-item inline-validation verdicts recorded for *thread_id*.

        Present only when the pipeline was built with a ``validator``: each
        item scored during the run leaves one record under
        ``validation::{thread_id}::item::{item_id}`` (score, passed, per-metric
        results, index/total), stored beside the run facts by
        ``validation.live.LiveValidator``. Ordered by item index; empty when
        no validator ran on the thread. Delegates to :class:`RunLedger`.
        """
        return self._ledger.validation_facts(thread_id)

    @property
    def token_usage(self) -> TokenUsage:
        """
        Tokens billed by this pipeline's LLM calls so far.

        Cumulative over the pipeline's lifetime, not per ``run_*`` call — a
        caller attributing one run's cost snapshots this before and subtracts
        after (``TokenUsage`` supports ``-``). Stays at zero when the gateway
        reports no usage block; see :attr:`llm_client.LLMClient.usage`.
        """
        return self._llm_client.usage

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def instructions_fingerprint(self) -> str | None:
        """
        Content fingerprint of the active user-instruction set, or ``None``
        when no instructions are active. Recorded into validation run history
        so eval runs with different instructions are never compared as equals.
        """
        builder = self._prompt_builder
        if builder is None or builder.user_instructions is None:
            return None
        return builder.user_instructions.fingerprint

    @staticmethod
    def _resolve_packing_budget(
        requested: int | None,
        max_input_tokens: int | None,
        system_prompt: str,
        model: str,
    ) -> int | None:
        """The packing token budget actually in force, or ``None`` for off.

        Precedence (the repo-wide rule): explicit argument, else config.json
        (both already folded into *requested*; ``0`` means "packing off"),
        else a derived default — with ``max_input_tokens`` set,
        ``0.8 × (max_input_tokens − system prompt − guidance headroom −
        history headroom)`` so a packed prompt plus its fixed companions
        stays inside the input budget (derived ≤ 0 disables packing: a tiny
        input budget leaves no room to pack); without one, a conservative
        ~64k-token default.
        """
        if requested is not None:
            return requested or None  # 0 = packing off
        if max_input_tokens is None:
            return _DEFAULT_MAX_MERGED_TOKENS
        system_tokens = tokens.count_text(system_prompt, model=model)
        derived = int(
            0.8
            * (
                max_input_tokens
                - system_tokens
                - _PACKING_GUIDANCE_HEADROOM_TOKENS
                - _PACKING_HISTORY_HEADROOM_TOKENS
            )
        )
        if derived < 1:
            logger.info(
                f"_resolve_packing_budget: max_input_tokens={max_input_tokens} "
                f"leaves no packing headroom (system={system_tokens}); "
                f"packing stays off"
            )
            return None
        return derived

    @staticmethod
    def _default_thread_id(source_ids: list[str]) -> str:
        return "run::" + "+".join(source_ids)

    def _items_as_batches(self, batch_result: SasBatchResult) -> list[SasBatch]:
        """The run's ordered items coalesced to :class:`SasBatch` only.

        Keeps the LLM invoked strictly per batch: dependency batches pass
        through and consecutive independent singletons are merged into
        ``merged-NNN`` batches (capped at ``max_merged_chunks`` members).
        With ``max_merged_tokens`` set, adjacent items — small dependency
        batches included — additionally pack into ``packed-NNN`` batches
        under that token budget. See
        :func:`chunker.batcher.coalesce_into_batches`.
        """
        return coalesce_into_batches(
            batch_result.all_ordered_items,
            max_chunks=self._max_merged_chunks,
            max_tokens=self._max_merged_tokens,
            item_cost=self._item_cost,
        )

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def _structured_response(
        self, envelope: dict[str, Any], thread_id: str
    ) -> AIMessage:
        """Turn a structured-output envelope into the AI message to persist.

        The persisted content is *rendered Markdown*, not the raw model output:
        under ``function_calling`` the raw content is empty (the answer rode in
        a tool call), and storing an empty AI turn would break the resume path
        (:meth:`RunLedger.recovered_response`), relevance-based history
        selection, and
        every validation metric. The document itself is carried alongside on
        ``additional_kwargs`` so a notebook can be rebuilt without re-prompting.

        A schema the gateway would not honour arrives here as ``parsed=None``;
        that degrades to the prose in ``raw`` rather than failing the run.
        """
        raw = envelope.get("raw")
        parsed = envelope.get("parsed")
        error = envelope.get("parsing_error")
        if parsed is None:
            if raw is None:
                raise ValueError(
                    "structured output returned neither a parsed document nor a "
                    f"raw message (parsing_error={error!r})"
                )
            logger.warning(
                f"_structured_response: no parsed document  thread='{thread_id}'  "
                f"parsing_error={error!r}; falling back to the raw response, "
                "whose notebook will be built by parsing its Markdown"
            )
            return raw if isinstance(raw, AIMessage) else AIMessage(str(raw))

        document: TranslationDocument = parsed
        message = AIMessage(
            content=document.to_markdown(self._target_language.default_fence),
            additional_kwargs={DOCUMENT_KEY: document.model_dump()},
        )
        if isinstance(raw, AIMessage):
            # Keep the accounting the gateway reported: usage drives
            # LLMClient.usage and the run's token totals, and id/metadata are
            # what a stored turn is traced by.
            message.id = raw.id
            message.response_metadata = raw.response_metadata
            message.usage_metadata = raw.usage_metadata
        logger.debug(
            f"_structured_response: parsed document  thread='{thread_id}'  "
            f"cells={len(document.cells)}  risks={len(document.risks)}"
        )
        return message

    def _instruction_messages(
        self, item: SasBatch
    ) -> tuple[list[BaseMessage], list[str]]:
        """Ephemeral reference guidance for *item*, as both artefacts of one
        retrieval: ``(messages, retrieval_context)``.

        *messages* is the 0-or-1 ``SystemMessage`` that gets prompted (never
        persisted); *retrieval_context* is the text of the chunks that were
        retrieved, **in the selector's priority order**, which the validation
        layer scores as this item's retrieval context (see
        ``validation.rag_metrics``). Both are empty when no prompt builder is
        attached or nothing was relevant.

        The two come from a single :meth:`PromptBuilder.select` call — scoring
        must see exactly the context the model saw, and re-retrieving later
        would neither be free nor guaranteed to agree.
        """
        if self._prompt_builder is None:
            return [], []
        constructs = _constructs_for_item(item)
        picks = self._prompt_builder.select(
            _query_for_item(item),
            constructs,
            output_language=self._output_language,
            kinds=_kinds_for_item(item),
            meta_flags=_meta_flags_for_item(item),
        )
        retrieval_context = [pick.chunk.text for pick in picks]
        # Label each section with the member that pulled it in — but only when
        # there is more than one member to tell apart. On a singleton every
        # label would name the same chunk, which is noise, so pass None and
        # render exactly what an unattributed build does.
        attribution = (
            _attribution_for_item(item) if len(item.chunks) > 1 else None
        )
        guidance = self._prompt_builder.build_from_picks(
            picks, constructs, attribution=attribution
        )
        if not guidance:
            return [], retrieval_context
        item_id = item.batch_id
        logger.debug(
            f"_instruction_messages: item={item_id}  guidance_chars={len(guidance)}"
            f"  retrieved={len(retrieval_context)}"
        )
        return [SystemMessage(guidance)], retrieval_context

    @staticmethod
    def _validation_feedback_message(result: Any) -> SystemMessage:
        """A corrective note naming the metrics an attempt failed.

        Injected — ephemerally, like reference guidance — before a retry so
        the model revises rather than repeats. Lists only the metrics that
        were scored and fell below threshold (skipped/passing ones carry no
        signal), with each metric's own ``details`` string.
        """
        failed = [m for m in result.metrics if not m.passed and not m.skipped]
        lines = [
            "## Automated validation of your previous answer FAILED",
            f"Overall score {result.score:.2f}. Revise the translation to fix "
            "the issues below, preserving everything that was already correct:",
        ]
        for m in failed:
            detail = m.details.strip() if m.details else "below threshold"
            lines.append(
                f"- **{m.metric}** (score {m.score:.2f} < {m.threshold:.2f}): {detail}"
            )
        return SystemMessage("\n".join(lines))

    def _answer_item(
        self,
        item: SasBatch,
        idx: int,
        total: int,
        *,
        thread_id: str,
        user_msg: str,
        base_instructions: list[BaseMessage],
        retrieval_context: Sequence[str] = (),
    ) -> tuple[str, dict[str, Any] | None, Any, int]:
        """Generate (and, if enabled, iteratively repair) one item's answer.

        Sends *user_msg* on *thread_id*; when a validator is attached the
        response is scored inline, with *user_msg* and *retrieval_context*
        handed over as the item's ``input`` / retrieved context so the
        judged metrics (``validation.rag_metrics``) have something to score
        against. With ``validation_retries > 0`` a failing
        verdict rolls the just-appended turn back off the thread (via
        :meth:`KVChatMessageHistory.truncate_to`) and re-prompts with a
        corrective note, up to the retry budget, so exactly one — the final —
        (human, AI) pair persists per item. Returns
        ``(response_text, document, CaseResult | None, attempts)``, where
        *document* is the structured
        :class:`~pipeline.response_models.TranslationDocument` dump when
        structured output produced one and ``None`` otherwise; the
        ``CaseResult`` is ``None`` when no validator ran or scoring raised
        (swallowed, as in the observe-only policy). Any LLM-call exception
        propagates to the caller, which records the error fact.
        """
        item_id = item.batch_id
        history = self._memory.get_thread(thread_id)
        max_attempts = 1 + self._validation_retries
        feedback: list[BaseMessage] = []
        attempt = 0
        while True:
            attempt += 1
            # Roll-back point: everything already committed for earlier items.
            # Only needed when a retry might follow (else skip the history load).
            len_before = history.message_count() if max_attempts > 1 else 0
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "instructions": base_instructions + feedback,
                }
            }
            state = self._graph.invoke({"messages": [HumanMessage(user_msg)]}, cfg)
            ai_message = state["messages"][-1]
            ai_text = ai_message.content
            document = document_of(ai_message)

            result: Any = None
            if self._validator is not None:
                try:
                    result = self._validator.validate_item(
                        item,
                        ai_text,
                        thread_id=thread_id,
                        kv=self._memory.kv,
                        index=idx,
                        total=total,
                        prompt=user_msg,
                        retrieval_context=retrieval_context,
                    )
                except Exception:
                    logger.warning(
                        f"_answer_item: inline validation failed  item={item_id}  "
                        f"thread={thread_id}",
                        exc_info=True,
                    )

            passed = result.passed if result is not None else True
            if passed or attempt >= max_attempts:
                if not passed:
                    logger.warning(
                        f"_answer_item: item={item_id} still failing after "
                        f"{attempt} attempt(s); accepting last answer  "
                        f"thread={thread_id}"
                    )
                self._extract_memories(thread_id, user_msg, str(ai_text))
                return ai_text, document, result, attempt

            # Reached only when `passed` is False, which needs a validator to
            # have produced a verdict — so `result` is set here. Spelled out
            # rather than asserted because the guard is two branches away.
            score = f"{result.score:.3f}" if result is not None else "n/a"
            logger.info(
                f"_answer_item: item={item_id} failed validation "
                f"(score={score}) on attempt {attempt}/{max_attempts}; "
                f"rolling back and retrying  thread={thread_id}"
            )
            # Drop this attempt's turn pair so the retry replaces it in place.
            history.truncate_to(len_before)
            feedback = [self._validation_feedback_message(result)]

    def _extract_memories(
        self, thread_id: str, user_msg: str, response_text: str
    ) -> None:
        """Route one accepted turn through the memory extractor, if any.

        Called on the attempt that is *kept*, never on one inline validation
        rolled back — a discarded answer must not leave a memory behind.
        Swallows everything: a memory write cannot be allowed to fail a
        translation run.
        """
        if self._memory_extractor is None:
            return
        try:
            self._memory_extractor.observe(thread_id, user_msg, response_text)
        except Exception:
            logger.warning(
                f"_extract_memories: extraction failed  thread='{thread_id}'",
                exc_info=True,
            )

    def _process(
        self,
        items: Sequence[SasBatch],
        diagnostics: list[SasDiagnostic],
        *,
        thread_id: str,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        total = len(items)
        if not items:
            logger.warning(f"_process: nothing to process  thread='{thread_id}'")
            return []

        # Resume: items whose run fact reads "ok" are skipped; their stored
        # responses (and inline verdicts) are recovered from the thread's
        # (human, AI) turn pairs. Error facts do NOT skip — a failed item is
        # reprocessed and its fact overwritten. With validation-driven retry
        # active, an ok-but-failing item is redone too (see
        # RunLedger.resume_state).
        completed: dict[str, dict[str, Any]] = {}
        completed_validations: dict[str, dict[str, Any]] = {}
        recovered: list[BaseMessage] = []
        if resume:
            completed, completed_validations, recovered = (
                self._ledger.resume_state(items, thread_id)
            )

        logger.info(
            f"_process: invoking LLM for {total} item(s)  thread='{thread_id}'  model={self.model}"
        )
        t_pipeline = time.perf_counter()
        outputs: list[dict[str, Any]] = []

        for idx, item in enumerate(items, start=1):
            item_id = item.batch_id

            user_msg = _format_batch_message(item, idx, total, diagnostics)
            # Per-item guidance rides in the config, not the state, so it is
            # prompted without ever entering the persisted message history.
            # Both are derived above the resume check on purpose: a skipped
            # item's output must still carry the prompt and retrieval context
            # the validation layer scores against, and neither costs an LLM
            # call — which is what the skip is there to avoid.
            base_instructions, retrieval_context = self._instruction_messages(item)

            fact = completed.get(item_id)
            if fact is not None:
                logger.info(
                    f"_process: item {idx}/{total}  id={item_id}  already "
                    f"complete; skipping  thread={thread_id}"
                )
                outputs.append(
                    {
                        "item_id": item_id,
                        "chunk_ids": item.chunk_ids,
                        "chunk_sources": {
                            c.chunk_id: c.source_id or "<inline>"
                            for c in item.chunks
                        },
                        "source_files": item.source_files,
                        "prompt": user_msg,
                        "retrieval_context": retrieval_context,
                        "response": RunLedger.recovered_response(recovered, fact),
                        "document": RunLedger.recovered_document(recovered, fact),
                        "thread_id": thread_id,
                        "skipped": True,
                        "validation": RunLedger.recovered_validation(
                            completed_validations.get(item_id)
                        ),
                    }
                )
                continue

            logger.info(
                f"_process: item {idx}/{total}  id={item_id}  "
                f"members={len(item.chunks)}  thread={thread_id}"
            )

            t_item = time.perf_counter()
            try:
                # Generate — and, when validation_retries > 0, iteratively
                # repair — the answer. Scoring, the retry loop, and the
                # roll-back of superseded attempts all live in _answer_item;
                # exactly one (human, AI) pair persists per item.
                ai_text, document, result, attempts = self._answer_item(
                    item,
                    idx,
                    total,
                    thread_id=thread_id,
                    user_msg=user_msg,
                    base_instructions=base_instructions,
                    retrieval_context=retrieval_context,
                )
            except Exception as exc:
                logger.error(
                    f"_process: LLM call failed  item={item_id}  thread={thread_id}",
                    exc_info=True,
                )
                self._ledger.record_run_fact(
                    thread_id, item_id, self._ledger.error_fact(idx, total, exc)
                )
                raise

            elapsed = time.perf_counter() - t_item
            self._ledger.record_run_fact(
                thread_id,
                item_id,
                self._ledger.ok_fact(
                    idx,
                    total,
                    elapsed_s=elapsed,
                    response_chars=len(ai_text),
                    attempts=attempts,
                ),
            )
            logger.info(
                f"_process: item {item_id} done  elapsed={elapsed:.3f}s  "
                f"attempts={attempts}  response_chars={len(ai_text)}"
            )
            logger.debug(
                f"_process: item {item_id} response preview: "
                f"{ai_text[:120].replace(chr(10), chr(0x21B5))!r}"
            )

            # The verdict is already stored in this thread's memory by
            # _answer_item (beside the run fact); attach it to the output.
            outputs.append(
                {
                    "item_id": item_id,
                    "chunk_ids": item.chunk_ids,
                    "chunk_sources": {
                        c.chunk_id: c.source_id or "<inline>"
                        for c in item.chunks
                    },
                    "source_files": item.source_files,
                    "prompt": user_msg,
                    "retrieval_context": retrieval_context,
                    "response": ai_text,
                    "document": document,
                    "thread_id": thread_id,
                    "skipped": False,
                    "validation": result.model_dump() if result is not None else None,
                }
            )

        elapsed_total = time.perf_counter() - t_pipeline
        logger.info(
            f"_process: all {total} item(s) processed  total_elapsed={elapsed_total:.3f}s  thread='{thread_id}'"
        )
        return outputs
