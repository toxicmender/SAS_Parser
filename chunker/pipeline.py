"""Glue layer: SAS chunker/batcher -> LangChain chat-memory threads -> LLM.
See chunker/README.md.

This module is the sole integration point between the chunker/batcher stack and
the ``memory`` / ``llm_client`` packages, so those stay independently usable.

Logger name: ``chunker.pipeline``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, NamedTuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import START, MessagesState, StateGraph
from llm_client import LLMClient, LLMClientConfig
from memory.relevance import RelevantHistorySelector
from memory.store import MemoryHub
from memory.summarize import RollingSummarizer
from prompt_builder import ConstructKey, PromptBuilder, UserInstructionSet

from .batcher import MultiFileBatcher, SasChunkBatcher
from .chunker import SasSemanticChunker
from .models import (
    SasBatch,
    SasChunk,
    SasChunkKind,
    SasCorpus,
    SasDiagnostic,
)
from .pipeline_constants import (
    _BATCH_CONTEXT_TEMPLATE,
    _BATCH_MEMBER_TEMPLATE,
    _CONTEXT_TEMPLATE,
    _SYSTEM_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt_list(xs: list[str] | None) -> str:
    return ", ".join(xs) if xs else "none"


# ---------------------------------------------------------------------------
# Item -> instruction-retrieval query / constructs
#
# This is the sole metadata -> prompt_builder mapping; keeping it here (not in
# prompt_builder) is what lets that package stay free of any chunker import.
# ---------------------------------------------------------------------------


def _query_for_chunk(chunk: SasChunk) -> str:
    """Free-text retrieval query from a chunk's *constructs*, not its source.

    Dataset names and literal source text are retrieval noise for reference
    guidance, which is organised by construct; kind, title, and the recognised
    functions / routines / statements are the signal.
    """
    m = chunk.metadata
    tokens = [chunk.kind.value.replace("_", " "), chunk.title or ""]
    if m.proc_name:
        tokens.append(m.proc_name)
    tokens.extend(m.recognized_functions)
    tokens.extend(m.recognized_call_routines)
    # "hash object", "hiter object", ... — the reference guides discuss
    # component objects by that phrasing ("hash object", "hash table").
    tokens.extend(f"{obj} object" for obj in m.component_objects)
    if m.global_statement_keyword:
        tokens.append(m.global_statement_keyword)
    if m.control_flow_op:
        tokens.append(m.control_flow_op)
    tokens.extend(m.invokes_macros)
    return " ".join(t for t in tokens if t)


def _query_for_item(item: SasBatch | SasChunk) -> str:
    chunks = item.chunks if isinstance(item, SasBatch) else [item]
    return " ".join(_query_for_chunk(c) for c in chunks)


class _IdentifierSets(NamedTuple):
    """An item's construct identifiers as deduplicated sets, plus hazard flags.

    A :class:`SasBatch` exposes these aggregates directly (its set properties);
    a lone :class:`SasChunk` is normalised to the same shape here, so
    :func:`_constructs_for_item` has a single, hashing-based code path.
    """

    proc_names: set[str]
    functions: set[str]
    call_routines: set[str]
    component_objects: set[str]
    global_statement_keywords: set[str]
    symput_scope_hazard: bool
    contains_abort: bool
    contains_computed_goto: bool


def _identifier_sets(item: SasBatch | SasChunk) -> _IdentifierSets:
    """The item's construct identifiers as sets (batch aggregates or one chunk)."""
    if isinstance(item, SasBatch):
        return _IdentifierSets(
            proc_names=item.proc_names,
            functions=item.recognized_functions,
            call_routines=item.recognized_call_routines,
            component_objects=item.component_objects,
            global_statement_keywords=item.global_statement_keywords,
            symput_scope_hazard=item.has_symput_scope_hazard,
            contains_abort=item.has_abort,
            contains_computed_goto=item.has_computed_goto,
        )
    m = item.metadata
    proc = (
        {m.proc_name}
        if item.kind is SasChunkKind.PROC_STEP and m.proc_name
        else set()
    )
    globals_ = (
        {m.global_statement_keyword} if m.global_statement_keyword else set()
    )
    return _IdentifierSets(
        proc_names=proc,
        functions=set(m.recognized_functions),
        call_routines=set(m.recognized_call_routines),
        component_objects=set(m.component_objects),
        global_statement_keywords=globals_,
        symput_scope_hazard=m.symput_scope_hazard,
        contains_abort=m.contains_abort,
        contains_computed_goto=m.contains_computed_goto,
    )


def _constructs_for_item(item: SasBatch | SasChunk) -> list[ConstructKey]:
    """The SAS constructs an item uses, as reference-lookup keys.

    Driven off the item's aggregated identifier *sets* (see
    :class:`_IdentifierSets`): each name becomes a frozen — therefore hashable
    — :class:`ConstructKey`, deduplicated through a hashed ``seen`` set, so the
    selector's construct lookup (also hash-based) fires an instruction for a
    construct only when that construct is actually present in the batch. Sets
    are iterated in sorted order to keep the key sequence deterministic.

    Hazard flags add their canonical construct even when the name extractor
    missed it, so the selector still pulls the SYMPUT / %GOTO / %ABORT section.
    """
    ids = _identifier_sets(item)
    keys: list[ConstructKey] = []
    seen: set[ConstructKey] = set()

    def add(kind: str, name: str | None) -> None:
        if not name:
            return
        key = ConstructKey(kind=kind, name=name.lower())
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for name in sorted(ids.proc_names):
        add("proc", name)
    for name in sorted(ids.functions):
        add("function", name)
    for name in sorted(ids.call_routines):
        add("call_routine", name)
    for name in sorted(ids.component_objects):
        add("component_object", name)
    for name in sorted(ids.global_statement_keywords):
        add("global_statement", name)
    if ids.symput_scope_hazard:
        add("call_routine", "symput")
    if ids.contains_computed_goto:
        add("macro_statement", "goto")
    if ids.contains_abort:
        add("macro_statement", "abort")
    return keys


def _kinds_for_item(item: SasBatch | SasChunk) -> set[str]:
    """The SasChunkKind values an item uses, as ``[kind: ...]`` scope tokens."""
    chunks = item.chunks if isinstance(item, SasBatch) else [item]
    return {c.kind.value for c in chunks}


# Metadata predicate flags, keyed by the ``[meta: ...]`` token an instruction
# scopes on. Each maps to a metadata attribute that is truthy when the flag
# holds; the pipeline owns this vocabulary so prompt_builder treats the tokens
# as opaque. Kept in sync with the docstring in user_instructions.py.
_META_FLAG_ATTRS: tuple[tuple[str, str], ...] = (
    ("symput_hazard", "symput_scope_hazard"),
    ("abort", "contains_abort"),
    ("computed_goto", "contains_computed_goto"),
    ("component_object", "component_objects"),
    ("unclosed_block", "has_unclosed_block"),
    ("includes", "includes"),
    ("defines_macros", "defines_macros"),
    ("invokes_macros", "invokes_macros"),
    ("produces_macrovars", "produces_macrovars"),
    ("automatic_vars", "referenced_automatic_vars"),
)


def _meta_flags_for_item(item: SasBatch | SasChunk) -> set[str]:
    """The metadata predicate flags an item raises, as ``[meta: ...]`` tokens.

    Unioned over member chunks (a batch flag holds if any member raises it),
    so an instruction scoped ``[meta: symput_hazard]`` fires for a batch that
    contains a SYMPUT scope hazard anywhere inside it.
    """
    chunks = item.chunks if isinstance(item, SasBatch) else [item]
    flags: set[str] = set()
    for chunk in chunks:
        m = chunk.metadata
        for token, attr in _META_FLAG_ATTRS:
            if token not in flags and getattr(m, attr):
                flags.add(token)
    return flags


def _diagnostics_for_chunk(
    chunk: SasChunk, diagnostics: list[SasDiagnostic]
) -> list[SasDiagnostic]:
    return [
        d
        for d in diagnostics
        if d.source_id in (None, chunk.source_id)
        and chunk.start_line <= d.start_line <= chunk.end_line
    ]


def _diagnostics_for_batch(
    batch: SasBatch, diagnostics: list[SasDiagnostic]
) -> list[SasDiagnostic]:
    seen: list[SasDiagnostic] = []
    seen_keys: set[tuple[str, int, str | None]] = set()
    for member in batch.chunks:
        for d in _diagnostics_for_chunk(member, diagnostics):
            key = (d.code, d.start_line, d.source_id)
            if key not in seen_keys:
                seen.append(d)
                seen_keys.add(key)
    return seen


def _format_chunk_message(
    chunk: SasChunk,
    index: int,
    total: int,
    diagnostics: list[SasDiagnostic],
) -> str:
    m = chunk.metadata
    diags = _diagnostics_for_chunk(chunk, diagnostics)
    msg = _CONTEXT_TEMPLATE.format(
        source_id=chunk.source_id or "unknown",
        chunk_id=chunk.chunk_id,
        index=index,
        total_items=total,
        kind=chunk.kind.value,
        title=chunk.title or "\u2014",
        datasets=_fmt_list(m.referenced_datasets),
        librefs=_fmt_list(m.referenced_librefs),
        input_datasets=_fmt_list(m.input_datasets),
        output_datasets=_fmt_list(m.output_datasets),
        macro_defs=_fmt_list(m.defines_macros),
        macro_calls=_fmt_list(m.invokes_macros),
        macro_var_op=m.macro_var_op or "none",
        declared_macro_vars=_fmt_list(m.declared_macro_vars),
        referenced_macro_vars=_fmt_list(m.referenced_macro_vars),
        produced_macrovars=_fmt_list(m.produces_macrovars),
        consumed_macrovars=_fmt_list(m.consumes_macrovars),
        global_statement_keyword=m.global_statement_keyword or "none",
        control_flow_op=m.control_flow_op or "none",
        sas_functions=_fmt_list(m.recognized_functions),
        call_routines=_fmt_list(m.recognized_call_routines),
        automatic_vars=_fmt_list(m.referenced_automatic_vars),
        symput_hazard=(
            f"yes ({_fmt_list(m.symput_hazard_vars)})"
            if m.symput_scope_hazard
            else "no"
        ),
        contains_abort="yes" if m.contains_abort else "no",
        contains_computed_goto="yes" if m.contains_computed_goto else "no",
        diagnostics="; ".join(f"[{d.code}] {d.message}" for d in diags) or "none",
        text=chunk.text,
    )
    logger.debug(
        f"_format_chunk_message: chunk={chunk.chunk_id}  prompt_chars={len(msg)}"
    )
    return msg


def _format_batch_message(
    batch: SasBatch,
    index: int,
    total: int,
    diagnostics: list[SasDiagnostic],
) -> str:
    diags = _diagnostics_for_batch(batch, diagnostics)
    members = "\n".join(
        _BATCH_MEMBER_TEMPLATE.format(
            chunk_id=c.chunk_id,
            kind=c.kind.value,
            source_id=c.source_id or "unknown",
            start_line=c.start_line,
            end_line=c.end_line,
            title=c.title or "\u2014",
            text=c.text,
        )
        for c in batch.chunks
    )
    msg = _BATCH_CONTEXT_TEMPLATE.format(
        batch_id=batch.batch_id,
        index=index,
        total_items=total,
        is_cross_file="yes" if batch.is_cross_file else "no",
        source_files=_fmt_list(batch.source_files),
        chunk_count=len(batch.chunks),
        start_line=batch.start_line,
        end_line=batch.end_line,
        reason=batch.reason or "none",
        input_datasets=_fmt_list(batch.input_datasets),
        output_datasets=_fmt_list(batch.output_datasets),
        required_macros=_fmt_list(batch.required_macros),
        defined_macros=_fmt_list(batch.defined_macros),
        required_librefs=_fmt_list(batch.required_librefs),
        standard_autocall_macros=_fmt_list(batch.standard_autocall_macros),
        required_macrovars=_fmt_list(batch.required_macrovars),
        produced_macrovars=_fmt_list(batch.produced_macrovars),
        diagnostics="; ".join(f"[{d.code}] {d.message}" for d in diags) or "none",
        members=members,
    )
    logger.debug(
        f"_format_batch_message: batch={batch.batch_id}  members={len(batch.chunks)}  prompt_chars={len(msg)}"
    )
    return msg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Bookkeeping keys the pipeline adds around a stored inline-validation verdict
# (index/total/ts, plus item_id added by the reader). Stripped when a verdict
# is recovered on resume so the recovered `validation` value matches the shape
# a freshly-scored item carries (the bare CaseResult dump).
_RECOVERED_VALIDATION_DROP = frozenset({"item_id", "index", "total", "ts"})


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

    Parameters
    ----------
    model : str
        LangChain chat-model string, e.g. ``"claude-sonnet-4-5"``.
    temperature : float | None
        Sampling temperature for the LLM. ``None`` (default) keeps the
        provider default. Forwarded to :class:`llm_client.LLMClientConfig`.
    max_input_tokens : int | None
        Input-token budget per LLM call; an over-budget prompt raises
        :class:`llm_client.InputTokenLimitError` instead of being sent.
        ``None`` (default) disables counting.
    requests_per_second : float | None
        Proactive client-side request throttle (``InMemoryRateLimiter``).
        ``None`` (default) disables it. Ignored when ``llm`` is injected —
        rate limiters attach at model construction time.
    max_retries : int
        Retries with exponential backoff for transient errors (rate
        limits, overload / 5xx, timeouts, connection drops); other
        errors are never retried.
    base_url : str | None
        Provider endpoint override (proxy / gateway URL). ``None``
        (default) defers to config.json ``llm_client.base_url``, then
        the provider default. Ignored when ``llm`` is injected.
    api_key : str | None
        Explicit API key; held as a masked ``SecretStr`` inside
        :class:`llm_client.LLMClientConfig`, never logged. ``None``
        (default) defers to the provider's environment variable (e.g.
        ``ANTHROPIC_API_KEY``). Ignored when ``llm`` is injected.
    url_headers : dict[str, str] | None
        Extra HTTP headers sent with every LLM request (forwarded as
        ``default_headers`` — gateway auth, tracing, ...). ``None``
        (default) defers to config.json ``llm_client.url_headers``.
        Ignored when ``llm`` is injected.
    timeout : float | None
        Per-request LLM timeout in seconds. ``None`` (default) defers to
        config.json ``llm_client.timeout``, then the provider default.
        Ignored when ``llm`` is injected.
    model_kwargs : dict | None
        Provider-specific request-body extras (e.g. ``{"top_k": 40}``).
        ``None`` (default) defers to config.json
        ``llm_client.model_kwargs``. Ignored when ``llm`` is injected.
    llm_kwargs : dict | None
        Escape hatch: arbitrary keyword arguments merged last into the
        ``init_chat_model`` call (the :class:`llm_client.LLMClientConfig`
        field ``kwargs``), overriding anything the named knobs produced.
        Ignored when ``llm`` is injected.
    min_words, max_words : int | None
        Forwarded to :class:`SasSemanticChunker`. ``None`` (default) lets
        the chunker read ``sas_chunker.*`` from config.json (see the
        ``app_config`` package), falling back to 300/700.
    output_language : str
        Target language named in the system prompt (default ``"PySpark"``).
    system_prompt : str | None
        Override the default prompt from ``pipeline_constants``.
    window_k : int | None
        Rolling-window size in (human, AI) turn-pairs kept in context per
        LLM call. ``None`` disables trimming (full history every call).
        Ignored when ``history_selector`` is set.
    history_selector : RelevantHistorySelector | None
        Relevance-based history selection: per LLM call, prompt only the
        turn pairs most relevant to the current batch/chunk message
        (BM25 + optional FAISS dense retrieval, RRF-fused) instead of the
        recency window. ``None`` (default) keeps ``window_k`` behaviour.
    summarizer : RollingSummarizer | None
        Rolling thread summarization (``memory.summarize``): turns older
        than the summarizer's recency tail are folded into one running
        summary per thread, prepended to every prompt as a SystemMessage
        after trimming/selection. Like reference guidance, the summary is
        **prompted but never persisted** to the ``msg::`` history — it
        lives in the KV layer and is re-derivable from the full stored
        thread. A summarizer constructed without a ``store`` is given this
        pipeline's ``memory.kv``. ``None`` (default) disables compression.
    include_options_chunks, include_comment_chunks : bool
        Forwarded to the batchers.
    memory : MemoryHub | None
        Pre-built memory.store facade. If omitted, one is constructed from
        ``spark``/``delta_table``.
    spark : SparkSession | None
        Forwarded to :class:`MemoryHub` if ``memory`` is omitted.
        Only needed when ``delta_table`` is set; if omitted then, a local
        in-process Spark session is created.  The in-memory store never
        touches Spark, so no session is started when ``delta_table`` is
        ``None``.
    delta_table : str | None
        Forwarded to :class:`MemoryHub` if ``memory`` is omitted.
        ``None`` (default) keeps the store in-memory — a plain dict, no
        Delta table, no Spark/JVM.
    llm : Any | None
        Pre-built LangChain chat model to use instead of constructing one
        from ``model`` via :class:`llm_client.LLMClient`.  Useful for
        injecting a fake or pre-configured client (e.g. in tests).  The
        retry and input-token-budget layers still wrap an injected model;
        the construction-time knobs (``temperature``, ``base_url``,
        ``api_key``, ``url_headers``, ``timeout``, ``model_kwargs``,
        ``llm_kwargs``, ``requests_per_second``) do not apply to it.
    prompt_builder : PromptBuilder | None
        Reference-PDF guidance source. When set, each item's prompt gains a
        block of instruction chunks relevant to that item's constructs
        (retrieved from the reference corpus). The guidance is **ephemeral**:
        it is prompted but never stored in the thread's history — see the
        load-bearing invariant on this in Architecture.md. ``None`` (default)
        disables guidance injection entirely.
    user_instructions : str | UserInstructionSet | None
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
    validator : Any | None
        Optional inline validator (``validation.live.LiveValidator`` —
        duck-typed, so this package imports nothing from ``validation``,
        which itself imports this one). When set, each item is scored the
        moment its response returns and the verdict is written to this run's
        conversation memory, beside its run fact (see
        :meth:`get_validation_facts`). With ``validation_retries == 0``
        (default) validation is observe-only: a failing or erroring
        validation never retries the item or aborts the run. ``None``
        (default) disables inline validation entirely.
    validation_retries : int
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
        model: str = "claude-sonnet-4-5",
        *,
        temperature: float | None = None,
        max_input_tokens: int | None = None,
        requests_per_second: float | None = None,
        max_retries: int = 3,
        base_url: str | None = None,
        api_key: str | None = None,
        url_headers: dict[str, str] | None = None,
        timeout: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        output_language: str = "PySpark",
        system_prompt: str | None = None,
        window_k: int | None = 6,
        history_selector: RelevantHistorySelector | None = None,
        summarizer: RollingSummarizer | None = None,
        include_options_chunks: bool = True,
        include_comment_chunks: bool = False,
        memory: MemoryHub | None = None,
        spark: "SparkSession | None" = None,
        delta_table: str | None = None,
        llm: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        user_instructions: "str | UserInstructionSet | None" = None,
        validator: Any | None = None,
        validation_retries: int = 0,
    ) -> None:
        if validation_retries < 0:
            raise ValueError(
                f"validation_retries must be >= 0, got {validation_retries}"
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
                    output_language=output_language,
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
        logger.info(
            f"SasLLMPipeline.__init__  model={model}  output_language={output_language}  "
            f"window_k={window_k}  guidance={'on' if prompt_builder else 'off'}"
        )
        self.model = model
        self.window_k = window_k
        self._output_language = output_language
        self._history_selector = history_selector
        self._prompt_builder = prompt_builder
        self._validator = validator
        # Validation-driven retry/resume is active only with both a validator
        # and a positive budget; otherwise validation stays observe-only.
        self._validation_retries = (
            validation_retries if validator is not None else 0
        )

        self.chunker = SasSemanticChunker(min_words=min_words, max_words=max_words)
        self.batcher = SasChunkBatcher(include_options_chunks=include_options_chunks)
        self.multi_batcher = MultiFileBatcher(
            include_options_chunks=include_options_chunks,
            include_comment_chunks=include_comment_chunks,
        )

        self._system_prompt = system_prompt or _SYSTEM_PROMPT_TEMPLATE.format(
            output_language=output_language
        )

        self._memory = memory or self._build_default_memory(spark, delta_table)

        self._summarizer = summarizer
        if summarizer is not None and summarizer.store is None:
            # Summaries persist beside the thread they compress, so
            # snapshot()/restore() carry them along with the history.
            summarizer.store = self._memory.kv

        # llm_client owns construction (temperature, endpoint overrides,
        # output cap, rate limiter) and invocation (transient-error retry,
        # input-token budget). An injected chat model replaces only the
        # construction half; retry and budget still apply.
        llm_config_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_input_tokens": max_input_tokens,
            "requests_per_second": requests_per_second,
            "max_retries": max_retries,
        }
        # Endpoint knobs are forwarded only when set, so an omitted argument
        # still defers to the config.json llm_client defaults (an explicit
        # None here would override them).
        for key, value in (
            ("base_url", base_url),
            ("api_key", api_key),
            ("url_headers", url_headers),
            ("timeout", timeout),
            ("model_kwargs", model_kwargs),
            ("kwargs", llm_kwargs),
        ):
            if value is not None:
                llm_config_kwargs[key] = value
        self._llm_client = LLMClient(LLMClientConfig(**llm_config_kwargs), llm=llm)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                MessagesPlaceholder("history"),
                # Ephemeral per-item reference guidance: 0 or 1 SystemMessage,
                # prompted but never persisted (see _call_model).
                MessagesPlaceholder("instructions"),
                ("human", "{input}"),
            ]
        )

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

        chain = RunnableLambda(_trim) | prompt | self._llm_client.as_runnable()

        def _call_model(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, list[BaseMessage]]:
            # One graph invocation == one conversational turn: only the LAST
            # state message is prompted, and exactly that message plus the
            # response is persisted (the store never records an unshown message).
            # Reference guidance is prompted via `instructions` but is NOT part
            # of the persisted turn — it is re-derivable, would bloat the store,
            # and would pollute relevance-based history selection.
            thread_id = config["configurable"]["thread_id"]
            instructions = config["configurable"].get("instructions", [])
            history = self._memory.get_thread(thread_id)
            input_message = state["messages"][-1]
            history_messages = history.messages
            # The rolling summary is ephemeral like the guidance: prompted
            # (prepended in _trim), never persisted to the msg:: history.
            summary = (
                self._summarizer.refresh(thread_id, history_messages)
                if self._summarizer is not None
                else None
            )
            response = chain.invoke(
                {
                    "input": input_message.content,
                    "history": history_messages,
                    "instructions": instructions,
                    "summary": summary,
                }
            )
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

    @staticmethod
    def _build_default_memory(
        spark: "SparkSession | None",
        delta_table: str | None,
    ) -> MemoryHub:
        if delta_table is None:
            # In-memory store never touches Spark, so don't boot a JVM session.
            logger.info(
                "SasLLMPipeline: in-memory message store (no Delta table, no "
                "Spark session needed)"
            )
            return MemoryHub(spark=spark, table=None)
        if spark is None:
            from pyspark.sql import SparkSession

            logger.info("SasLLMPipeline: no SparkSession provided, starting local one")
            spark = (
                SparkSession.builder.master("local[*]")
                .appName("chunker_pipeline")
                .getOrCreate()
            )
        return MemoryHub(spark=spark, table=delta_table)

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
            batch_result.all_ordered_items,
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
            batch_result.all_ordered_items,
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
        file_results = [self.chunker.chunk_file(p) for p in paths]
        corpus = SasCorpus(file_results=file_results)
        multi_result = self.multi_batcher.batch(corpus)
        tid = thread_id or self._default_thread_id(corpus.source_ids)
        return self._process(
            multi_result.all_ordered_items,
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
        copied = self._memory.fork_thread(
            src_thread_id,
            dst_thread_id,
            upto_messages=None if upto_items is None else upto_items * 2,
        )
        for fact in self.get_run_facts(src_thread_id):
            index = fact.get("index", 0)
            if upto_items is not None and index > upto_items:
                continue
            if fact.get("status") != "ok":
                continue
            value = {k: v for k, v in fact.items() if k != "item_id"}
            self._record_run_fact(dst_thread_id, fact["item_id"], value)
        # Carry the copied items' inline verdicts onto the fork too, so a
        # forked-then-resumed run recovers them the same way a plain resume
        # does (no-op when the source run had no validator).
        for fact in self.get_validation_facts(src_thread_id):
            if upto_items is not None and (fact.get("index") or 0) > upto_items:
                continue
            value = {k: v for k, v in fact.items() if k != "item_id"}
            self._record_validation_fact(dst_thread_id, fact["item_id"], value)
        logger.info(
            f"fork_run: '{src_thread_id}' -> '{dst_thread_id}'  "
            f"upto_items={upto_items}  messages_copied={copied}"
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
        timing) that later batches, later runs, and the planned resume
        feature can query without replaying the message history.
        """
        prefix = f"run::{thread_id}::item::"
        facts = [
            {"item_id": item["key"][len(prefix) :], **item["value"]}
            for item in self._memory.kv.all_items()
            if item["key"].startswith(prefix)
        ]
        facts.sort(key=lambda f: f.get("index", 0))
        return facts

    def get_validation_facts(self, thread_id: str) -> list[dict[str, Any]]:
        """Per-item inline-validation verdicts recorded for *thread_id*.

        Present only when the pipeline was built with a ``validator``: each
        item scored during the run leaves one record under
        ``validation::{thread_id}::item::{item_id}`` (score, passed, per-metric
        results, index/total), stored beside the run facts by
        ``validation.live.LiveValidator``. Ordered by item index; empty when
        no validator ran on the thread.
        """
        prefix = f"validation::{thread_id}::item::"
        facts = [
            {"item_id": item["key"][len(prefix) :], **item["value"]}
            for item in self._memory.kv.all_items()
            if item["key"].startswith(prefix)
        ]
        facts.sort(key=lambda f: f.get("index") or 0)
        return facts

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
    def _default_thread_id(source_ids: list[str]) -> str:
        return "run::" + "+".join(source_ids)

    @staticmethod
    def _recovered_response(messages: list[BaseMessage], fact: dict[str, Any]) -> str | None:
        """The stored AI response for a completed item, or ``None``.

        A run halts on its first failure and each item persists exactly one
        (human, AI) pair, so completed item *i* (1-based fact index) maps to
        message ``2 * i - 1``. Anything inconsistent — a hand-edited thread,
        retention pruning — degrades to ``None`` rather than guessing.
        """
        position = 2 * fact.get("index", 0) - 1
        if 0 < position < len(messages) and isinstance(messages[position], AIMessage):
            return messages[position].content
        return None

    @staticmethod
    def _recovered_validation(
        fact: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """The stored inline-validation verdict for a recovered item, or ``None``.

        Normalises the KV-stored fact back to the bare ``CaseResult`` dump a
        freshly-scored item carries — dropping the pipeline's bookkeeping keys
        (see :data:`_RECOVERED_VALIDATION_DROP`) — so a resumed run's outputs
        are shaped identically whether an item was replayed or recovered.
        ``None`` when the original attempt had no validator (or scoring failed
        then and left no verdict).
        """
        if not fact:
            return None
        return {
            k: v for k, v in fact.items() if k not in _RECOVERED_VALIDATION_DROP
        }

    def _record_run_fact(
        self, thread_id: str, item_id: str, fact: dict[str, Any]
    ) -> None:
        """Write one per-item outcome record to the KV layer (the "write
        context" channel): small facts only — the full response already
        lives in the msg:: history and is never duplicated here."""
        self._memory.kv.set(
            f"run::{thread_id}::item::{item_id}",
            fact,
            tags=["run-item", thread_id],
            source="pipeline",
        )

    def _record_validation_fact(
        self, thread_id: str, item_id: str, fact: dict[str, Any]
    ) -> None:
        """Upsert one inline-validation verdict under the same key schema
        ``validation.live.LiveValidator`` uses (``validation::{thread_id}::``
        ``item::{item_id}``), so a fork's copied verdicts read back through
        :meth:`get_validation_facts` exactly like inline-written ones."""
        self._memory.kv.set(
            f"validation::{thread_id}::item::{item_id}",
            fact,
            tags=["validation", thread_id],
            source="pipeline",
        )

    def _instruction_messages(
        self, item: SasBatch | SasChunk
    ) -> list[BaseMessage]:
        """Ephemeral reference-guidance message(s) for *item* — [] when none."""
        if self._prompt_builder is None:
            return []
        query = _query_for_item(item)
        constructs = _constructs_for_item(item)
        guidance = self._prompt_builder.build(
            query,
            constructs,
            output_language=self._output_language,
            kinds=_kinds_for_item(item),
            meta_flags=_meta_flags_for_item(item),
        )
        if not guidance:
            return []
        item_id = item.batch_id if isinstance(item, SasBatch) else item.chunk_id
        logger.debug(
            f"_instruction_messages: item={item_id}  guidance_chars={len(guidance)}"
        )
        return [SystemMessage(guidance)]

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
        item: SasBatch | SasChunk,
        idx: int,
        total: int,
        *,
        thread_id: str,
        user_msg: str,
        base_instructions: list[BaseMessage],
    ) -> tuple[str, Any, int]:
        """Generate (and, if enabled, iteratively repair) one item's answer.

        Sends *user_msg* on *thread_id*; when a validator is attached the
        response is scored inline. With ``validation_retries > 0`` a failing
        verdict rolls the just-appended turn back off the thread (via
        :meth:`KVChatMessageHistory.truncate_to`) and re-prompts with a
        corrective note, up to the retry budget, so exactly one — the final —
        (human, AI) pair persists per item. Returns
        ``(response_text, CaseResult | None, attempts)``; the ``CaseResult``
        is ``None`` when no validator ran or scoring raised (swallowed, as in
        the observe-only policy). Any LLM-call exception propagates to the
        caller, which records the error fact.
        """
        item_id = item.batch_id if isinstance(item, SasBatch) else item.chunk_id
        history = self._memory.get_thread(thread_id)
        max_attempts = 1 + self._validation_retries
        feedback: list[BaseMessage] = []
        attempt = 0
        while True:
            attempt += 1
            # Roll-back point: everything already committed for earlier items.
            # Only needed when a retry might follow (else skip the history load).
            len_before = len(history.messages) if max_attempts > 1 else 0
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "instructions": base_instructions + feedback,
                }
            }
            state = self._graph.invoke({"messages": [HumanMessage(user_msg)]}, cfg)
            ai_text = state["messages"][-1].content

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
                return ai_text, result, attempt

            logger.info(
                f"_answer_item: item={item_id} failed validation "
                f"(score={result.score:.3f}) on attempt {attempt}/{max_attempts}; "
                f"rolling back and retrying  thread={thread_id}"
            )
            # Drop this attempt's turn pair so the retry replaces it in place.
            history.truncate_to(len_before)
            feedback = [self._validation_feedback_message(result)]

    def _resume_state(
        self, items: list[SasBatch | SasChunk], thread_id: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[BaseMessage]]:
        """Resolve what a resume can skip and what it must redo.

        Returns ``(completed, completed_validations, recovered)`` where
        *completed* maps item_id -> run fact for items that may be skipped and
        their responses recovered, *completed_validations* maps item_id ->
        stored verdict, and *recovered* is the pre-rewind message snapshot the
        skipped items' responses are read from.

        With validation-driven retry active, an ``ok`` item whose stored
        verdict *failed* is not "done": the thread is rewound to the earliest
        such (or otherwise-missing) item and that item's — and every later
        item's — run/validation facts and turns are dropped, so the main loop
        regenerates from there on a clean, consistent history. Without it,
        this reproduces the original policy: every ``ok`` item is skipped.
        """
        ok_facts = {
            f["item_id"]: f
            for f in self.get_run_facts(thread_id)
            if f.get("status") == "ok"
        }
        completed_validations = {
            f["item_id"]: f for f in self.get_validation_facts(thread_id)
        }
        recovered: list[BaseMessage] = []
        if ok_facts:
            recovered = self._memory.get_thread(thread_id).messages

        if not self._validation_retries:
            if ok_facts:
                logger.info(
                    f"_process: resume  thread='{thread_id}'  {len(ok_facts)} "
                    f"item(s) already complete  "
                    f"{len(completed_validations)} stored verdict(s)"
                )
            return ok_facts, completed_validations, recovered

        # Validation-aware resume: find the first item that is not "done and
        # good" (missing, errored, or an ok item whose stored verdict failed).
        def _satisfied(item_id: str) -> bool:
            if item_id not in ok_facts:
                return False
            verdict = completed_validations.get(item_id)
            # No verdict → cannot call it a failure; treat as satisfied.
            return verdict is None or verdict.get("passed", True)

        redo_start: int | None = None
        for pos, item in enumerate(items, start=1):
            item_id = item.batch_id if isinstance(item, SasBatch) else item.chunk_id
            if not _satisfied(item_id):
                redo_start = pos
                break

        if redo_start is None:
            logger.info(
                f"_process: resume  thread='{thread_id}'  all {len(items)} "
                f"item(s) complete and passing; nothing to redo"
            )
            return ok_facts, completed_validations, recovered

        self._rewind_for_resume(items, thread_id, redo_start)
        completed = {
            item.batch_id if isinstance(item, SasBatch) else item.chunk_id: ok_facts[
                item.batch_id if isinstance(item, SasBatch) else item.chunk_id
            ]
            for pos, item in enumerate(items, start=1)
            if pos < redo_start
        }
        logger.info(
            f"_process: resume  thread='{thread_id}'  keeping {len(completed)} "
            f"passing item(s), regenerating from item {redo_start}/{len(items)}"
        )
        return completed, completed_validations, recovered

    def _rewind_for_resume(
        self, items: list[SasBatch | SasChunk], thread_id: str, redo_start: int
    ) -> None:
        """Rewind *thread_id* to just before item *redo_start* (1-based).

        Truncates the thread to the ``redo_start - 1`` completed (human, AI)
        pairs that precede it and drops the run/validation facts of item
        *redo_start* and every later item, so the main loop regenerates them
        onto a clean, append-only history instead of leaving stale turns and
        facts behind.
        """
        keep_pairs = redo_start - 1
        removed = self._memory.get_thread(thread_id).truncate_to(keep_pairs * 2)
        for pos, item in enumerate(items, start=1):
            if pos < redo_start:
                continue
            item_id = item.batch_id if isinstance(item, SasBatch) else item.chunk_id
            self._memory.kv.delete(f"run::{thread_id}::item::{item_id}")
            self._memory.kv.delete(f"validation::{thread_id}::item::{item_id}")
        logger.info(
            f"_rewind_for_resume: thread='{thread_id}'  rewound to item "
            f"{redo_start} (kept {keep_pairs} pair(s), removed {removed} message(s))"
        )

    def _process(
        self,
        items: list[SasBatch | SasChunk],
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
        # active, an ok-but-failing item is redone too (see _resume_state).
        completed: dict[str, dict[str, Any]] = {}
        completed_validations: dict[str, dict[str, Any]] = {}
        recovered: list[BaseMessage] = []
        if resume:
            completed, completed_validations, recovered = self._resume_state(
                items, thread_id
            )

        logger.info(
            f"_process: invoking LLM for {total} item(s)  thread='{thread_id}'  model={self.model}"
        )
        t_pipeline = time.perf_counter()
        outputs: list[dict[str, Any]] = []

        for idx, item in enumerate(items, start=1):
            is_batch = isinstance(item, SasBatch)
            item_id = item.batch_id if is_batch else item.chunk_id

            fact = completed.get(item_id)
            if fact is not None:
                logger.info(
                    f"_process: item {idx}/{total}  id={item_id}  already "
                    f"complete; skipping  thread={thread_id}"
                )
                outputs.append(
                    {
                        "item_id": item_id,
                        "is_batch": is_batch,
                        "chunk_ids": item.chunk_ids
                        if is_batch
                        else [item.chunk_id],
                        "source_files": item.source_files
                        if is_batch
                        else [item.source_id or "unknown"],
                        "kind": None if is_batch else item.kind.value,
                        "response": self._recovered_response(recovered, fact),
                        "thread_id": thread_id,
                        "skipped": True,
                        "validation": self._recovered_validation(
                            completed_validations.get(item_id)
                        ),
                    }
                )
                continue

            logger.info(
                f"_process: item {idx}/{total}  id={item_id}  is_batch={is_batch}  thread={thread_id}"
            )

            user_msg = (
                _format_batch_message(item, idx, total, diagnostics)
                if is_batch
                else _format_chunk_message(item, idx, total, diagnostics)
            )
            # Per-item guidance rides in the config, not the state, so it is
            # prompted without ever entering the persisted message history.
            base_instructions = self._instruction_messages(item)

            t_item = time.perf_counter()
            try:
                # Generate — and, when validation_retries > 0, iteratively
                # repair — the answer. Scoring, the retry loop, and the
                # roll-back of superseded attempts all live in _answer_item;
                # exactly one (human, AI) pair persists per item.
                ai_text, result, attempts = self._answer_item(
                    item,
                    idx,
                    total,
                    thread_id=thread_id,
                    user_msg=user_msg,
                    base_instructions=base_instructions,
                )
            except Exception as exc:
                logger.error(
                    f"_process: LLM call failed  item={item_id}  thread={thread_id}",
                    exc_info=True,
                )
                self._record_run_fact(
                    thread_id,
                    item_id,
                    {
                        "status": "error",
                        "index": idx,
                        "total": total,
                        "is_batch": is_batch,
                        "error": repr(exc),
                        "ts": time.time(),
                    },
                )
                raise

            elapsed = time.perf_counter() - t_item
            self._record_run_fact(
                thread_id,
                item_id,
                {
                    "status": "ok",
                    "index": idx,
                    "total": total,
                    "is_batch": is_batch,
                    "elapsed_s": round(elapsed, 3),
                    "response_chars": len(ai_text),
                    "attempts": attempts,
                    "ts": time.time(),
                },
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
                    "is_batch": is_batch,
                    "chunk_ids": item.chunk_ids if is_batch else [item.chunk_id],
                    "source_files": item.source_files
                    if is_batch
                    else [item.source_id or "unknown"],
                    "kind": None if is_batch else item.kind.value,
                    "response": ai_text,
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
