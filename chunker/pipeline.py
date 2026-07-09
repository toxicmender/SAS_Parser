"""Glue layer: SAS chunker/batcher -> LangChain chat-memory threads -> LLM.
See chunker/README.md.

This module is the sole integration point between the chunker/batcher stack and
the ``memory`` / ``llm_client`` packages, so those stay independently usable.

Logger name: ``chunker.pipeline``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import START, MessagesState, StateGraph
from llm_client import LLMClient, LLMClientConfig
from memory.relevance import RelevantHistorySelector
from memory.short_mem import DatabricksMemory
from prompt_builder import ConstructKey, PromptBuilder

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
    if m.global_statement_keyword:
        tokens.append(m.global_statement_keyword)
    if m.control_flow_op:
        tokens.append(m.control_flow_op)
    tokens.extend(m.invokes_macros)
    return " ".join(t for t in tokens if t)


def _query_for_item(item: SasBatch | SasChunk) -> str:
    chunks = item.chunks if isinstance(item, SasBatch) else [item]
    return " ".join(_query_for_chunk(c) for c in chunks)


def _constructs_for_item(item: SasBatch | SasChunk) -> list[ConstructKey]:
    """The SAS constructs an item uses, as reference-lookup keys.

    Hazard flags add their canonical construct even when the name extractor
    missed it, so the selector still pulls the SYMPUT / %GOTO / %ABORT section.
    """
    chunks = item.chunks if isinstance(item, SasBatch) else [item]
    keys: list[ConstructKey] = []
    seen: set[ConstructKey] = set()

    def add(kind: str, name: str | None) -> None:
        if not name:
            return
        key = ConstructKey(kind=kind, name=name.lower())
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for chunk in chunks:
        m = chunk.metadata
        if chunk.kind is SasChunkKind.PROC_STEP:
            add("proc", m.proc_name)
        for fn in m.recognized_functions:
            add("function", fn)
        for routine in m.recognized_call_routines:
            add("call_routine", routine)
        add("global_statement", m.global_statement_keyword)
        if m.symput_scope_hazard:
            add("call_routine", "symput")
        if m.contains_computed_goto:
            add("macro_statement", "goto")
        if m.contains_abort:
            add("macro_statement", "abort")
    return keys


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


class SasLLMPipeline:
    """
    End-to-end pipeline: SAS source(s) -> semantic chunks -> dependency
    batches -> LLM responses, with a memory.short_mem.py-backed chat-memory
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
        LangChain chat-model string, e.g. ``"claude-haiku-4-5-20251001"``.
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
        Retries with exponential backoff for rate-limit (429-shaped)
        errors; other errors are never retried.
    min_words, max_words : int
        Forwarded to :class:`SasSemanticChunker`.
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
    include_options_chunks, include_comment_chunks : bool
        Forwarded to the batchers.
    memory : DatabricksMemory | None
        Pre-built memory.short_mem facade. If omitted, one is constructed from
        ``spark``/``delta_table``.
    spark : SparkSession | None
        Forwarded to :class:`DatabricksMemory` if ``memory`` is omitted.
        Only needed when ``delta_table`` is set; if omitted then, a local
        in-process Spark session is created.  The in-memory store never
        touches Spark, so no session is started when ``delta_table`` is
        ``None``.
    delta_table : str | None
        Forwarded to :class:`DatabricksMemory` if ``memory`` is omitted.
        ``None`` (default) keeps the store in-memory — a plain dict, no
        Delta table, no Spark/JVM.
    llm : Any | None
        Pre-built LangChain chat model to use instead of constructing one
        from ``model`` via :class:`llm_client.LLMClient`.  Useful for
        injecting a fake or pre-configured client (e.g. in tests).  The
        retry and input-token-budget layers still wrap an injected model;
        ``temperature`` / ``requests_per_second`` do not apply to it.
    prompt_builder : PromptBuilder | None
        Reference-PDF guidance source. When set, each item's prompt gains a
        block of instruction chunks relevant to that item's constructs
        (retrieved from the reference corpus). The guidance is **ephemeral**:
        it is prompted but never stored in the thread's history — see the
        load-bearing invariant on this in Architecture.md. ``None`` (default)
        disables guidance injection entirely.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        temperature: float | None = None,
        max_input_tokens: int | None = None,
        requests_per_second: float | None = None,
        max_retries: int = 3,
        min_words: int = 300,
        max_words: int = 700,
        output_language: str = "PySpark",
        system_prompt: str | None = None,
        window_k: int | None = 6,
        history_selector: RelevantHistorySelector | None = None,
        include_options_chunks: bool = True,
        include_comment_chunks: bool = False,
        memory: DatabricksMemory | None = None,
        spark: "SparkSession | None" = None,
        delta_table: str | None = None,
        llm: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        logger.info(
            f"SasLLMPipeline.__init__  model={model}  output_language={output_language}  "
            f"window_k={window_k}  guidance={'on' if prompt_builder else 'off'}"
        )
        self.model = model
        self.window_k = window_k
        self._history_selector = history_selector
        self._prompt_builder = prompt_builder

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

        # llm_client owns construction (temperature, output cap, rate limiter)
        # and invocation (429 retry, input-token budget). An injected chat model
        # replaces only the construction half; retry and budget still apply.
        self._llm_client = LLMClient(
            LLMClientConfig(
                model=model,
                temperature=temperature,
                max_input_tokens=max_input_tokens,
                requests_per_second=requests_per_second,
                max_retries=max_retries,
            ),
            llm=llm,
        )
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
            if self._history_selector is not None:
                selected = self._history_selector.select(history, inputs["input"])
                logger.debug(
                    f"_trim: relevance selector kept {len(selected)}/{len(history)} message(s)"
                )
                return {
                    "input": inputs["input"],
                    "history": selected,
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
                "history": history,
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
            response = chain.invoke(
                {
                    "input": input_message.content,
                    "history": history.messages,
                    "instructions": instructions,
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
    ) -> DatabricksMemory:
        if delta_table is None:
            # In-memory store never touches Spark, so don't boot a JVM session.
            logger.info(
                "SasLLMPipeline: in-memory message store (no Delta table, no "
                "Spark session needed)"
            )
            return DatabricksMemory(spark=spark, table=None)
        if spark is None:
            from pyspark.sql import SparkSession

            logger.info("SasLLMPipeline: no SparkSession provided, starting local one")
            spark = (
                SparkSession.builder.master("local[*]")
                .appName("chunker_pipeline")
                .getOrCreate()
            )
        return DatabricksMemory(spark=spark, table=delta_table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_file(
        self, path: str, *, thread_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Chunk + batch the SAS file at *path*, run every item through the LLM."""
        logger.info(f"run_file: '{path}'")
        result = self.chunker.chunk_file(path)
        batch_result = self.batcher.batch(result)
        tid = thread_id or self._default_thread_id([result.source_id or path])
        return self._process(
            batch_result.all_ordered_items, result.diagnostics, thread_id=tid
        )

    def run_text(
        self,
        source: str,
        *,
        source_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk + batch the SAS *source* string, run every item through the LLM."""
        label = source_id or "<inline>"
        logger.info(f"run_text: source_id='{label}'  chars={len(source)}")
        result = self.chunker.chunk_text(source, source_id=source_id)
        batch_result = self.batcher.batch(result)
        tid = thread_id or self._default_thread_id([result.source_id or label])
        return self._process(
            batch_result.all_ordered_items, result.diagnostics, thread_id=tid
        )

    def run_files(
        self, paths: list[str], *, thread_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Chunk every file in *paths*, resolve cross-file dependency batches
        via :class:`MultiFileBatcher`, and run every batch/singleton
        through the LLM on **one shared thread** for the whole corpus.
        """
        logger.info(f"run_files: {len(paths)} file(s)")
        file_results = [self.chunker.chunk_file(p) for p in paths]
        corpus = SasCorpus(file_results=file_results)
        multi_result = self.multi_batcher.batch(corpus)
        tid = thread_id or self._default_thread_id(corpus.source_ids)
        return self._process(
            multi_result.all_ordered_items, corpus.all_diagnostics, thread_id=tid
        )

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
        Delegates straight to :meth:`DatabricksMemory.snapshot` — pipeline
        does not re-implement export logic that memory.short_mem.py already owns.
        """
        logger.info("snapshot: delegating to DatabricksMemory")
        return self._memory.snapshot()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_thread_id(source_ids: list[str]) -> str:
        return "run::" + "+".join(source_ids)

    def _instruction_messages(
        self, item: SasBatch | SasChunk
    ) -> list[BaseMessage]:
        """Ephemeral reference-guidance message(s) for *item* — [] when none."""
        if self._prompt_builder is None:
            return []
        query = _query_for_item(item)
        constructs = _constructs_for_item(item)
        guidance = self._prompt_builder.build(query, constructs)
        if not guidance:
            return []
        item_id = item.batch_id if isinstance(item, SasBatch) else item.chunk_id
        logger.debug(
            f"_instruction_messages: item={item_id}  guidance_chars={len(guidance)}"
        )
        return [SystemMessage(guidance)]

    def _process(
        self,
        items: list[SasBatch | SasChunk],
        diagnostics: list[SasDiagnostic],
        *,
        thread_id: str,
    ) -> list[dict[str, Any]]:
        total = len(items)
        if not items:
            logger.warning(f"_process: nothing to process  thread='{thread_id}'")
            return []

        logger.info(
            f"_process: invoking LLM for {total} item(s)  thread='{thread_id}'  model={self.model}"
        )
        t_pipeline = time.perf_counter()
        outputs: list[dict[str, Any]] = []

        for idx, item in enumerate(items, start=1):
            is_batch = isinstance(item, SasBatch)
            item_id = item.batch_id if is_batch else item.chunk_id
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
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "instructions": self._instruction_messages(item),
                }
            }

            t_item = time.perf_counter()
            try:
                state = self._graph.invoke(
                    {"messages": [HumanMessage(user_msg)]}, cfg
                )
                response = state["messages"][-1]
            except Exception:
                logger.error(
                    f"_process: LLM call failed  item={item_id}  thread={thread_id}",
                    exc_info=True,
                )
                raise

            elapsed = time.perf_counter() - t_item
            ai_text = response.content
            logger.info(
                f"_process: item {item_id} done  elapsed={elapsed:.3f}s  response_chars={len(ai_text)}"
            )
            logger.debug(
                f"_process: item {item_id} response preview: "
                f"{ai_text[:120].replace(chr(10), chr(0x21B5))!r}"
            )

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
                }
            )

        elapsed_total = time.perf_counter() - t_pipeline
        logger.info(
            f"_process: all {total} item(s) processed  total_elapsed={elapsed_total:.3f}s  thread='{thread_id}'"
        )
        return outputs
