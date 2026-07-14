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
        for obj in m.component_objects:
            add("component_object", obj)
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
        ``temperature`` / ``requests_per_second`` do not apply to it.
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
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        temperature: float | None = None,
        max_input_tokens: int | None = None,
        requests_per_second: float | None = None,
        max_retries: int = 3,
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
    ) -> None:
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
                    [], user_instructions=user_instructions
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

        self._summarizer = summarizer
        if summarizer is not None and summarizer.store is None:
            # Summaries persist beside the thread they compress, so
            # snapshot()/restore() carry them along with the history.
            summarizer.store = self._memory.kv

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
        this thread are skipped (their stored responses are returned), so a
        crashed run picks up where it stopped instead of replaying — and
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
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        total = len(items)
        if not items:
            logger.warning(f"_process: nothing to process  thread='{thread_id}'")
            return []

        # Resume: items whose run fact reads "ok" are skipped; their stored
        # responses are recovered from the thread's (human, AI) turn pairs.
        # Error facts do NOT skip — a failed item is reprocessed and its
        # fact overwritten.
        completed: dict[str, dict[str, Any]] = {}
        recovered: list[BaseMessage] = []
        if resume:
            completed = {
                f["item_id"]: f
                for f in self.get_run_facts(thread_id)
                if f.get("status") == "ok"
            }
            if completed:
                recovered = self._memory.get_thread(thread_id).messages
                logger.info(
                    f"_process: resume  thread='{thread_id}'  "
                    f"{len(completed)} item(s) already complete"
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
            ai_text = response.content
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
                    "ts": time.time(),
                },
            )
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
                    "skipped": False,
                }
            )

        elapsed_total = time.perf_counter() - t_pipeline
        logger.info(
            f"_process: all {total} item(s) processed  total_elapsed={elapsed_total:.3f}s  thread='{thread_id}'"
        )
        return outputs
