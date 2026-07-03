"""
pipeline.py — glue layer: SAS chunker/batcher -> LangChain chat-memory
threads (memory.short_mem.py / persistent_memory.py) -> LLM.

Architecture
------------
                 +----------------------+
  SAS source(s) ->| SasSemanticChunker  |--> SasChunkResult(s)
                 +----------------------+
                          |
                          v
                 +----------------------+
  SasCorpus ----->| SasChunkBatcher /   |--> SasBatchResult /
                 | MultiFileBatcher    |    SasMultiBatchResult
                 +----------------------+
                          |  all_ordered_items (SasBatch | SasChunk)
                          v
                 +------------------------------------------------+
                 | pipeline.py (this module)                       |
                 |  - thread_id = run id (all batches/files for    |
                 |    one run share ONE KVChatMessageHistory)      |
                 |  - RunnableWithMessageHistory(prompt | llm)     |
                 |  - prompt text sourced from pipeline_constants  |
                 +------------------------------------------------+
                          |
                          v
                     LLM responses, one per batch/singleton

memory.short_mem.py (persistent_memory.py) is never imported by chunker.py,
models.py, or batcher.py, and this module never reaches into its
internals beyond the public DatabricksMemory facade — pipeline.py is
the sole integration point between the chunker/batcher stack and the
memory backend, so memory.short_mem.py stays independently usable/testable.

Logging
-------
Logger name: ``chunker.pipeline``

  Level    When emitted
  -------  ---------------------------------------------------------------
  DEBUG    Message formatting details, trim decisions
  INFO     Pipeline start/finish, per-item LLM call start/finish
  WARNING  No items to process
  ERROR    LLM invocation failure (exception re-raised after logging)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from .batcher import MultiFileBatcher, SasChunkBatcher
from .chunker import SasSemanticChunker
from .models import (
    SasBatch,
    SasChunk,
    SasCorpus,
    SasDiagnostic,
)
from .persistent_memory import DatabricksMemory
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
        macro_defs=_fmt_list(m.defined_macros or m.defines_macros),
        macro_calls=_fmt_list(m.called_macros or m.invokes_macros),
        macro_var_op=m.macro_var_op or "none",
        declared_macro_vars=_fmt_list(m.declared_macro_vars),
        referenced_macro_vars=_fmt_list(m.referenced_macro_vars),
        produced_macrovars=_fmt_list(m.produces_macrovars),
        consumed_macrovars=_fmt_list(m.consumes_macrovars),
        global_statement_keyword=m.global_statement_keyword or "none",
        control_flow_op=m.control_flow_op or "none",
        sas_functions=_fmt_list(m.recognized_functions or m.referenced_sas_functions),
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
        "_format_chunk_message: chunk=%s  prompt_chars=%d", chunk.chunk_id, len(msg)
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
        standard_autocall_macros=_fmt_list(batch.standard_autocall_macros),
        required_macrovars=_fmt_list(batch.required_macrovars),
        produced_macrovars=_fmt_list(batch.produced_macrovars),
        diagnostics="; ".join(f"[{d.code}] {d.message}" for d in diags) or "none",
        members=members,
    )
    logger.debug(
        "_format_batch_message: batch=%s  members=%d  prompt_chars=%d",
        batch.batch_id,
        len(batch.chunks),
        len(msg),
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
    min_words, max_words : int
        Forwarded to :class:`SasSemanticChunker`.
    output_language : str
        Target language named in the system prompt (default ``"PySpark"``).
    system_prompt : str | None
        Override the default prompt from ``pipeline_constants``.
    window_k : int | None
        Rolling-window size in (human, AI) turn-pairs kept in context per
        LLM call. ``None`` disables trimming (full history every call).
    include_options_chunks, include_comment_chunks : bool
        Forwarded to the batchers.
    memory : DatabricksMemory | None
        Pre-built memory.short_mem facade. If omitted, one is constructed from
        ``spark``/``delta_table``.
    spark : SparkSession | None
        Forwarded to :class:`DatabricksMemory` if ``memory`` is omitted.
        If also omitted, a local in-process Spark session is created.
    delta_table : str | None
        Forwarded to :class:`DatabricksMemory` if ``memory`` is omitted.
        ``None`` keeps the store in-memory (no Delta table).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        min_words: int = 300,
        max_words: int = 700,
        output_language: str = "PySpark",
        system_prompt: str | None = None,
        window_k: int | None = 6,
        include_options_chunks: bool = True,
        include_comment_chunks: bool = False,
        memory: DatabricksMemory | None = None,
        spark: "SparkSession | None" = None,
        delta_table: str | None = None,
        llm: Any | None = None,
    ) -> None:
    ) -> None:
        logger.info(
            "SasLLMPipeline.__init__  model=%s  output_language=%s  window_k=%s",
            model,
            output_language,
            window_k,
        )
        self.model = model
        self.window_k = window_k

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

        llm = init_chat_model(model)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        def _trim(inputs: dict[str, Any]) -> dict[str, Any]:
            history: list[BaseMessage] = inputs.get("history", [])
            k = self.window_k
            if k is not None and len(history) > k * 2:
                dropped = len(history) - k * 2
                logger.debug(
                    "_trim: dropping %d old message(s), window_k=%d", dropped, k
                )
                history = history[-(k * 2) :]
            return {"input": inputs["input"], "history": history}

        chain = RunnableLambda(_trim) | prompt | llm

        logger.debug("SasLLMPipeline: wiring RunnableWithMessageHistory")
        self._runnable = RunnableWithMessageHistory(
            chain,
            get_session_history=self._memory.get_thread,
            input_messages_key="input",
            history_messages_key="history",
        )
        logger.debug("SasLLMPipeline: ready")

    @staticmethod
    def _build_default_memory(
        spark: "SparkSession | None",
        delta_table: str | None,
    ) -> DatabricksMemory:
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
        logger.info("run_file: '%s'", path)
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
        logger.info("run_text: source_id='%s'  chars=%d", label, len(source))
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
        logger.info("run_files: %d file(s)", len(paths))
        file_results = [self.chunker.chunk_file(p) for p in paths]
        corpus = SasCorpus(file_results=file_results)
        multi_result = self.multi_batcher.batch(corpus)
        tid = thread_id or self._default_thread_id(corpus.source_ids)
        return self._process(
            multi_result.all_ordered_items, corpus.all_diagnostics, thread_id=tid
        )

    def get_thread_messages(self, thread_id: str) -> list[BaseMessage]:
        """Return the raw message history stored for *thread_id*."""
        logger.debug("get_thread_messages: thread_id='%s'", thread_id)
        msgs = self._memory.get_thread(thread_id).messages
        logger.debug(
            "get_thread_messages: thread_id='%s'  messages=%d", thread_id, len(msgs)
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

    def _process(
        self,
        items: list[SasBatch | SasChunk],
        diagnostics: list[SasDiagnostic],
        *,
        thread_id: str,
    ) -> list[dict[str, Any]]:
        total = len(items)
        if not items:
            logger.warning("_process: nothing to process  thread='%s'", thread_id)
            return []

        logger.info(
            "_process: invoking LLM for %d item(s)  thread='%s'  model=%s",
            total,
            thread_id,
            self.model,
        )
        t_pipeline = time.perf_counter()
        outputs: list[dict[str, Any]] = []
        cfg: RunnableConfig = {"configurable": {"session_id": thread_id}}

        for idx, item in enumerate(items, start=1):
            is_batch = isinstance(item, SasBatch)
            item_id = item.batch_id if is_batch else item.chunk_id
            logger.info(
                "_process: item %d/%d  id=%s  is_batch=%s  thread=%s",
                idx,
                total,
                item_id,
                is_batch,
                thread_id,
            )

            user_msg = (
                _format_batch_message(item, idx, total, diagnostics)
                if is_batch
                else _format_chunk_message(item, idx, total, diagnostics)
            )

            t_item = time.perf_counter()
            try:
                response = self._runnable.invoke({"input": user_msg}, cfg)
            except Exception:
                logger.error(
                    "_process: LLM call failed  item=%s  thread=%s",
                    item_id,
                    thread_id,
                    exc_info=True,
                )
                raise

            elapsed = time.perf_counter() - t_item
            ai_text = response.content
            logger.info(
                "_process: item %s done  elapsed=%.3fs  response_chars=%d",
                item_id,
                elapsed,
                len(ai_text),
            )
            logger.debug(
                "_process: item %s response preview: %r",
                item_id,
                ai_text[:120].replace("\n", "\u21b5"),
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
            "_process: all %d item(s) processed  total_elapsed=%.3fs  thread='%s'",
            total,
            elapsed_total,
            thread_id,
        )
        return outputs
