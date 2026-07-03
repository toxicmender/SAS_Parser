"""
pipeline.py — SAS chunker → LangChain in-memory short-term memory → LLM.

Architecture
------------
                 ┌──────────────────────┐
  SAS source ──► │ SasSemanticChunker   │ ──► list[SasChunk]
                 └──────────────────────┘
                          │  per-chunk
                          ▼
                 ┌──────────────────────────────────────────────┐
                 │ LangChain agent (create_agent)               │
                 │  • InMemorySaver checkpointer                │
                 │  • @before_model trim_messages middleware     │
                 │  • thread_id = chunk_id  (scoped per chunk)  │
                 └──────────────────────────────────────────────┘

Logging
-------
Logger name: ``sas_chunker.pipeline``

  Level    When emitted
  -------  ---------------------------------------------------------------
  DEBUG    Agent construction details, per-chunk thread assignments,
           message-trim decisions inside middleware
  INFO     Pipeline start / finish, per-chunk LLM call start / finish,
           token counts (when available), snapshot stats
  WARNING  Missing API key hint, no chunks to process
  ERROR    LLM invocation failure (exception re-raised after logging)
"""

from __future__ import annotations

import logging
import textwrap
import time
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from .chunker import SasSemanticChunker
from .models import SasChunk, SasChunkResult
from .pipeline_constants import _CONTEXT_TEMPLATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert SAS-to-PySpark migration assistant.
    You will be given a single semantic chunk of Base SAS source code
    (a DATA step, PROC step, macro definition, global statement, etc.)
    together with brief context about the surrounding program.

    For each chunk:
    1. Identify the SAS construct and its purpose.
    2. Translate it to equivalent PySpark, noting any semantic differences
       (date epoch offsets, MERGE vs join defaults, PDV vs DAG execution,
        macro expansion, PROC step equivalents, etc.).
    3. Flag any P0 silent-error risks with a ⚠️  marker.
    4. If translation is ambiguous or unsafe, say so explicitly rather
       than guessing.

    Be concise. Respond in structured Markdown.
""")


# ---------------------------------------------------------------------------
# @before_model middleware — rolling-window trim
# ---------------------------------------------------------------------------


def _make_trim_middleware(window_k: int):
    """
    Return a ``@before_model`` middleware that keeps only the last
    *window_k* (human, AI) turn-pairs plus the initial context message.

    Implements the LangChain short-term memory pattern from
    https://docs.langchain.com/oss/python/langchain/short-term-memory
    """
    mw_logger = logging.getLogger(f"{__name__}.trim_middleware")

    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        keep = 1 + window_k * 2  # system/context + (human+ai) pairs

        mw_logger.debug(
            "trim_messages: thread has %d messages  keep_limit=%d",
            len(messages),
            keep,
        )
        if len(messages) <= keep:
            mw_logger.debug("trim_messages: no trim needed")
            return None

        first = messages[0]
        recent = messages[-(window_k * 2) :]
        dropped = len(messages) - 1 - len(recent)  # excluding first
        mw_logger.info(
            "trim_messages: trimming %d old message(s) to stay within window_k=%d",
            dropped,
            window_k,
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                first,
                *recent,
            ]
        }

    return trim_messages


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SasLLMPipeline:
    """
    End-to-end pipeline: SAS source → semantic chunks → LLM responses,
    with LangChain ``InMemorySaver``-backed short-term memory per chunk thread.

    Parameters
    ----------
    model : str
        LangChain model string, e.g. ``"claude-haiku-4-5-20251001"``.
    min_words : int
        Soft lower bound for the chunker (words per chunk).
    max_words : int
        Hard upper bound for the chunker (words per chunk).
    window_k : int
        Rolling-window size in (human, AI) turn-pairs kept per thread.
    system_prompt : str | None
        Override the default SAS migration system prompt.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        min_words: int = 300,
        max_words: int = 700,
        window_k: int = 6,
        system_prompt: str | None = None,
    ) -> None:
        logger.info(
            "SasLLMPipeline.__init__  model=%s  min_words=%d  max_words=%d  window_k=%d",
            model,
            min_words,
            max_words,
            window_k,
        )
        self.model = model
        self.window_k = window_k

        self.chunker = SasSemanticChunker(min_words=min_words, max_words=max_words)

        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._checkpointer = InMemorySaver()

        logger.debug("SasLLMPipeline: creating agent with InMemorySaver checkpointer")
        trim_mw = _make_trim_middleware(window_k)
        self._agent = create_agent(
            model,
            tools=[],
            system_prompt=self._system_prompt,
            middleware=[trim_mw],
            checkpointer=self._checkpointer,
        )
        logger.debug("SasLLMPipeline: agent ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_file(self, path: str) -> list[dict[str, Any]]:
        """Chunk the SAS file at *path* and run each chunk through the LLM."""
        logger.info("run_file: '%s'", path)
        result = self.chunker.chunk_file(path)
        return self._process(result)

    def run_text(
        self,
        source: str,
        *,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk the SAS *source* string and run each chunk through the LLM."""
        label = source_id or "<inline>"
        logger.info("run_text: source_id='%s'  chars=%d", label, len(source))
        result = self.chunker.chunk_text(source, source_id=source_id)
        return self._process(result)

    def get_thread_messages(self, thread_id: str) -> list:
        """Return the raw message history stored for *thread_id*."""
        logger.debug("get_thread_messages: thread_id='%s'", thread_id)
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = self._agent.get_state(cfg)
        msgs = state.values.get("messages", [])
        logger.debug(
            "get_thread_messages: thread_id='%s'  messages=%d",
            thread_id,
            len(msgs),
        )
        return msgs

    def snapshot(self) -> dict[str, Any]:
        """
        Export all thread states from the ``InMemorySaver`` as a
        JSON-serialisable dict keyed by ``thread_id``.

        Suitable for hand-off to ``InProcessMemory.restore()`` or
        cross-process transfer.
        """
        logger.info("snapshot: exporting all thread states")
        snap: dict[str, Any] = {}
        namespaces = list(self._checkpointer.list_namespaces())
        logger.debug("snapshot: found %d thread namespace(s)", len(namespaces))

        for thread_id in namespaces:
            msgs = self.get_thread_messages(thread_id)
            snap[thread_id] = [
                {"type": m.__class__.__name__, "content": m.content} for m in msgs
            ]
            logger.debug(
                "snapshot: thread_id='%s'  messages=%d",
                thread_id,
                len(msgs),
            )

        import json as _json

        size = len(_json.dumps(snap))
        logger.info(
            "snapshot: done  threads=%d  approx_bytes=%d",
            len(snap),
            size,
        )
        return snap

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process(self, result: SasChunkResult) -> list[dict[str, Any]]:
        chunks = result.chunks
        total = len(chunks)
        label = result.source_id or "<inline>"

        if not chunks:
            logger.warning("_process: no chunks produced for source_id='%s'", label)
            return []

        logger.info(
            "_process: invoking LLM for %d chunk(s)  source_id='%s'  model=%s",
            total,
            label,
            self.model,
        )
        t_pipeline = time.perf_counter()
        outputs: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=1):
            thread_id = chunk.chunk_id
            logger.info(
                "_process: chunk %d/%d  id=%s  kind=%s  lines=%d-%d  thread=%s",
                idx,
                total,
                chunk.chunk_id,
                chunk.kind.value,
                chunk.start_line,
                chunk.end_line,
                thread_id,
            )
            user_msg = self._format_chunk_message(chunk, idx, total, result)
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            t_chunk = time.perf_counter()
            try:
                response = self._agent.invoke({"messages": user_msg}, cfg)
            except Exception:
                logger.error(
                    "_process: LLM call failed  chunk=%s  thread=%s",
                    chunk.chunk_id,
                    thread_id,
                    exc_info=True,
                )
                raise

            elapsed_chunk = time.perf_counter() - t_chunk
            ai_text = response["messages"][-1].content
            logger.info(
                "_process: chunk %s done  elapsed=%.3fs  response_chars=%d",
                chunk.chunk_id,
                elapsed_chunk,
                len(ai_text),
            )
            logger.debug(
                "_process: chunk %s response preview: %r",
                chunk.chunk_id,
                ai_text[:120].replace("\n", "↵"),
            )

            chunk_diags = [
                d
                for d in result.diagnostics
                if chunk.start_line <= d.start_line <= chunk.end_line
            ]
            outputs.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.kind.value,
                    "title": chunk.title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "diagnostics": [d.model_dump() for d in chunk_diags],
                    "response": ai_text,
                    "thread_id": thread_id,
                }
            )

        elapsed_total = time.perf_counter() - t_pipeline
        logger.info(
            "_process: all %d chunk(s) processed  total_elapsed=%.3fs  source_id='%s'",
            total,
            elapsed_total,
            label,
        )
        return outputs

    @staticmethod
    def _format_chunk_message(
        chunk: SasChunk,
        index: int,
        total: int,
        result: SasChunkResult,
    ) -> str:
        chunk_diags = [
            d
            for d in result.diagnostics
            if chunk.start_line <= d.start_line <= chunk.end_line
        ]
        msg = _CONTEXT_TEMPLATE.format(
            source_id=result.source_id or "unknown",
            total_chunks=total,
            chunk_id=chunk.chunk_id,
            index=index,
            kind=chunk.kind.value,
            title=chunk.title or "—",
            datasets=", ".join(chunk.metadata.referenced_datasets) or "none",
            librefs=", ".join(chunk.metadata.referenced_librefs) or "none",
            macro_defs=", ".join(chunk.metadata.defined_macros) or "none",
            macro_calls=", ".join(chunk.metadata.called_macros) or "none",
            diagnostics="; ".join(f"[{d.code}] {d.message}" for d in chunk_diags)
            or "none",
            text=chunk.text,
        )
        logger.debug(
            "_format_chunk_message: chunk=%s  prompt_chars=%d",
            chunk.chunk_id,
            len(msg),
        )
        return msg
