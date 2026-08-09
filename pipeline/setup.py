"""Grouped construction settings for :class:`pipeline.engine.SasLLMPipeline`.

One dataclass per concern, so the pipeline's constructor names concerns rather
than knobs. It took 22 keyword arguments before these; it takes seven now, and
each of the four groups is a thing a caller can build once and reuse.

:class:`MemorySetup`
    The store, the two instruction memories, the extractor, the chat identity,
    and the history policy (window, selector, summarizer). It also owns the
    cross-injection logic that used to live inline in the pipeline
    constructor: defaulting the hub, binding store-less components to it, and
    letting an extractor imply a thread memory.
:class:`ChunkingSetup`
    How the source is split and grouped — word limits, what to include, the
    merge caps, and the dataset-name mapping the batchers apply.
:class:`PromptingSetup`
    What the model is asked and how — the system prompt, structured output,
    prompt caching, and the two instruction channels.
:class:`ValidationSetup`
    The inline validator and its retry budget.

The rule is the same one commit 97b5e8a applied to ``llm_config`` and
``memory_setup``: **one spelling per knob**. The individual keyword arguments
these replaced are gone rather than deprecated, so there is never a second
place to set the same thing — and no rejection branch to stop someone setting
it in both.

Logger name: ``pipeline.setup``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import app_config
from app_config.spark import describe_master, master_url
from memory.context import MemoryContext
from memory.extractor import MemoryExtractor
from memory.policy import TaskPolicy
from memory.relevance import RelevantHistorySelector
from memory.store import MemoryHub
from memory.summarize import RollingSummarizer
from memory.thread_mem import ThreadMemory

if TYPE_CHECKING:
    from prompt_builder import PromptBuilder, UserInstructionSet
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


class BuiltMemory(NamedTuple):
    """The wired result of :meth:`MemorySetup.build`."""

    hub: MemoryHub
    chat_id: str
    task_id: str | None
    task_policy: TaskPolicy | None
    thread_memory: ThreadMemory | None
    extractor: MemoryExtractor | None
    context: MemoryContext


@dataclass
class MemorySetup:
    """Memory wiring for one pipeline instance.

    Fields carry exactly the semantics of the same-named ``SasLLMPipeline``
    keyword arguments (see its docstring): ``memory`` is a pre-built
    :class:`MemoryHub` (default: in-memory store, or Delta when
    ``delta_table`` or config.json ``memory.delta_table`` is set);
    ``cdf_audit_table`` likewise falls back to config.json
    ``memory.cdf_audit_table``; both configured names must be
    ``Catalog.Schema.Table``. ``task_id`` / ``task_policy`` name or carry the
    long-term policy; ``thread_memory`` holds short-term notes (implied by a
    ``memory_extractor`` when omitted); ``chat_id`` identifies this
    instance's span of every thread it writes; ``window_k`` /
    ``history_selector`` / ``summarizer`` decide which stored turns are
    actually prompted.
    """

    memory: MemoryHub | None = None
    task_id: str | None = None
    task_policy: TaskPolicy | None = None
    thread_memory: ThreadMemory | None = None
    memory_extractor: MemoryExtractor | None = None
    chat_id: str | None = None
    spark: "SparkSession | None" = None
    delta_table: str | None = None
    cdf_consumer_id: str | None = None
    cdf_audit_table: str | None = None
    max_delta_write_retries: int = 3
    # History policy: which of the stored turns are actually prompted. It
    # belongs here rather than in a group of its own because it is the same
    # subject as the store — what is remembered versus what is re-sent — and
    # splitting the two would put `memory` and `window_k` in different
    # objects while they describe the same conversation.
    window_k: int | None = 6
    history_selector: RelevantHistorySelector | None = None
    summarizer: RollingSummarizer | None = None

    def build(self) -> BuiltMemory:
        """Wire everything together and return the built components.

        Store-less components are bound to the hub's KV store; a pre-built
        ``task_policy``'s own task id wins over ``task_id``; an extractor
        implies a :class:`ThreadMemory` (it needs somewhere to put temporary
        memories) and shares the policy/thread memory unless it brought its
        own.
        """
        delta_table = self.delta_table or app_config.memory_table_value("delta_table")
        cdf_audit_table = (
            self.cdf_audit_table
            or app_config.memory_table_value("cdf_audit_table")
        )
        hub = self.memory or self._default_hub(
            self.spark,
            delta_table,
            cdf_consumer_id=self.cdf_consumer_id,
            cdf_audit_table=cdf_audit_table,
            max_write_retries=self.max_delta_write_retries,
        )
        chat_id = self.chat_id or uuid.uuid4().hex[:12]

        task_id = self.task_id
        task_policy = self.task_policy
        if task_policy is None and task_id is not None:
            task_policy = TaskPolicy(task_id, store=hub.kv)
        elif task_policy is not None:
            task_id = task_policy.task_id
            if task_policy.store is None:
                task_policy.store = hub.kv
                task_policy.reload()

        thread_memory = self.thread_memory
        extractor = self.memory_extractor
        if thread_memory is None and extractor is not None:
            thread_memory = extractor.thread_memory or ThreadMemory()
        if thread_memory is not None and thread_memory.store is None:
            thread_memory.store = hub.kv
        if extractor is not None:
            # An extractor sharing a policy/thread memory with the pipeline
            # is the point — otherwise it would write memories nothing reads.
            if extractor.store is None:
                extractor.store = hub.kv
            if extractor.policy is None:
                extractor.policy = task_policy
            if extractor.thread_memory is None:
                extractor.thread_memory = thread_memory

        return BuiltMemory(
            hub=hub,
            chat_id=chat_id,
            task_id=task_id,
            task_policy=task_policy,
            thread_memory=thread_memory,
            extractor=extractor,
            context=MemoryContext(policy=task_policy, thread_memory=thread_memory),
        )

    @staticmethod
    def _default_hub(
        spark: "SparkSession | None",
        delta_table: str | None,
        *,
        cdf_consumer_id: str | None = None,
        cdf_audit_table: str | None = None,
        max_write_retries: int = 3,
    ) -> MemoryHub:
        if delta_table is None:
            if cdf_consumer_id is not None or cdf_audit_table is not None:
                raise ValueError("CDF settings require MemorySetup.delta_table")
            # In-memory store never touches Spark, so don't boot a JVM session.
            logger.info(
                "MemorySetup: in-memory message store (no Delta table, no "
                "Spark session needed)"
            )
            return MemoryHub(spark=spark, table=None)
        if spark is None:
            from pyspark.sql import SparkSession

            master = master_url()
            logger.info(
                f"MemorySetup: no SparkSession provided, building one against "
                f"{describe_master(master)}"
            )
            spark = (
                SparkSession.builder.master(master)
                .appName("chunker_pipeline")
                .getOrCreate()
            )
        return MemoryHub(
            spark=spark,
            table=delta_table,
            cdf_consumer_id=cdf_consumer_id,
            cdf_audit_table=cdf_audit_table,
            max_write_retries=max_write_retries,
        )


@dataclass
class ChunkingSetup:
    """How the SAS source is split and grouped, for one pipeline instance.

    Fields carry exactly the semantics of the same-named ``SasLLMPipeline``
    keyword arguments they replaced. ``None`` on the word limits defers to
    ``config.json`` ``sas_chunker.*``; ``max_merged_tokens`` ``None`` derives a
    packing budget from the model's input headroom and ``0`` turns packing off.

    ``databricks_mapping`` is the dataset-name cross-reference, which both
    batchers apply as a post-pass after grouping. Sourcing one is the caller's
    job — see :mod:`xref.sourcing`.
    """

    min_words: int | None = None
    max_words: int | None = None
    include_options_chunks: bool = True
    include_comment_chunks: bool = False
    max_merged_chunks: int = 8
    max_merged_tokens: int | None = None
    databricks_mapping: dict[str, str] | None = None


@dataclass
class PromptingSetup:
    """What the model is asked, and how, for one pipeline instance.

    Fields carry exactly the semantics of the same-named ``SasLLMPipeline``
    keyword arguments they replaced: ``system_prompt`` overrides the built-in
    template outright, ``structured_output`` asks for a typed
    :class:`~pipeline.response_models.TranslationDocument` instead of Markdown,
    ``prompt_caching`` puts a cache breakpoint on the system block, and the two
    instruction channels supply per-item guidance.

    ``None`` on the three flags defers to ``config.json``; ``user_instructions``
    ``None`` still loads a standing instructions file if one is configured,
    which is why it is not simply "off".
    """

    system_prompt: str | None = None
    structured_output: bool | None = None
    prompt_caching: bool | None = None
    prompt_builder: "PromptBuilder | None" = None
    user_instructions: "str | UserInstructionSet | None" = None


@dataclass
class ValidationSetup:
    """Inline validation for one pipeline instance.

    ``validator`` scores each item the moment its response returns; ``retries``
    re-generates a failing item with the failed metrics fed back as correction.
    ``0`` is observe-only — score and store, never act — and retries without a
    validator is a no-op the pipeline warns about, since it reads as protection
    that is not there.
    """

    validator: Any = None
    retries: int = 0
