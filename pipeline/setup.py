"""Grouped construction settings for :class:`pipeline.engine.SasLLMPipeline`.

:class:`MemorySetup` bundles the pipeline's memory-wiring knobs — the store,
the two instruction memories, the extractor, and the chat identity — and owns
the cross-injection logic that used to live inline in the pipeline
constructor: defaulting the hub, binding store-less components to it, and
letting an extractor imply a thread memory. The pipeline takes one of these
(``memory_setup=``) and nothing else — the individual kwargs it replaced are
gone, so there is one place a memory knob can be set.

Logger name: ``pipeline.setup``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from app_config.spark import describe_master, master_url
from memory.context import MemoryContext
from memory.extractor import MemoryExtractor
from memory.policy import TaskPolicy
from memory.store import MemoryHub
from memory.thread_mem import ThreadMemory

if TYPE_CHECKING:
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
    ``delta_table`` is set); ``task_id`` / ``task_policy`` name or carry the
    long-term policy; ``thread_memory`` holds short-term notes (implied by a
    ``memory_extractor`` when omitted); ``chat_id`` identifies this
    instance's span of every thread it writes.
    """

    memory: MemoryHub | None = None
    task_id: str | None = None
    task_policy: TaskPolicy | None = None
    thread_memory: ThreadMemory | None = None
    memory_extractor: MemoryExtractor | None = None
    chat_id: str | None = None
    spark: "SparkSession | None" = None
    delta_table: str | None = None

    def build(self) -> BuiltMemory:
        """Wire everything together and return the built components.

        Store-less components are bound to the hub's KV store; a pre-built
        ``task_policy``'s own task id wins over ``task_id``; an extractor
        implies a :class:`ThreadMemory` (it needs somewhere to put temporary
        memories) and shares the policy/thread memory unless it brought its
        own.
        """
        hub = self.memory or self._default_hub(self.spark, self.delta_table)
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
        spark: "SparkSession | None", delta_table: str | None
    ) -> MemoryHub:
        if delta_table is None:
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
        return MemoryHub(spark=spark, table=delta_table)
