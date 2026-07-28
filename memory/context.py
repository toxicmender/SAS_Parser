"""Assembly of the memory channels into one prompt-ready context.

The retrieval pipeline, in the order the channels are resolved::

    user query
        -> long-term task policy      (memory.policy   — stable, cacheable)
        -> short-term thread notes    (memory.thread_mem — volatile, ephemeral)
        -> rolling summary            (memory.summarize)
        -> selected recent messages   (memory.relevance / window_k)
        -> response

This module owns the first two and the boundary between them, because it is
the only place that sees both: a thread note may override an *overridable*
policy instruction, and may not override a fixed one, and that rule has to be
stated in the prompt where both are visible.

Where each half lands is a caching decision, not a stylistic one:

* :attr:`AssembledContext.system_suffix` — the policy — is appended to the
  **system prompt**, inside the ``cache_control`` breakpoint. It is the same
  text for every thread and every item, so it costs one cache write and then
  rides free.
* :attr:`AssembledContext.ephemeral` — the notes — are prompted **after** the
  breakpoint, through the pipeline's ``instructions`` channel: they differ per
  thread, and folding them into the cached prefix would miss the cache on
  every thread. They are prompted, never persisted, and never scored by
  ``RelevantHistorySelector`` (Architecture.md invariant 5).

Logger name: ``memory.context``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

_PRECEDENCE_NOTE = (
    "Where a conversation-specific instruction conflicts with a standing "
    "task instruction, follow the conversation-specific one — unless the "
    "standing instruction is marked (fixed), which always wins; say so "
    "plainly when you decline an override for that reason."
)


@dataclass(frozen=True)
class AssembledContext:
    """One turn's memory context, split by where it is allowed to go.

    Fields
    ------
    system_suffix
        Long-term policy text to append to the system prompt (cacheable).
        ``""`` when no policy is attached or the policy is empty.
    ephemeral
        Short-term messages to prompt after the cache breakpoint, through the
        ephemeral ``instructions`` channel. Empty when the thread has no
        live notes.
    policy_fingerprint
        The policy's content hash, ``""`` without one — recorded so a scored
        run knows which policy produced it.
    note_count
        How many live thread notes were rendered.
    """

    system_suffix: str = ""
    ephemeral: list[BaseMessage] = field(default_factory=list)
    policy_fingerprint: str = ""
    note_count: int = 0


class MemoryContext:
    """
    Resolves the long- and short-term channels for one task.

    Parameters
    ----------
    policy : memory.policy.TaskPolicy | None
        Long-term, task-scoped instructions. ``None`` disables the channel.
    thread_memory : memory.thread_mem.ThreadMemory | None
        Short-term, thread-scoped notes. ``None`` disables the channel.

    Both are duck-typed (``render()`` plus, for the policy, ``fingerprint``),
    so this module imports neither — the same independence the other
    ``memory`` feature modules keep.

    Usage
    -----
        ctx = MemoryContext(policy=policy, thread_memory=notes)
        assembled = ctx.assemble("thread-1")
        system_prompt = base_prompt + "\\n\\n" + assembled.system_suffix
        prompt_messages = assembled.ephemeral + [...]
    """

    def __init__(
        self,
        *,
        policy: Any | None = None,
        thread_memory: Any | None = None,
    ) -> None:
        self.policy = policy
        self.thread_memory = thread_memory
        logger.info(
            f"MemoryContext: policy={'on' if policy is not None else 'off'}  "
            f"thread_memory={'on' if thread_memory is not None else 'off'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def system_suffix(self) -> str:
        """The long-term half alone — read once when a consumer builds its
        (cacheable) system prompt. ``""`` without a policy."""
        return self.policy.render() if self.policy is not None else ""

    @property
    def policy_fingerprint(self) -> str:
        return getattr(self.policy, "fingerprint", "") if self.policy else ""

    def thread_messages(self, thread_id: str) -> list[BaseMessage]:
        """The short-term half alone, as ephemeral messages to prompt.

        The per-call half of :meth:`assemble`: a consumer that snapshotted
        :attr:`system_suffix` at construction (as ``SasLLMPipeline`` does)
        needs only this on each turn.
        """
        if self.thread_memory is None:
            return []
        block = self.thread_memory.render(thread_id)
        if not block:
            return []
        # The precedence rule is only meaningful when there is a policy to
        # take precedence over; a notes-only prompt would just be noise.
        if self.system_suffix:
            block = f"{block}\n\n{_PRECEDENCE_NOTE}"
        return [SystemMessage(content=block)]

    def assemble(self, thread_id: str) -> AssembledContext:
        """Both channels for *thread_id*, each tagged with where it may go."""
        ephemeral = self.thread_messages(thread_id)
        assembled = AssembledContext(
            system_suffix=self.system_suffix,
            ephemeral=ephemeral,
            policy_fingerprint=self.policy_fingerprint,
            note_count=(
                len(self.thread_memory.notes(thread_id))
                if self.thread_memory is not None
                else 0
            ),
        )
        logger.debug(
            f"assemble: thread='{thread_id}'  "
            f"policy={'yes' if assembled.system_suffix else 'no'}  "
            f"notes={assembled.note_count}"
        )
        return assembled
