"""Long-term memory — durable, task-scoped instructions.

The **policy** channel: what the agent should do for a given *task*, stable
across every thread and every session that task spawns. Where
``memory.summarize`` compresses what was said and ``memory.relevance``
selects which of it to re-show, this module holds what was *decided* —
"always verify account ownership before discussing billing", "escalate
refunds above $500" — and keeps it out of the conversation entirely.

Stored under ``policy::{task_id}`` in a KV store as one record::

    {"version": 3, "instructions": [{"id", "text", "overridable", ...}]}

Two design points:

* **Overridable vs fixed.** Each instruction declares whether a
  conversation-specific note (``memory.thread_mem``) may override it.
  Preferences ("prefer concise responses") are overridable; guardrails
  ("escalate refunds above $500") are not, and
  :class:`~memory.context.MemoryContext` renders that distinction into the
  prompt so a short-term exception can never quietly disable one.
* **Snapshot per consumer, not per call.** :meth:`render` is meant to be
  read once and folded into the system prompt, where it is cacheable — a
  policy that changed mid-run would invalidate the provider prompt cache on
  every item. ``SasLLMPipeline`` therefore snapshots the policy at
  construction; the snapshot's :attr:`fingerprint` identifies which policy a
  run actually used. Call :meth:`reload` to pick up an out-of-band edit.

The ``store`` is duck-typed (``get``/``set``/``delete``) exactly as
:class:`~memory.summarize.RollingSummarizer`'s is —
:class:`memory.store.KVMemoryStore` fits, and this module imports neither
``store`` nor ``relevance``. Without a store, state is process-local.

Logger name: ``memory.policy``.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Appended to a non-overridable instruction wherever the policy is rendered.
# Part of the prompted text, so it is part of the fingerprint too: flipping an
# instruction from advisory to fixed is a different policy.
_FIXED_MARK = (
    "  (fixed: may not be overridden by a conversation-specific instruction)"
)


class PolicyInstruction(BaseModel):
    """One standing instruction for a task.

    Fields
    ------
    id
        Stable short identifier, assigned on add; the handle
        :meth:`TaskPolicy.remove` and the extractor's approval path use.
    text
        The instruction itself, as it will be prompted.
    overridable
        Whether a thread-scoped note may override this instruction. ``False``
        marks a guardrail: escalation thresholds, verification steps, legal
        or safety requirements.
    created_at
        Unix timestamp of when the instruction entered the policy.
    source
        Free-text provenance, e.g. ``"operator"`` or ``"MemoryExtractor"``.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    overridable: bool = True
    created_at: float = Field(default_factory=time.time)
    source: str | None = None


class TaskPolicy:
    """
    The long-term instruction set for one ``task_id``.

    Parameters
    ----------
    task_id : str
        Identifies the task these instructions govern, e.g.
        ``"sas_translation"`` or ``"customer_support"``. One task's policy is
        shared by every thread run under it.
    store : Any | None
        Where the policy persists: anything with ``get(key, default)`` /
        ``set(key, value, tags=..., source=...)`` / ``delete(key)``.
        ``None`` (default) keeps it process-local; ``SasLLMPipeline`` injects
        its own ``memory.kv`` into a store-less policy.
    default_instructions : Iterable[str | PolicyInstruction] | None
        Applied **only when the task has no stored policy yet** — a
        declarative bootstrap, not an override. Re-running a program with a
        different default never silently rewrites a policy an operator has
        since edited.

    Usage
    -----
        policy = TaskPolicy("customer_support", store=mem.kv)
        policy.add("Prefer concise responses.")
        policy.add("Escalate refund requests above $500.", overridable=False)
        print(policy.render())
    """

    def __init__(
        self,
        task_id: str,
        *,
        store: Any | None = None,
        default_instructions: Iterable[str | PolicyInstruction] | None = None,
    ) -> None:
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        self.task_id = task_id
        self.store = store
        self._local: dict[str, Any] | None = None
        state = self._load()
        if not state["instructions"] and default_instructions:
            state = {
                "version": 1,
                "instructions": [
                    _as_instruction(entry, source="default").model_dump()
                    for entry in default_instructions
                ],
            }
            self._save(state)
            logger.info(
                f"TaskPolicy: seeded task '{task_id}' with "
                f"{len(state['instructions'])} default instruction(s)"
            )
        self._state = state
        logger.info(
            f"TaskPolicy: task='{task_id}'  v{self.version}  "
            f"{len(self.instructions)} instruction(s)  "
            f"fingerprint={self.fingerprint or '-'}  "
            f"store={'external' if store is not None else 'process-local'}"
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @property
    def instructions(self) -> list[PolicyInstruction]:
        """The task's instructions, in the order they will be prompted."""
        return [PolicyInstruction(**raw) for raw in self._state["instructions"]]

    @property
    def version(self) -> int:
        """Monotonic edit counter — bumped by every mutation."""
        return int(self._state["version"])

    @property
    def fingerprint(self) -> str:
        """Content hash of the prompted text, ``""`` for an empty policy.

        Identifies *what the model was told*, so two runs under different
        policies are never compared as equals (the same role
        ``UserInstructionSet.fingerprint`` plays for reference guidance).
        Reordering or rewording changes it; the version counter does not
        enter into it.
        """
        rendered = self._rendered_body()
        if not rendered:
            return ""
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]

    def texts(self, *, overridable: bool | None = None) -> list[str]:
        """Instruction texts, optionally restricted by *overridable*."""
        return [
            i.text
            for i in self.instructions
            if overridable is None or i.overridable == overridable
        ]

    def as_dicts(self) -> list[dict[str, Any]]:
        """Plain dicts, the shape ``validation.EvaluationRun.task_policy``
        carries so a scored run knows what it was supposed to obey."""
        return [i.model_dump() for i in self.instructions]

    def render(self) -> str:
        """The policy as a prompt block, or ``""`` when it is empty.

        Meant to be appended to the system prompt (stable ⇒ cacheable).
        Fixed instructions are marked inline so a conversation-specific note
        cannot appear to outrank one.
        """
        body = self._rendered_body()
        if not body:
            return ""
        return (
            "Standing instructions for this task — follow them in every "
            "response:\n" + body
        )

    def _rendered_body(self) -> str:
        return "\n".join(
            f"{n}. {i.text}{'' if i.overridable else _FIXED_MARK}"
            for n, i in enumerate(self.instructions, start=1)
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        overridable: bool = True,
        source: str | None = None,
    ) -> PolicyInstruction:
        """Append one instruction and persist. Returns what was stored."""
        instruction = _as_instruction(text, overridable=overridable, source=source)
        state = {
            "version": self.version + 1,
            "instructions": [*self._state["instructions"], instruction.model_dump()],
        }
        self._commit(state)
        logger.info(
            f"TaskPolicy.add: task='{self.task_id}'  id={instruction.id}  "
            f"overridable={overridable}  v{self.version}"
        )
        return instruction

    def remove(self, instruction_id: str) -> bool:
        """Drop the instruction with *instruction_id*. False if absent."""
        kept = [
            raw for raw in self._state["instructions"] if raw.get("id") != instruction_id
        ]
        if len(kept) == len(self._state["instructions"]):
            return False
        self._commit({"version": self.version + 1, "instructions": kept})
        logger.info(
            f"TaskPolicy.remove: task='{self.task_id}'  id={instruction_id}  "
            f"v{self.version}"
        )
        return True

    def replace(self, instructions: Sequence[str | PolicyInstruction]) -> None:
        """Replace the whole set (one version bump, not one per entry)."""
        self._commit(
            {
                "version": self.version + 1,
                "instructions": [
                    _as_instruction(entry).model_dump() for entry in instructions
                ],
            }
        )
        logger.info(
            f"TaskPolicy.replace: task='{self.task_id}'  "
            f"{len(instructions)} instruction(s)  v{self.version}"
        )

    def clear(self) -> None:
        """Delete the task's policy record entirely."""
        if self.store is not None:
            self.store.delete(self._key())
        else:
            self._local = None
        self._state = {"version": 0, "instructions": []}
        logger.info(f"TaskPolicy.clear: task='{self.task_id}'")

    def reload(self) -> None:
        """Re-read the stored policy, picking up out-of-band edits.

        Consumers that snapshot :meth:`render` (the pipeline folds it into a
        cached system prompt) do not see the new text until they rebuild.
        """
        self._state = self._load()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key(self) -> str:
        return f"policy::{self.task_id}"

    def _commit(self, state: dict[str, Any]) -> None:
        self._save(state)
        self._state = state

    def _load(self) -> dict[str, Any]:
        stored = (
            self.store.get(self._key(), None) if self.store is not None else self._local
        )
        if not isinstance(stored, dict):
            return {"version": 0, "instructions": []}
        raw = stored.get("instructions")
        instructions = (
            [entry for entry in raw if isinstance(entry, dict)]
            if isinstance(raw, list)
            else []
        )
        return {"version": int(stored.get("version", 0)), "instructions": instructions}

    def _save(self, state: dict[str, Any]) -> None:
        if self.store is not None:
            self.store.set(
                self._key(),
                state,
                tags=["policy", self.task_id],
                source="TaskPolicy",
            )
        else:
            self._local = dict(state)


def _as_instruction(
    entry: str | PolicyInstruction,
    *,
    overridable: bool = True,
    source: str | None = None,
) -> PolicyInstruction:
    if isinstance(entry, PolicyInstruction):
        return entry
    text = str(entry).strip()
    if not text:
        raise ValueError("instruction text must be non-empty")
    return PolicyInstruction(text=text, overridable=overridable, source=source)
