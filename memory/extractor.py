"""Memory extraction — deciding what a turn should be remembered *as*.

The write path for the other two memory channels::

    conversation turn
        -> cue gate            (cheap, offline; no cue => no LLM call)
        -> classifier          (one LLM call)
            |- permanent  -> a TaskPolicy proposal, held for approval
            '- temporary  -> a ThreadMemory note, applied immediately

Two asymmetries are deliberate, and they are the whole safety story of this
module:

* **The cue gate runs before the model.** Extraction is a per-turn LLM call;
  running it on every turn of a long SAS translation run would roughly double
  the call count for nothing, because ordinary work turns declare no policy.
  A turn with no instruction-shaped cue ("from now on", "always", "never",
  "instead", "exception", "approved", …) is skipped offline.
* **Permanent memories are proposed, not written.** A temporary note is
  bounded — it expires, and it dies with its thread. A policy instruction is
  durable and applies to every future thread of that task, so a
  misclassified turn would be a lasting behavioural change nobody asked for.
  Permanent candidates land in a pending list and become policy only through
  :meth:`MemoryExtractor.approve` (or ``auto_apply_permanent=True``, which is
  off by default and logs a warning at construction).

The judge model is duck-typed the way :class:`~memory.summarize.RollingSummarizer`'s
is: an :class:`llm_client.LLMClient` (structured output, retries, token
accounting) or any LangChain-style ``invoke`` object. An unusable reply is a
warning and zero candidates, never an exception — a memory write must not be
able to fail a translation run.

Logger name: ``memory.extractor``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Instruction-shaped language. Deliberately over-inclusive: a false positive
# costs one classifier call that returns nothing, a false negative loses the
# memory entirely.
DEFAULT_CUES: tuple[str, ...] = (
    r"from now on",
    r"going forward",
    r"always",
    r"never",
    r"make sure",
    r"be sure to",
    r"don'?t",
    r"do not",
    r"stop\b",
    r"instead of",
    r"prefer",
    r"rather than",
    r"remember",
    r"keep in mind",
    r"note that",
    r"exception",
    r"approved",
    r"authorized",
    r"authorised",
    r"just this (?:once|time)",
    r"for this (?:thread|conversation|run|call|session)",
    r"only this",
    r"policy",
    r"\brule\b",
    r"standard(?:ly)?\b",
    r"escalate",
)

_EXTRACTION_PROMPT = """\
You maintain an agent's memory. Read one conversation turn and decide what,
if anything, should be remembered as an instruction for future work.

Two kinds of memory exist:
- "permanent": a standing rule for this task, true beyond this conversation
  ("always verify account ownership before discussing billing").
- "temporary": an instruction or exception scoped to THIS conversation only
  ("exception approved by supervisor", "the customer prefers email",
  "do not offer the standard discount on this call").

Rules:
- Extract only instructions that change how future responses should behave.
  Questions, facts about the work itself, and one-off requests that the turn
  already satisfied are NOT memories.
- Anything phrased as an exception, an approval, or as applying to this
  conversation/customer/run is "temporary", never "permanent".
- Rewrite each memory as one short imperative sentence, standing on its own
  without the conversation.
- Extract nothing rather than guess. An empty list is a valid answer.

User turn:
{user_text}

Assistant turn:
{response_text}

Reply with JSON:
{{"memories": [{{"text": "...", "scope": "permanent|temporary",
"kind": "preference|exception|constraint|note", "reason": "..."}}]}}
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class MemoryCandidate(BaseModel):
    """One extracted memory, before it is written anywhere."""

    text: str = ""
    scope: Literal["permanent", "temporary"] = "temporary"
    kind: str = "note"
    reason: str = ""


class Extraction(BaseModel):
    """The classifier's reply schema."""

    memories: list[MemoryCandidate] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """What one :meth:`MemoryExtractor.observe` call did.

    Fields
    ------
    applied
        Temporary memories written to the thread's short-term memory, as
        rendered texts.
    pending
        Permanent candidates held for approval, as stored records
        (``id`` is the handle :meth:`MemoryExtractor.approve` takes).
    promoted
        Permanent memories written straight to the task policy — only
        non-empty under ``auto_apply_permanent=True``.
    gated
        True when the cue gate skipped the turn without an LLM call.
    """

    applied: list[str] = Field(default_factory=list)
    pending: list[dict[str, Any]] = Field(default_factory=list)
    promoted: list[str] = Field(default_factory=list)
    gated: bool = False


class MemoryExtractor:
    """
    Classifies conversation turns into long- and short-term memory.

    Parameters
    ----------
    model : Any
        The classifier: an :class:`llm_client.LLMClient` (asked for the
        :class:`Extraction` schema) or any LangChain-style ``invoke`` object
        (whose reply text is parsed as JSON).
    policy : memory.policy.TaskPolicy | None
        Where approved permanent memories go. ``None`` means permanent
        candidates are still recorded as pending but cannot be approved.
    thread_memory : memory.thread_mem.ThreadMemory | None
        Where temporary memories go. ``None`` disables the temporary half.
    store : Any | None
        ``get``/``set``/``delete`` store for the pending list. ``None`` keeps
        it process-local; ``SasLLMPipeline`` injects its own ``memory.kv``.
    cues : Iterable[str] | None
        Regex fragments that open the gate, matched case-insensitively
        against the user turn. ``None`` uses :data:`DEFAULT_CUES`; an empty
        iterable disables gating (every turn reaches the model).
    ttl_s : float | None
        Lifetime given to each written temporary note. ``None`` (default)
        defers to the ``ThreadMemory``'s own default.
    auto_apply_permanent : bool
        Write permanent candidates straight into the policy instead of
        holding them for approval. Off by default — see the module docstring.
    max_memories : int
        Cap on how many candidates one turn may produce, so a confused
        classifier cannot flood a thread.
    """

    def __init__(
        self,
        model: Any,
        *,
        policy: Any | None = None,
        thread_memory: Any | None = None,
        store: Any | None = None,
        cues: Iterable[str] | None = None,
        ttl_s: float | None = None,
        auto_apply_permanent: bool = False,
        max_memories: int = 5,
    ) -> None:
        if max_memories < 1:
            raise ValueError(f"max_memories must be >= 1, got {max_memories}")
        self._model = model
        # Annotated Any rather than `Any | None`: these are duck-typed
        # collaborators (a TaskPolicy / ThreadMemory / KV store, or nothing),
        # and Any already admits None — the union only made every attribute
        # access read as possibly-None to type checkers.
        self.policy: Any = policy
        self.thread_memory: Any = thread_memory
        self.store: Any = store
        self.ttl_s = ttl_s
        self.auto_apply_permanent = auto_apply_permanent
        self.max_memories = max_memories
        patterns = DEFAULT_CUES if cues is None else tuple(cues)
        self._cue_re = (
            re.compile("|".join(patterns), re.IGNORECASE) if patterns else None
        )
        self._local: dict[str, dict[str, Any]] = {}
        self._structured: bool | None = None
        if auto_apply_permanent:
            logger.warning(
                "MemoryExtractor: auto_apply_permanent=True — extracted "
                "standing rules are written to the task policy with no "
                "approval step, and apply to every future thread of this task"
            )
        logger.info(
            f"MemoryExtractor: cues={'off' if self._cue_re is None else len(patterns)}  "
            f"policy={'on' if policy is not None else 'off'}  "
            f"thread_memory={'on' if thread_memory is not None else 'off'}  "
            f"auto_apply_permanent={auto_apply_permanent}  "
            f"store={'external' if store is not None else 'process-local'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self, thread_id: str, user_text: str, response_text: str = ""
    ) -> ExtractionResult:
        """Classify one completed turn and route what it found.

        Called after the turn is accepted (never on an attempt that inline
        validation rolled back), so a discarded answer cannot leave a memory
        behind. Never raises.
        """
        if not self._gate(user_text):
            logger.debug(
                f"observe: thread='{thread_id}' no instruction cue; skipped"
            )
            return ExtractionResult(gated=True)
        try:
            candidates = self._classify(user_text, response_text)
        except Exception:
            logger.warning(
                f"observe: extraction failed  thread='{thread_id}'", exc_info=True
            )
            return ExtractionResult()
        result = ExtractionResult()
        for candidate in candidates:
            if candidate.scope == "temporary":
                text = self._apply_temporary(thread_id, candidate)
                if text:
                    result.applied.append(text)
            elif self.auto_apply_permanent and self.policy is not None:
                self.policy.add(candidate.text, source="MemoryExtractor")
                result.promoted.append(candidate.text)
            else:
                result.pending.append(self._hold(thread_id, candidate))
        if result.applied or result.pending or result.promoted:
            logger.info(
                f"observe: thread='{thread_id}'  applied={len(result.applied)}  "
                f"pending={len(result.pending)}  promoted={len(result.promoted)}"
            )
        return result

    def pending(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        """Permanent candidates awaiting approval, oldest first.

        Restricted to one thread's proposals when *thread_id* is given.
        """
        records = sorted(
            self._all_pending().values(), key=lambda r: r.get("created_at", 0.0)
        )
        if thread_id is None:
            return records
        return [r for r in records if r.get("thread_id") == thread_id]

    def approve(self, candidate_id: str, *, overridable: bool = True) -> Any | None:
        """Promote a pending candidate into the task policy.

        Returns the created :class:`~memory.policy.PolicyInstruction`, or
        ``None`` when the id is unknown or no policy is attached.
        """
        record = self._all_pending().get(candidate_id)
        if record is None:
            logger.warning(f"approve: unknown candidate id {candidate_id!r}")
            return None
        if self.policy is None:
            logger.warning(
                f"approve: candidate {candidate_id!r} cannot be promoted — "
                "no TaskPolicy attached"
            )
            return None
        instruction = self.policy.add(
            record["text"],
            overridable=overridable,
            source="MemoryExtractor",
        )
        self.reject(candidate_id)
        logger.info(
            f"approve: candidate {candidate_id} -> policy instruction "
            f"{instruction.id}  overridable={overridable}"
        )
        return instruction

    def reject(self, candidate_id: str) -> bool:
        """Discard a pending candidate. False when the id is unknown."""
        if self.store is not None:
            return bool(self.store.delete(self._key(candidate_id)))
        return self._local.pop(candidate_id, None) is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gate(self, user_text: str) -> bool:
        if self._cue_re is None:
            return True
        return bool(user_text and self._cue_re.search(user_text))

    def _classify(self, user_text: str, response_text: str) -> list[MemoryCandidate]:
        extraction = self._ask(
            _EXTRACTION_PROMPT.format(
                user_text=user_text.strip() or "(empty)",
                response_text=(response_text or "").strip() or "(empty)",
            )
        )
        if extraction is None:
            return []
        candidates = [
            candidate
            for candidate in extraction.memories
            if candidate.text and candidate.text.strip()
        ]
        if len(candidates) > self.max_memories:
            logger.warning(
                f"_classify: {len(candidates)} candidate(s) exceed "
                f"max_memories={self.max_memories}; keeping the first"
            )
            candidates = candidates[: self.max_memories]
        return candidates

    def _ask(self, prompt: str) -> Extraction | None:
        """One model call for :class:`Extraction`, ``None`` when unusable."""
        text = ""
        if self._supports_structured():
            envelope = self._model.invoke_structured(Extraction, prompt)
            parsed = envelope.get("parsed")
            if isinstance(parsed, Extraction):
                return parsed
            text = str(getattr(envelope.get("raw"), "content", "") or "")
        else:
            reply = self._model.invoke(prompt)
            text = str(getattr(reply, "content", reply))
        result = _parse_json_reply(text)
        if result is None:
            logger.warning(f"_ask: unusable classifier reply: {text[:120]!r}")
        return result

    def _supports_structured(self) -> bool:
        if self._structured is None:
            supports = getattr(self._model, "supports_structured_output", None)
            self._structured = bool(supports(Extraction)) if supports else False
        return self._structured

    def _apply_temporary(self, thread_id: str, candidate: MemoryCandidate) -> str:
        if self.thread_memory is None:
            logger.debug(
                "_apply_temporary: no ThreadMemory attached; dropping "
                f"{candidate.text!r}"
            )
            return ""
        existing = {text.strip().lower() for text in self.thread_memory.texts(thread_id)}
        if candidate.text.strip().lower() in existing:
            logger.debug(
                f"_apply_temporary: thread='{thread_id}' already holds "
                f"{candidate.text!r}; skipping"
            )
            return ""
        note = self.thread_memory.add(
            thread_id,
            candidate.text,
            kind=candidate.kind,
            source="MemoryExtractor",
            ttl_s=self.ttl_s,
        )
        return note.text

    def _hold(self, thread_id: str, candidate: MemoryCandidate) -> dict[str, Any]:
        record = {
            "id": uuid.uuid4().hex[:8],
            "text": candidate.text.strip(),
            "kind": candidate.kind,
            "reason": candidate.reason,
            "thread_id": thread_id,
            "created_at": time.time(),
        }
        if self.store is not None:
            self.store.set(
                self._key(record["id"]),
                record,
                tags=["pending_memory", thread_id],
                source="MemoryExtractor",
            )
        else:
            self._local[record["id"]] = record
        return record

    @staticmethod
    def _key(candidate_id: str) -> str:
        return f"pending::{candidate_id}"

    def _all_pending(self) -> dict[str, dict[str, Any]]:
        if self.store is None:
            return dict(self._local)
        records: dict[str, dict[str, Any]] = {}
        # items_with_prefix is KVMemoryStore's pushed-down prefix scan; a
        # store without it degrades to whatever get/set it does have, which
        # is enough for the process-local path above.
        items = getattr(self.store, "items_with_prefix", None)
        if items is None:
            return records
        for entry in items("pending::"):
            value = entry.get("value")
            if isinstance(value, dict) and value.get("id"):
                records[str(value["id"])] = value
        return records


def _parse_json_reply(text: str) -> Extraction | None:
    """*text* validated as an :class:`Extraction`, or ``None``.

    Tries the raw text, then any fenced block, then the outermost
    brace-delimited span — a classifier that wraps its JSON in prose is still
    understood, and one that answers in prose alone is simply unusable.
    """
    stripped = text.strip()
    candidates = [stripped, *(body.strip() for body in _FENCE_RE.findall(text))]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return Extraction.model_validate(json.loads(candidate))
        except Exception:
            continue
    return None


def cue_hit(text: str, cues: Sequence[str] = DEFAULT_CUES) -> bool:
    """Whether *text* carries instruction-shaped language.

    Exposed for callers that want the gate without the extractor — e.g. a
    UI that offers "save this as a rule?" only when it would have fired.
    """
    if not cues:
        return True
    return bool(text and re.search("|".join(cues), text, re.IGNORECASE))
