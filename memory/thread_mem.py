"""Short-term memory — conversation-scoped notes, exceptions and overrides.

The counterpart to :mod:`memory.policy`: instructions that hold *for this
thread only*, discovered while it runs. "Customer prefers email", "exception
approved by supervisor", "do not offer the standard discount". They are not
task policy — writing them there would leak one conversation's exception into
every other — and they are not conversation history either: they must survive
the recency window, relevance selection, and summarization untouched.

Stored under ``tmem::{thread_id}`` in a KV store as one record holding the
thread's notes. Two properties follow from where the notes are prompted:

* **Prompted, never persisted to the message history.** Notes ride the
  pipeline's ephemeral ``instructions`` channel — the same rule reference
  guidance and the rolling summary follow (Architecture.md invariant 5).
  They must stay out of ``RelevantHistorySelector``'s scoring, and out of the
  cached system-prompt prefix: a note changes per thread, and folding it into
  the cached block would cost a cache miss on every thread.
* **Expiring.** ``ttl_s`` bounds a note's life ("do not offer the discount
  *on this call*"). Expired notes are filtered on read and purged on the next
  write, so a stale override cannot outlive the exception that justified it.

The ``store`` is duck-typed (``get``/``set``/``delete``) as in
:mod:`memory.policy` and :mod:`memory.summarize`; without one, state is
process-local.

Logger name: ``memory.thread_mem``.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Free-form, but these four are what the extractor emits and what the
# override-compliance metric knows how to phrase a check for.
NOTE_KINDS = ("preference", "exception", "constraint", "note")


class ThreadNote(BaseModel):
    """One conversation-scoped instruction.

    Fields
    ------
    id
        Stable short identifier, the handle :meth:`ThreadMemory.remove` takes.
    text
        The note itself, as it will be prompted.
    kind
        One of :data:`NOTE_KINDS` — advisory only, but it is what the rendered
        block groups by and what ``override_compliance`` scores against.
    source
        Provenance: ``"operator"``, ``"MemoryExtractor"``, a supervisor name.
    created_at, expires_at
        Unix timestamps. ``expires_at is None`` means the note lives as long
        as the thread does.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    kind: str = "note"
    source: str | None = None
    created_at: float = Field(default_factory=time.time)
    expires_at: float | None = None

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


class ThreadMemory:
    """
    Per-thread notes, keyed ``tmem::{thread_id}``.

    One store record per thread (not per note): a thread carries a handful of
    notes and every read wants all of them, so a single row is one KV get
    instead of a prefix scan — the same trade
    :class:`~memory.summarize.RollingSummarizer` makes for summaries.

    Parameters
    ----------
    store : Any | None
        ``get``/``set``/``delete``; :class:`memory.store.KVMemoryStore` fits.
        ``None`` (default) keeps notes process-local; ``SasLLMPipeline``
        injects its own ``memory.kv`` into a store-less instance.
    default_ttl_s : float | None
        Applied to :meth:`add` calls that pass no ``ttl_s``. ``None``
        (default) means notes do not expire on their own.

    Usage
    -----
        notes = ThreadMemory(store=mem.kv)
        notes.add("thread-1", "Customer prefers email.", kind="preference")
        notes.add("thread-1", "Do not offer the standard discount.",
                  kind="exception", source="supervisor", ttl_s=3600)
        print(notes.render("thread-1"))
    """

    def __init__(
        self,
        *,
        store: Any | None = None,
        default_ttl_s: float | None = None,
    ) -> None:
        if default_ttl_s is not None and default_ttl_s <= 0:
            raise ValueError(f"default_ttl_s must be > 0, got {default_ttl_s}")
        self.store = store
        self.default_ttl_s = default_ttl_s
        self._local: dict[str, list[dict[str, Any]]] = {}
        logger.info(
            f"ThreadMemory: default_ttl_s={default_ttl_s}  "
            f"store={'external' if store is not None else 'process-local'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        thread_id: str,
        text: str,
        *,
        kind: str = "note",
        source: str | None = None,
        ttl_s: float | None = None,
    ) -> ThreadNote:
        """Append one note to *thread_id* and persist. Returns what was stored.

        Adding also purges anything that has expired, so an abandoned thread
        that is written to again does not carry dead overrides forward.
        """
        clean = text.strip()
        if not clean:
            raise ValueError("note text must be non-empty")
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        note = ThreadNote(
            text=clean,
            kind=kind,
            source=source,
            expires_at=(time.time() + ttl) if ttl else None,
        )
        live = self.notes(thread_id)
        self._save(thread_id, [*live, note])
        logger.info(
            f"ThreadMemory.add: thread='{thread_id}'  id={note.id}  kind={kind}  "
            f"ttl={'-' if ttl is None else f'{ttl:.0f}s'}  "
            f"{len(live) + 1} live note(s)"
        )
        return note

    def notes(self, thread_id: str, *, now: float | None = None) -> list[ThreadNote]:
        """*thread_id*'s live notes, oldest first. Expired ones are filtered
        out here and purged on the next write."""
        moment = now if now is not None else time.time()
        live = []
        expired = 0
        for raw in self._load(thread_id):
            try:
                note = ThreadNote(**raw)
            except Exception:
                logger.warning(
                    f"notes: thread='{thread_id}' has an unreadable note "
                    f"record; skipping it"
                )
                continue
            if note.expired(moment):
                expired += 1
                continue
            live.append(note)
        if expired and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"notes: thread='{thread_id}' dropped {expired} expired note(s)"
            )
        return live

    def texts(self, thread_id: str) -> list[str]:
        """Live note texts — what ``validation.EvaluationRun.thread_notes``
        carries so a scored run knows which overrides were in force."""
        return [note.text for note in self.notes(thread_id)]

    def remove(self, thread_id: str, note_id: str) -> bool:
        """Drop one note by id. False when the thread has no such note."""
        live = self.notes(thread_id)
        kept = [note for note in live if note.id != note_id]
        if len(kept) == len(live):
            return False
        self._save(thread_id, kept)
        logger.info(f"ThreadMemory.remove: thread='{thread_id}'  id={note_id}")
        return True

    def clear(self, thread_id: str) -> None:
        """Delete every note for *thread_id*."""
        if self.store is not None:
            self.store.delete(self._key(thread_id))
        else:
            self._local.pop(thread_id, None)
        logger.info(f"ThreadMemory.clear: thread='{thread_id}'")

    def fork(self, src_thread_id: str, dst_thread_id: str) -> int:
        """Copy *src*'s live notes onto *dst*, returning how many were copied.

        The branch half of ``memory.store``'s thread fork: a rewound
        conversation continues under the same exceptions it was granted —
        losing "exception approved by supervisor" on a fork would silently
        change what the branch is allowed to do. Expired notes are not
        carried, and *dst*'s existing notes are replaced.
        """
        live = self.notes(src_thread_id)
        self._save(dst_thread_id, live)
        logger.info(
            f"ThreadMemory.fork: copied {len(live)} note(s) "
            f"'{src_thread_id}' -> '{dst_thread_id}'"
        )
        return len(live)

    def render(self, thread_id: str) -> str:
        """The thread's notes as a prompt block, or ``""`` when it has none.

        Precedence against the task policy is *not* stated here — that is
        :class:`memory.context.MemoryContext`'s job, which is the only place
        that sees both sides.
        """
        live = self.notes(thread_id)
        if not live:
            return ""
        lines = [
            f"- [{note.kind}] {note.text}"
            + (f" (source: {note.source})" if note.source else "")
            for note in live
        ]
        return (
            "Instructions specific to this conversation:\n" + "\n".join(lines)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"tmem::{thread_id}"

    def _load(self, thread_id: str) -> list[dict[str, Any]]:
        if self.store is not None:
            stored = self.store.get(self._key(thread_id), None)
        else:
            stored = self._local.get(thread_id)
        if not isinstance(stored, list):
            return []
        return [raw for raw in stored if isinstance(raw, dict)]

    def _save(self, thread_id: str, notes: list[ThreadNote]) -> None:
        payload = [note.model_dump() for note in notes]
        if self.store is not None:
            self.store.set(
                self._key(thread_id),
                payload,
                tags=["thread_memory", thread_id],
                source="ThreadMemory",
            )
        else:
            self._local[thread_id] = payload
