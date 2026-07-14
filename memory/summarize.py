"""Rolling thread summarization — the "compress" half of context management.

Selection (``memory.relevance``) decides which turns are prompted verbatim;
this module guarantees a floor of information about everything else: once
the turns older than a recency tail exceed a token threshold, they are
folded — monotonically, oldest first — into one running summary per thread.
The summary is stored in the KV layer (never the ``msg::`` history), so it
is prompted-but-not-persisted context, re-derivable at any time from the
full stored thread.

Coverage is positional (a prefix of the thread's turns), not
selection-based: what the relevance selector drops varies per query, so
summarizing "dropped" turns would re-summarize the same content endlessly.
A monotonic prefix is summarized exactly once, and the selector remains
free to surface any covered turn verbatim when it is relevant again.

Logger name: ``memory.summarize``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import BaseMessage, SystemMessage

from .turns import approx_token_count, group_turns, turn_text

logger = logging.getLogger(__name__)


_DEFAULT_PROMPT = """\
You maintain a running summary of an ongoing technical conversation.
Fold the new turns into the summary. Preserve exact identifiers (dataset,
table, macro, file, function, and variable names), naming conventions,
and decisions made — later work depends on recalling them precisely.
Drop conversational filler. Keep the result under {max_words} words.

Current summary (may be empty):
{summary}

New conversation turns:
{turns}

Updated summary:"""


class RollingSummarizer:
    """
    Maintain one rolling summary per thread, stored under
    ``summary::{thread_id}`` in a KV store.

    Parameters
    ----------
    model : Any
        What produces the summary text: a LangChain chat model (anything
        with ``.invoke(str)`` returning a message-like object), or a plain
        ``Callable[[str], str]``.
    store : Any | None
        Where summaries persist: anything with ``get(key, default)`` /
        ``set(key, value, tags=...)`` / ``delete(key)`` —
        :class:`memory.store.KVMemoryStore` fits (duck-typed on
        purpose: this module imports neither store nor relevance).
        ``None`` (default) keeps summaries in a process-local dict;
        ``SasLLMPipeline`` injects its own KV store into a store-less
        summarizer at construction.
    trigger_tokens : int
        Fold pending turns into the summary only once their combined
        token estimate reaches this threshold — batching LLM calls
        instead of summarizing every turn.
    keep_last_turns : int
        Newest turns never folded (they are prompt-eligible verbatim and
        still moving); folding eligibility ends this many turns from the
        end of the thread.
    max_summary_words : int
        Word budget given to the summarization prompt.
    token_counter : Callable[[str], int] | None
        Counter for the trigger threshold. ``None`` (default) uses the
        offline ~4-chars/token estimate.
    prompt_template : str | None
        Override the summarization prompt; must accept ``{summary}``,
        ``{turns}``, and ``{max_words}`` placeholders.
    """

    def __init__(
        self,
        model: Any,
        *,
        store: Any | None = None,
        trigger_tokens: int = 2048,
        keep_last_turns: int = 4,
        max_summary_words: int = 300,
        token_counter: Callable[[str], int] | None = None,
        prompt_template: str | None = None,
    ) -> None:
        if trigger_tokens < 1:
            raise ValueError(f"trigger_tokens must be >= 1, got {trigger_tokens}")
        if keep_last_turns < 0:
            raise ValueError(f"keep_last_turns must be >= 0, got {keep_last_turns}")
        self._model = model
        self.store = store
        self.trigger_tokens = trigger_tokens
        self.keep_last_turns = keep_last_turns
        self.max_summary_words = max_summary_words
        self._count_tokens = token_counter or approx_token_count
        self._prompt_template = prompt_template or _DEFAULT_PROMPT
        self._local: dict[str, dict[str, Any]] = {}
        logger.info(
            f"RollingSummarizer: trigger_tokens={trigger_tokens}  "
            f"keep_last_turns={keep_last_turns}  "
            f"max_summary_words={max_summary_words}  "
            f"store={'external' if store is not None else 'process-local'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(
        self, thread_id: str, history: list[BaseMessage]
    ) -> SystemMessage | None:
        """
        Bring *thread_id*'s summary up to date with *history* and return it
        as a single SystemMessage to prepend to the prompt — or ``None``
        while no summary exists yet.

        Folding is incremental: only turns past the stored coverage mark
        and older than the ``keep_last_turns`` tail are candidates, and
        they are folded (one LLM call) only once they jointly reach
        ``trigger_tokens``.
        """
        turns = group_turns(history)
        state = self._load(thread_id)
        if state["covered_turns"] > len(turns):
            # The thread shrank under us (cleared or forked): the stored
            # summary describes turns that no longer exist. Start over.
            logger.warning(
                f"refresh: thread '{thread_id}' has {len(turns)} turn(s) but "
                f"summary covered {state['covered_turns']}; resetting summary"
            )
            state = {"summary": "", "covered_turns": 0}
            self._save(thread_id, state)

        frontier = max(len(turns) - self.keep_last_turns, 0)
        if frontier > state["covered_turns"]:
            pending = turns[state["covered_turns"] : frontier]
            pending_text = "\n\n".join(turn_text(t) for t in pending)
            pending_tokens = self._count_tokens(pending_text)
            if pending_tokens >= self.trigger_tokens:
                logger.info(
                    f"refresh: folding turns "
                    f"[{state['covered_turns']}, {frontier}) of thread "
                    f"'{thread_id}' into summary (~{pending_tokens} tokens)"
                )
                state = {
                    "summary": self._summarize(state["summary"], pending_text),
                    "covered_turns": frontier,
                }
                self._save(thread_id, state)

        if not state["summary"]:
            return None
        return SystemMessage(
            content=(
                f"Summary of the {state['covered_turns']} earliest "
                f"conversation turn(s) (their full text may be omitted "
                f"below):\n{state['summary']}"
            )
        )

    def reset(self, thread_id: str) -> None:
        """Discard the stored summary for *thread_id*."""
        if self.store is not None:
            self.store.delete(self._key(thread_id))
        self._local.pop(thread_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"summary::{thread_id}"

    def _load(self, thread_id: str) -> dict[str, Any]:
        if self.store is not None:
            stored = self.store.get(self._key(thread_id), None)
        else:
            stored = self._local.get(thread_id)
        if not isinstance(stored, dict):
            return {"summary": "", "covered_turns": 0}
        return {
            "summary": stored.get("summary", ""),
            "covered_turns": int(stored.get("covered_turns", 0)),
        }

    def _save(self, thread_id: str, state: dict[str, Any]) -> None:
        if self.store is not None:
            self.store.set(
                self._key(thread_id),
                state,
                tags=["summary", thread_id],
                source="RollingSummarizer",
            )
        else:
            self._local[thread_id] = dict(state)

    def _summarize(self, existing: str, new_turns: str) -> str:
        prompt = self._prompt_template.format(
            summary=existing or "(empty)",
            turns=new_turns,
            max_words=self.max_summary_words,
        )
        if hasattr(self._model, "invoke"):
            result = self._model.invoke(prompt)
            text = getattr(result, "content", result)
        else:
            text = self._model(prompt)
        return str(text).strip()
