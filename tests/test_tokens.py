"""Unit tests for llm_client.tokens (encoding resolution, counting,
degradation) and the memory.turns default counter.

No test here touches the network: real-encoding tests stub the loaded
encoding, and fallback tests force the load to fail — matching the repo rule
that the suite runs offline.
"""

import logging
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client import tokens
from memory import turns


class _WordEncoding:
    """Stub tiktoken encoding: one token per whitespace-separated word."""

    def encode(self, text: str) -> list[str]:
        return text.split()


# ---------------------------------------------------------------------------
# Encoding resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        # The concrete motivating case: tiktoken's own model table keys on
        # "gpt-5-" (dash) and would raise KeyError for the dotted id.
        ("gpt-5.4", "o200k_base"),
        ("gpt-5", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("gpt-4.1-mini", "o200k_base"),
        ("o1-preview", "o200k_base"),
        ("o3", "o200k_base"),
        # Older GPT families keep their real vocabulary.
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        # Non-OpenAI and unknown ids get the o200k stand-in.
        ("claude-sonnet-4-5", "o200k_base"),
        ("anthropic:claude-opus-4-6", "o200k_base"),
        ("gemini-3.1-pro", "o200k_base"),
        ("some-future-model", "o200k_base"),
        ("", "o200k_base"),
        (None, "o200k_base"),
    ],
)
def test_encoding_name_for_model(model, expected):
    assert tokens.encoding_name_for_model(model) == expected


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_count_text_uses_the_resolved_encoding(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", lambda name: _WordEncoding())
    assert tokens.count_text("alpha beta gamma", model="gpt-5.4") == 3


def test_count_text_falls_back_to_chars_over_four(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", lambda name: None)
    assert tokens.count_text("x" * 40) == 10


def test_count_messages_adds_framing_and_reply_primer(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", lambda name: _WordEncoding())
    messages = [HumanMessage("one two"), HumanMessage("three")]
    # 3 reply primer + (3 framing + 2) + (3 framing + 1)
    assert tokens.count_messages(messages) == 12


def test_count_messages_reads_text_content_parts(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", lambda name: _WordEncoding())
    message = SystemMessage(
        content=[
            {"type": "text", "text": "one two three"},
            {"type": "image_url", "image_url": {"url": "https://x"}},
        ]
    )
    # 3 reply primer + 3 framing + 3 text tokens; the non-text part is free.
    assert tokens.count_messages([message]) == 9


def test_count_messages_accepts_plain_strings(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", lambda name: _WordEncoding())
    assert tokens.count_messages(["one two"]) == 8  # 3 + 3 + 2


def test_unloadable_encoding_is_cached_and_warned_once(monkeypatch, caplog):
    def _boom(name):
        raise OSError("no network")

    import tiktoken as tiktoken_mod

    monkeypatch.setattr(tokens, "_encodings", {})
    monkeypatch.setattr(tokens, "_warned_encodings", set())
    monkeypatch.setattr(tiktoken_mod, "get_encoding", _boom)

    with caplog.at_level(logging.WARNING, logger="llm_client.tokens"):
        assert tokens.count_text("x" * 8) == 2
        assert tokens.count_text("x" * 8) == 2

    warnings = [r for r in caplog.records if "could not load" in r.message]
    assert len(warnings) == 1  # cached failure: one fetch attempt, one warning


# ---------------------------------------------------------------------------
# memory.turns default counter
# ---------------------------------------------------------------------------


def test_turns_token_count_uses_tiktoken_when_available(monkeypatch):
    monkeypatch.setattr(turns, "_encoding", _WordEncoding())
    assert turns.token_count("alpha beta gamma") == 3


def test_turns_token_count_falls_back_to_approximation(monkeypatch):
    monkeypatch.setattr(turns, "_encoding", False)
    assert turns.token_count("x" * 40) == turns.approx_token_count("x" * 40)
