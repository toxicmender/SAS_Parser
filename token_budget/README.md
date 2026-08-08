# token_budget

How many tokens a piece of text costs, answered in one place.

Dependency-free (stdlib + optional `tiktoken`) and imported by `llm_client`,
`pipeline`, and `prompt_builder` — which is why it is its own package rather
than living inside any of them.

## Why it is a package

It used to be `llm_client/tokens.py`. Everything that budgets prompt space
needs to count tokens, but importing `llm_client.tokens` executes
`llm_client/__init__.py`, which imports the chat-model stack:

| import | time | modules | langchain |
|---|---|---|---|
| `llm_client.tokens` | 7.5s | 1,642 | yes |
| `token_budget` | 0.1s | 77 | no |

`prompt_builder` is deliberately free of `llm_client` — it has to be usable,
and testable, without a configured gateway — so counting tokens there was
impossible until this moved. Same reasoning that made `target_language` its own
package.

## Use

```python
import token_budget

token_budget.count_text("SELECT 1")                    # 3
token_budget.count_text(sql, model="gpt-5.4")          # under that model's encoding
token_budget.count_messages(messages, model=model)     # + ChatML framing
token_budget.encoding_name_for_model("claude-opus-5")  # "o200k_base"
```

## What it promises, and does not

An **estimate that tracks the real count**, not a billing figure. Encoding is
resolved by explicit prefix map — never `tiktoken.encoding_for_model`, whose
lookup raises `KeyError` for names it does not know — so an unknown or
non-OpenAI model id can never fail a call mid-run. Anything unrecognised
(Claude, Gemini, a future id) counts under `o200k_base`: a real tokenizer under
a stand-in vocabulary, far closer than a `chars // 4` guess.

Loading an encoding needs tiktoken's BPE data, fetched over the network on
first use. Where that fails — offline, a blocking proxy — every counter
degrades to ~4 chars/token with a one-time WARNING per encoding, and the
failure is cached so later calls do not re-pay the attempt.

## Who counts what

| Caller | Budget |
|---|---|
| `llm_client` | the request's input-token budget, and usage accounting |
| `pipeline.prompting` | `max_merged_tokens` batch packing |
| `prompt_builder` | `max_instruction_tokens` retrieval budget, and the instruction-chunker window |

`chunker` is the deliberate exception: `sas_chunker.min_words`/`max_words` size
SAS *source* into semantic units, which is not a prompt-cost question. The
batcher then packs those units by token cost on top.

Logger name: `token_budget`.
