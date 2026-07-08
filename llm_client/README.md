# llm_client

LangChain chat-model construction and invocation. Owns everything about *how*
the LLM is called, so callers (currently only `chunker.pipeline`) never touch
`init_chat_model` or provider error types directly.

The package imports nothing from `chunker` or `memory`.

## Quick start

```python
from llm_client import LLMClient, LLMClientConfig

client = LLMClient(LLMClientConfig(
    model="claude-haiku-4-5-20251001",
    temperature=0.2,
    max_input_tokens=100_000,
    requests_per_second=2.0,
))
response = client.invoke("translate this DATA step ...")

# or inside a LCEL chain, in place of the raw model:
chain = prompt | client.as_runnable()
```

## What it owns

- **Construction** via `langchain.chat_models.init_chat_model` with an optional
  `temperature`, output-token cap, and a proactive
  `langchain_core.rate_limiters.InMemoryRateLimiter` that throttles request
  *starts* client-side.
- **Reactive rate-limit handling**: HTTP 429 / rate-limit-shaped errors are
  retried with capped exponential backoff and jitter (attempt *n* waits
  `min(base * 2**(n-1), max)` scaled by 0.5–1.5× jitter); every other exception
  propagates unchanged on the first occurrence.
- **Input-token budget**: when `max_input_tokens` is set, the prompt is counted
  before the call and `InputTokenLimitError` is raised instead of sending an
  over-budget request. Counting uses the model's own
  `get_num_tokens_from_messages`, falling back to a `chars // 4` approximation
  (with a one-time WARNING) when that is unavailable — e.g. offline, or for fake
  models without a tokenizer.

An injected `llm` (e.g. a fake in tests) still gets the retry and input-token
layers; `temperature` / `max_output_tokens` / `requests_per_second` do not
apply to it, since those attach at construction time.

## Public API

| Name | Purpose |
|------|---------|
| `LLMClient` | Thin invocation layer around a LangChain chat model. Use `invoke()` directly, or `as_runnable()` to drop it into a LCEL chain. |
| `LLMClientConfig` | Declarative knobs (model, temperature, token caps, rate limiter, retry schedule, custom token counter). |
| `InputTokenLimitError` | Raised when a prompt exceeds `max_input_tokens`; nothing is sent. |

## Logging

Logger name: `llm_client.client`

| Level | When emitted |
|-------|--------------|
| DEBUG | Token-count results per call |
| INFO | Model construction |
| WARNING | Rate-limit retry waits; token-counter fallback (once) |
| ERROR | Input-token budget exceeded (exception raised after logging) |
