# llm_client

LangChain chat-model construction and invocation. Owns everything about *how*
the LLM is called, so callers (currently only `chunker.pipeline`) never touch
`init_chat_model` or provider error types directly.

The package imports nothing from `chunker` or `memory`.

## Quick start

```python
from llm_client import LLMClient, LLMClientConfig

client = LLMClient(LLMClientConfig(
    model="claude-sonnet-4-5",   # alias: model_name=
    temperature=0.2,
    max_input_tokens=100_000,
    requests_per_second=2.0,
    # endpoint overrides (all optional; also settable via config.json):
    base_url="https://llm-gateway.example/v1",
    api_key="...",                       # SecretStr — masked in repr/logs
    url_headers={"X-Team": "sas"},       # sent as default_headers
    timeout=60.0,
    cert_file="certs/gateway.crt",       # TLS trust for the gateway
    model_kwargs={"top_k": 40},          # provider request-body extras
    kwargs={},                           # escape hatch, merged last
))
response = client.invoke("translate this DATA step ...")
response = await client.ainvoke("translate this DATA step ...")  # async twin

# or inside a LCEL chain, in place of the raw model
# (chain.invoke and chain.ainvoke both work):
chain = prompt | client.as_runnable()
```

## What it owns

- **Construction** via `langchain.chat_models.init_chat_model` with optional
  `temperature`, output-token cap, endpoint overrides (`base_url`, `api_key`,
  `url_headers` → `default_headers`, `timeout`, `model_kwargs`, plus a raw
  `kwargs` escape hatch merged last), and a proactive
  `langchain_core.rate_limiters.InMemoryRateLimiter` that throttles request
  *starts* client-side. `api_key` is a `SecretStr`: masked in `repr`, never
  logged, and deliberately not readable from config.json.
- **Gateway TLS trust**: `cert_file` names a PEM certificate bundle (e.g.
  `gateway.crt`) used to verify the endpoint's TLS certificate — needed when
  `base_url` points at a gateway signed by an internal CA. It is exported as
  the standard `SSL_CERT_FILE` environment variable (process-wide) before the
  model is built, which the httpx-based provider SDKs honour; there is no
  per-model hook, as e.g. `ChatAnthropic` builds its HTTP client internally.
  A missing file is skipped with a WARNING and the default trust store
  applies.
- **Transient-error handling**: rate limits (HTTP 429), overload / server
  errors (500, 502, 503, 504, 529), timeouts, and connection drops are retried
  with capped exponential backoff and jitter (attempt *n* waits
  `min(base * 2**(n-1), max)` scaled by 0.5–1.5× jitter); every other exception
  propagates unchanged on the first occurrence, and exhausted retries are
  logged at ERROR before the last exception propagates. Callers that persist
  progress (e.g. `chunker.pipeline`'s per-item run facts + `resume=True`)
  therefore only ever see failures that survived the retry budget.
- **Input-token budget**: when `max_input_tokens` is set, the prompt is counted
  before the call and `InputTokenLimitError` is raised instead of sending an
  over-budget request. Counting uses the model's own
  `get_num_tokens_from_messages`, falling back to a `chars // 4` approximation
  (with a one-time WARNING) when that is unavailable — e.g. offline, or for fake
  models without a tokenizer.
- **Sync and async invocation**: `invoke` / `ainvoke` share the same budget and
  retry semantics (`ainvoke` backs off with `asyncio.sleep` and counts tokens in
  a worker thread, since model-native counters may call a sync HTTP endpoint);
  `as_runnable()` binds both, so the LCEL chain works under `invoke` and
  `ainvoke` alike.

Every `LLMClientConfig` knob except `api_key`, `kwargs`, `token_counter`, and
the rate-limiter/backoff shape can also be defaulted from the `llm_client`
section of config.json (precedence: explicit argument > config.json > hard
default — see the `app_config` package). File values are parsed through
`app_config.llm_client_value`, which type-checks them against the section
schema: a wrong-typed entry (e.g. `"timeout": "sixty"`) is ignored with a
WARNING and the hard default applies, instead of failing construction.

An injected `llm` (e.g. a fake in tests) still gets the retry and input-token
layers; construction-time knobs (`temperature`, `max_output_tokens`,
`base_url`, `api_key`, `url_headers`, `timeout`, `cert_file`, `model_kwargs`,
`kwargs`, `requests_per_second`) do not apply to it.

## Public API

| Name | Purpose |
|------|---------|
| `LLMClient` | Thin invocation layer around a LangChain chat model. Use `invoke()` / `ainvoke()` directly, or `as_runnable()` to drop it into a LCEL chain (sync and async). |
| `LLMClientConfig` | Declarative knobs (model, endpoint overrides, temperature, token caps, rate limiter, retry schedule, custom token counter, provider kwargs). |
| `InputTokenLimitError` | Raised when a prompt exceeds `max_input_tokens`; nothing is sent. |

## Logging

Logger name: `llm_client.client`

| Level | When emitted |
|-------|--------------|
| DEBUG | Token-count results per call |
| INFO | Model construction (api_key presence and header *names* only — never values) |
| WARNING | Transient-error retry waits; token-counter fallback (once); `cert_file` missing or overriding a pre-set `SSL_CERT_FILE` |
| ERROR | Input-token budget exceeded; retry budget exhausted (exception raised after logging) |
