# llm_client

OpenAI-compatible chat-model construction and invocation. Owns everything about
*how* the LLM is called, so callers (currently only `pipeline.engine`) never
touch the chat-model class or provider error types directly.

**One transport, many models.** The AI Gateway speaks the OpenAI
`/chat/completions` protocol for *every* model it fronts, including the
Anthropic- and Google-served ones. So this package builds exactly one kind of
client — `langchain_openai.ChatOpenAI`, a thin LangChain wrapper over the
`openai` SDK — and the model name selects the model, not the transport.
`claude-sonnet-4-5`, `gpt-5.4` and `gemini-3.1-pro` all go out the same way; a
leftover LangChain provider prefix (`anthropic:claude-opus-4-6`) is stripped,
since routing is the gateway's job. When no `api_key` is set the SDK's own
`OPENAI_API_KEY` applies, whatever the model.

The package imports nothing from `chunker` or `memory`.

## Quick start

```python
from llm_client import LLMClient, LLMClientConfig

client = LLMClient(LLMClientConfig(
    model="claude-sonnet-4-5",   # alias: model_name=; gateway model id
    temperature=0.2,
    max_input_tokens=100_000,
    requests_per_second=2.0,
    # endpoint overrides (all optional; also settable via config.json):
    base_url="https://llm-gateway.example/v1",
    api_key="...",                       # SecretStr — masked in repr/logs
    url_headers={"X-Team": "sas"},       # sent as default_headers
    timeout=60.0,
    cert_file="certs/gateway.crt",       # TLS trust for the gateway
    model_kwargs={"top_k": 40},          # extra /chat/completions body fields
    kwargs={},                           # escape hatch, merged last
))
response = client.invoke("translate this DATA step ...")
response = await client.ainvoke("translate this DATA step ...")  # async twin

# or inside a LCEL chain, in place of the raw model
# (chain.invoke and chain.ainvoke both work):
chain = prompt | client.as_runnable()
```

## What it owns

- **Construction** of a `langchain_openai.ChatOpenAI` with optional
  `temperature`, output-token cap, endpoint overrides (`base_url`, `api_key`,
  `url_headers` → `default_headers`, `timeout`, `model_kwargs`, plus a raw
  `kwargs` escape hatch merged last), and a proactive
  `langchain_core.rate_limiters.InMemoryRateLimiter` that throttles request
  *starts* client-side. `api_key` is a `SecretStr`: masked in `repr`, never
  logged, and deliberately not readable from config.json. Two settings are
  pinned (both overridable via `kwargs`): `max_retries=0`, so the `openai`
  SDK's retry layer cannot re-send a 429 out of sight of the handling below,
  and `use_responses_api=False`, since a compatible gateway generally
  implements `/chat/completions` only.
- **Gateway addressing**: a `base_url` ending in `/` marks a per-deployment
  route — the resolved model id is appended as the final path segment
  (`https://gw.example/openai/` + `gpt-5.4`) — for a gateway that routes by URL
  rather than by request body. Without a trailing slash the URL is used
  verbatim. The `api_key` goes out twice: as the SDK's bearer credential and
  mirrored into an `api-key` header, which is what a gateway fronting Azure
  OpenAI authenticates on. An `api-key` you set yourself via `url_headers`
  wins; neither copy is ever logged, since the construction log records header
  *names* only.
- **Gateway TLS trust**: `cert_file` names a PEM certificate bundle (e.g.
  `gateway.crt`) used to verify the endpoint's TLS certificate — needed when
  `base_url` points at a gateway signed by an internal CA. It is applied two
  ways: exported as the standard `SSL_CERT_FILE` environment variable
  (process-wide) before the model is built, and passed as an explicit
  `http_client` / `http_async_client` pair pinned to that bundle. Belt and
  braces — the export alone would be at the mercy of anything else in the
  process rewriting it, and the explicit clients alone would not cover code
  reaching the gateway outside this model. Both httpx clients are set, since
  the sync one alone would leave `ainvoke` verifying against the default trust
  store. A missing file is skipped with a WARNING and the default trust store
  applies; if httpx cannot build the clients, the `SSL_CERT_FILE` export
  stands alone (also a WARNING).
- **Transient-error handling**: rate limits (HTTP 429), overload / server
  errors (500, 502, 503, 504, 529), timeouts, and connection drops are retried.
  When the gateway sends its own timing — a `Retry-After` (delta-seconds or
  HTTP-date) or `retry-after-ms` header on the response — that wait is
  **honored verbatim** (no jitter, replacing the backoff), bounded by
  `retry_after_max_seconds` (default 300s) so a bad header can't hang the run;
  otherwise the client falls back to capped exponential backoff with jitter
  (attempt *n* waits `min(base * 2**(n-1), max)` scaled by 0.5–1.5×). Every
  other exception propagates unchanged on the first occurrence, and exhausted
  retries are logged at ERROR before the last exception propagates. Callers
  that persist progress (e.g. `pipeline.engine`'s per-item run facts +
  `resume=True`) therefore only ever see failures that survived the retry
  budget.
- **Prompt-cache compatibility**: a `cache_control` breakpoint (the
  Anthropic-native content-part key `pipeline.engine` sets on the system
  block when `prompt_caching` is on) is not something every OpenAI-compatible
  gateway forwards — LiteLLM- and OpenRouter-style proxies pass it through to
  Anthropic, a strict OpenAI schema rejects unknown content-part keys with a
  400. Support is therefore settled by *asking*: the breakpoint is sent as-is,
  and if the gateway refuses it (400/422 naming `cache_control`), the client
  strips it, re-sends immediately, and remembers the answer — so the fallback
  costs one failed request per process, not per call, and later calls go out
  pre-stripped. The caller's messages are copied, never edited in place. An
  unrelated 400 is left alone and fails fast, since stripping would not help
  it. A reported cache *hit* is taken as confirmation the breakpoint survived.
- **Token accounting**: every successful call logs `in` / `out` / `total`
  tokens plus cache read and write counts at INFO, and folds them into
  `client.usage` (a `TokenUsage`: per-run totals and a call count). Counts
  come from the response's `usage_metadata`, falling back to the raw
  `response_metadata['token_usage']` block for a gateway whose usage LangChain
  did not normalize. A gateway that reports no usage at all leaves `usage`
  empty and warns once. Malformed counts degrade to zero rather than failing a
  call that already succeeded. The attribute is plain arithmetic, not
  synchronized — a client shared across threads can undercount. `TokenUsage`
  supports `+` and `-`, so a caller that shares one client across several runs
  attributes each run's spend by snapshotting before and subtracting after —
  which is what `ValidationRunner` does. It flows outward from here:
  `SasLLMPipeline.token_usage` → `demo_run`'s run summary and
  `validation/summary.json` → `ValidationReport.token_usage`, rendered into
  `to_markdown()` and so into the PDF.
- **Proactive throttle**: an `InMemoryRateLimiter` paces request *starts* at
  `requests_per_second` (burst `max_bucket_size`) so calls stay under the
  gateway's limit before ever tripping a 429. Off by default;
  `LLMClientConfig.from_ai_gateway(...)` turns it on (`2.0`) for the gateway
  credential path, and `llm_client.requests_per_second` in config.json sets it
  for every path. Attaches at construction, so it does not apply to an
  injected `llm`.
- **Input-token budget**: when `max_input_tokens` is set, the prompt is counted
  before the call and `InputTokenLimitError` is raised instead of sending an
  over-budget request. Counting is client-owned, via the shared tiktoken
  counter in `llm_client.tokens`: the encoding resolves from the model id by
  an explicit prefix map — `o200k_base` for the modern GPT families **and**
  for every non-OpenAI id (Claude, Gemini: a real tokenizer run under a
  stand-in vocabulary — an *estimate*, not that provider's own tokenization,
  but far closer than a guess), `cl100k_base` for the older GPT families —
  never a bare `tiktoken.encoding_for_model` lookup that can raise on an
  unknown name (`"gpt-5.4"` included). An injected `llm` needs no tokenizer
  of its own. Pass `token_counter` when the budget must be exact. When
  tiktoken cannot load its encoding data (offline, a blocking proxy),
  counting degrades to a `chars // 4` approximation with a one-time WARNING.
- **Sync and async invocation**: `invoke` / `ainvoke` share the same budget and
  retry semantics (`ainvoke` backs off with `asyncio.sleep` and counts tokens in
  a worker thread, since model-native counters may call a sync HTTP endpoint);
  `as_runnable()` binds both, so the LCEL chain works under `invoke` and
  `ainvoke` alike.
- **Structured output**: `invoke_structured(schema, ...)` is `invoke` with a
  schema-bound model, so the input budget, retries, `cache_control` fallback,
  and usage accounting all still apply. It always binds `include_raw=True` and
  returns LangChain's `{"raw", "parsed", "parsing_error"}` envelope — the
  caller needs `raw` for token usage, and needs a model that ignored the schema
  to arrive as `parsing_error` rather than an exception, since a usable prose
  answer is still in `raw`. `supports_structured_output(schema)` answers the
  binding question up front (an injected stub, or an integration without
  `with_structured_output`, says no) so a caller can pick its prompt before the
  first call rather than failing mid-run. `as_structured_runnable(schema)` is
  the LCEL form.

Every `LLMClientConfig` knob except `api_key`, `kwargs`, `token_counter`, and
the backoff shape (`retry_base_seconds` / `retry_max_seconds` /
`retry_after_max_seconds`) can also be defaulted from the `llm_client`
section of config.json — including `requests_per_second` and `max_bucket_size`
(precedence: explicit argument > config.json > hard default — see the
`app_config` package). File values are parsed through
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
| `LLMClient` | Thin invocation layer around an OpenAI-compatible chat model. Use `invoke()` / `ainvoke()` directly, or `as_runnable()` to drop it into a LCEL chain (sync and async). |
| `LLMClientConfig` | Declarative knobs (model, endpoint overrides, temperature, token caps, rate limiter, retry schedule, custom token counter, request-body kwargs). |
| `TokenUsage` | Tokens billed for one call or accumulated over a run (input / output / total, cache read + write, call count). Read it off `LLMClient.usage`. |
| `InputTokenLimitError` | Raised when a prompt exceeds `max_input_tokens`; nothing is sent. |

## Logging

Logger name: `llm_client.client`

| Level | When emitted |
|-------|--------------|
| DEBUG | Token-count results per call |
| INFO | Model construction (api_key presence and header *names* only — never values); per-call token usage and the running total |
| WARNING | Transient-error retry waits; gateway refusing a `cache_control` breakpoint (once); no usage reported by the gateway (once); token-counter fallback (once); `cert_file` missing or overriding a pre-set `SSL_CERT_FILE` |
| ERROR | Input-token budget exceeded; retry budget exhausted (exception raised after logging) |
