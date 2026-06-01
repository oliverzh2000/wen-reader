# Claude API notes (for wen-reader / WSD bulk work)

Summary of what we worked through. Context: calling Claude from an AWS work
laptop, potentially using it for WSD labeling or bulk translation.

## Auth — Bedrock, not direct Anthropic API

- We have access via **Amazon Bedrock** (AWS IAM auth), not the direct Anthropic
  API. No `ANTHROPIC_API_KEY` involved.
- AWS identity used for verification: `arn:aws:sts::444625565502:assumed-role/Admin/olivezh-Isengard`
  in `us-east-1`.
- Python client: `from anthropic import AnthropicBedrock` — picks up AWS creds
  from the usual env (`AWS_PROFILE`, `AWS_REGION`).
- Newer Claude models on Bedrock require **cross-region inference profile IDs**
  prefixed with `us.` (or `global.`). Bare `anthropic.claude-...` IDs return a
  400 "on-demand throughput isn't supported".

### Confirmed model IDs in us-east-1 (2026-04-21)

| Model      | Bedrock ID                                       |
|------------|--------------------------------------------------|
| Opus 4.7   | `us.anthropic.claude-opus-4-7`                   |
| Opus 4.6   | `us.anthropic.claude-opus-4-6-v1` (note `-v1`)   |
| Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6`                 |
| Haiku 4.5  | `us.anthropic.claude-haiku-4-5-20251001-v1:0`    |

### Feature caveats on Bedrock

- **Managed Agents** (Anthropic-hosted stateful agents) is first-party only —
  not available on Bedrock.
- **Messages Batches API** (`/v1/messages/batches`) is not on Bedrock either.
  Bedrock has its own batch inference via `bedrock:CreateModelInvocationJob`
  (S3 in/out).
- Messages API, tool use, streaming, prompt caching, structured outputs all
  work fine.

## Billing

- Bedrock usage shows up as "Amazon Bedrock" in AWS Cost Explorer, on the
  same bill as the rest of your AWS usage.
- Claude Code (editor) and direct script usage via `AnthropicBedrock` both
  land in the same line item — no tagging distinguishes them. If you need to
  separate them, use different IAM roles / inference profiles / accounts.
- Bedrock pricing typically matches the Anthropic direct API list prices:
  - Opus 4.7 / 4.6: $5.00 / $25.00 per 1M tokens (input / output)
  - Sonnet 4.6:     $3.00 / $15.00
  - Haiku 4.5:      $1.00 / $5.00
- Canonical source: https://aws.amazon.com/bedrock/pricing/

## Prompt caching

How it works:

- You explicitly mark a stable prefix with `cache_control: {"type": "ephemeral"}`
  on the last block of that prefix. Render order is `tools` → `system` →
  `messages`.
- On subsequent requests, the API does an **exact byte prefix match** up to
  each breakpoint. If it matches, tokens are served from cache automatically.
  No flag to read — it just happens.

Economics (vs base input price):

| Token type           | Cost   |
|----------------------|--------|
| Cache read           | ~0.10× |
| Cache write, 5m TTL  | ~1.25× |
| Cache write, 1h TTL  | ~2.00× |
| Uncached input       | 1.00×  |

Break-even: 2 requests for 5m TTL, 3 requests for 1h TTL.

Gotchas:

- Any byte change in the prefix invalidates everything after it. `datetime.now()`,
  UUIDs, or unsorted `json.dumps()` in the system prompt silently kill caching.
- Max 4 `cache_control` breakpoints per request.
- Min cacheable prefix: 1024 tokens (Sonnet 4.6), 2048 (Haiku 3.5), 4096 (Opus
  4.6 / 4.7, Haiku 4.5). Shorter prefixes silently don't cache.
- Verify with `response.usage.cache_read_input_tokens`. If it's 0 across
  identical-prefix requests, something's silently invalidating.

## Batch processing

Two flavors, pick based on auth:

1. **Anthropic-native Messages Batches API** — cleaner, JSONL shape, not
   available on Bedrock. Needs a direct `ANTHROPIC_API_KEY`.
2. **Bedrock batch inference** (`bedrock:CreateModelInvocationJob`) — S3 in,
   S3 out. Uses existing AWS auth.

Both give **50% off** all tokens (input, output, cached, uncached).

### What batches actually do

Batches are **NOT** a caching feature. They're orthogonal. Three things:

1. Async bulk submission (up to 100K requests / 256 MB per batch)
2. 50% discount on every token
3. Dedicated queue that doesn't eat into your sync rate limits

Each request inside a batch is a fully independent Messages call. You still
pass full `system` / `messages` / `tools` per request. Caching works inside
batches the same way it does outside — you still mark `cache_control`
explicitly.

### Lifecycle (fire and forget)

1. **Submit** — returns a `batch_id` immediately. Close your laptop.
2. **Processing** — server-side; most finish <1 hour, hard cap is 24 hours.
3. **Poll whenever** — `retrieve(batch_id)` returns status (`in_progress` /
   `canceling` / `ended`). No webhooks, no push notifications.
4. **Download** — once `ended`, stream results keyed by your `custom_id`.
   Results available for 29 days.

### Cache TTL interaction with batches

Cache TTL is **wall-clock from the write time**, not scoped to the batch.
The first request processed writes the cache; the timer starts there.

- Small/fast batch (~10 min processing): default 5m TTL is usually fine.
- Large/slow batch: **use 1-hour TTL** to avoid repeated re-writes. The
  write cost only applies to the write(s); reads stay ~0.1×.
- For very large workloads (100K+ items), split into sequential chunks sized
  to fit comfortably inside the TTL window.
- Cache is account/org-scoped, so a recent sync request with the same prefix
  can warm the cache ahead of a batch.

### Concrete cost example (1000 Haiku requests, 5K shared prompt, 200 input, 100 output each)

| Strategy                | Approx cost |
|-------------------------|-------------|
| Naive sync, no cache    | ~$5.70      |
| Sync + caching          | ~$1.20      |
| Batch + caching         | ~$0.60      |

Order of magnitude savings when the shared prefix is significant.

## Subagents vs batching (for bulk work)

Spawning Claude Code subagents to process items in parallel is:

- Token-inefficient (each subagent loads the full Claude Code system prompt)
- Doesn't share caching across subagents
- Pays interactive-tier pricing (no 50% batch discount)
- Fine for a handful of genuinely independent exploration tasks, not for
  bulk labeling of 10K items

For real bulk work (WSD, translation, classification at scale), use the
Batch API + prompt caching.

## Quick start snippets

### Single inference call via Bedrock

```python
from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region="us-east-1")
resp = client.messages.create(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens=32,
    messages=[{"role": "user", "content": "Reply with just: ok"}],
)
print(resp.content[0].text)
print(resp.usage)
```

### Shape of a cached-prefix request (applies in sync or batch)

```python
client.messages.create(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens=512,
    system=[{
        "type": "text",
        "text": SHARED_TRANSLATION_RUBRIC,  # same string for every request
        "cache_control": {"type": "ephemeral", "ttl": "1h"},  # 1h for batches
    }],
    messages=[{"role": "user", "content": sentence}],
)
```
