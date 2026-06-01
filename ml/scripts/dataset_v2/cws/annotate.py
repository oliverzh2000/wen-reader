"""LLM-annotate CWS tasks → segmented.jsonl.

Reads cws_tasks.jsonl, sends candidate positions to an LLM, reconstructs
full segmentation from picks.

Supports:
  - Multiple providers: DeepSeek (OpenAI-compat) and Anthropic (via Bedrock)
  - Fork-and-join concurrency with adaptive batch sizing
  - Resume (skips already-completed IDs)
  - Dry-run mode (dumps prompts + estimates cost, no API calls)
  - Retry with backoff on 429/5xx

Usage:
  from cws.annotate import annotate_cws
  annotate_cws(tasks_path, output_path, model="deepseek-v4-flash")
  annotate_cws(tasks_path, output_path, model="claude-sonnet-4")
  annotate_cws(tasks_path, output_path, dry_run=True)  # cost estimate only
"""
import asyncio
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI

from cws.cedict_lookup import reconstruct_segments

_DIR = Path(__file__).parent
_SYSTEM_PROMPT = (_DIR / "system_prompt_v4.md").read_text(encoding="utf-8")
_API_KEY_FILE = _DIR.parent / "DEEPSEEK_API_DO_NOT_COMMIT.txt"
_ANTHROPIC_API_KEY_FILE = _DIR.parent / "ANTHROPIC_API_DO_NOT_COMMIT.txt"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Bedrock config
_BEDROCK_REGION = "us-east-1"
_BEDROCK_PROFILE = "olivezh-aws-profile"

DEFAULT_CONCURRENCY = 8  # parallel API calls per batch
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds
REQUEST_TIMEOUT = 120  # seconds per API call
MAX_OUTPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS_THINKING = 10000  # includes reasoning + visible output

# ---------------------------------------------------------------------------
# Model registry: maps short names → provider config
# ---------------------------------------------------------------------------

_MODELS = {
    # DeepSeek models
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "api_model": "deepseek-v4-pro",
        "pricing": {"input": 0.435, "output": 0.87},
        "thinking": False,
    },
    "deepseek-v4-pro-thinking": {
        "provider": "deepseek",
        "api_model": "deepseek-v4-pro",
        "pricing": {"input": 0.435, "output": 0.87},
        "thinking": True,
    },
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "api_model": "deepseek-v4-flash",
        "pricing": {"input": 0.14, "output": 0.28},
        "thinking": False,
    },
    # Anthropic via Bedrock (adaptive thinking)
    "claude-opus-4.6": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-opus-4-6-v1",
        "pricing": {"input": 15.0, "output": 75.0},
        "thinking": False,
    },
    "claude-opus-4.6-thinking": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-opus-4-6-v1",
        "pricing": {"input": 15.0, "output": 75.0},
        "thinking": "adaptive",
        "effort": "high",
    },
    "claude-opus-4.6-thinking-medium": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-opus-4-6-v1",
        "pricing": {"input": 15.0, "output": 75.0},
        "thinking": "adaptive",
        "effort": "medium",
    },
    "claude-opus-4.6-thinking-low": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-opus-4-6-v1",
        "pricing": {"input": 15.0, "output": 75.0},
        "thinking": "adaptive",
        "effort": "low",
    },
    "claude-sonnet-4.6": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": False,
    },
    "claude-sonnet-4.6-thinking": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": "adaptive",
        "effort": "high",
    },
    "claude-sonnet-4.6-thinking-medium": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": "adaptive",
        "effort": "medium",
    },
    "claude-sonnet-4.6-thinking-low": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": "adaptive",
        "effort": "low",
    },
    "claude-haiku-4.5": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "pricing": {"input": 0.80, "output": 4.0},
        "thinking": False,
    },
    # Anthropic direct API (for Batch API support)
    "sonnet-4.6": {
        "provider": "anthropic",
        "api_model": "claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": False,
    },
    "sonnet-4.6-thinking": {
        "provider": "anthropic",
        "api_model": "claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": "adaptive",
        "effort": "high",
    },
    "opus-4.6": {
        "provider": "anthropic",
        "api_model": "claude-opus-4-6",
        "pricing": {"input": 15.0, "output": 75.0},
        "thinking": False,
    },
    "haiku-4.5": {
        "provider": "anthropic",
        "api_model": "claude-haiku-4-5",
        "pricing": {"input": 0.80, "output": 4.0},
        "thinking": False,
    },
}


def _get_model_config(model: str) -> dict:
    """Look up model config by short name."""
    if model not in _MODELS:
        available = ", ".join(sorted(_MODELS.keys()))
        raise ValueError(f"Unknown model '{model}'. Available: {available}")
    return _MODELS[model]


def _load_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        return _API_KEY_FILE.read_text().strip()
    raise RuntimeError(f"No API key. Set DEEPSEEK_API_KEY or create {_API_KEY_FILE}")


def _get_bedrock_client(read_timeout: int = 300):
    """Create a boto3 bedrock-runtime client (sync — used in thread)."""
    import boto3
    from botocore.config import Config
    config = Config(read_timeout=read_timeout, connect_timeout=10, retries={"max_attempts": 0})
    session = boto3.Session(profile_name=_BEDROCK_PROFILE, region_name=_BEDROCK_REGION)
    return session.client("bedrock-runtime", config=config)


def _format_user_prompt(task: dict) -> str:
    """Build the per-sentence user prompt from a task."""
    lines = [f"Text: {task['text']}"]
    candidates = task["candidates"]
    if candidates:
        lines.append("Candidates:")
        for pos in sorted(candidates.keys(), key=int):
            words = candidates[pos]
            lines.append(f"  pos {pos}: {', '.join(words)}")
    defs = task.get("defs", {})
    if defs:
        lines.append("Definitions:")
        for word in sorted(defs.keys()):
            lines.append(f"  {word} {defs[word]}")
    lines.append("")
    lines.append("Respond with JSON: {\"picks\": {...}}")
    return "\n".join(lines)


def _format_batch_prompt(tasks: list[dict]) -> str:
    """Build a multi-sentence user prompt for batched annotation."""
    if len(tasks) == 1:
        return _format_user_prompt(tasks[0])

    lines = []
    for i, task in enumerate(tasks, 1):
        lines.append(f"Sentence {i}:")
        lines.append(f"Text: {task['text']}")
        candidates = task["candidates"]
        if candidates:
            lines.append("Candidates:")
            for pos in sorted(candidates.keys(), key=int):
                words = candidates[pos]
                lines.append(f"  pos {pos}: {', '.join(words)}")
        defs = task.get("defs", {})
        if defs:
            lines.append("Definitions:")
            for word in sorted(defs.keys()):
                lines.append(f"  {word} {defs[word]}")
        lines.append("")

    lines.append(f"Respond with JSON: {{\"results\": [{{\"picks\": {{...}}}}, ...]}}")
    return "\n".join(lines)


def _validate_segments(text: str, segments: list[str]) -> bool:
    return "".join(segments) == text


def _extract_json(raw: str) -> dict:
    """Extract JSON object from LLM response, handling markdown fences.

    Used for Bedrock responses where structured output isn't available.
    Raises json.JSONDecodeError if no valid JSON can be extracted.
    """
    import re
    text = raw.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code fence
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    # Find first { ... } block by brace matching
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    raise json.JSONDecodeError("No JSON found in response", text, 0)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(tasks_path: Path, output_path: Path, model: str, n_samples: int = 5):
    """Dump sample prompts and estimate total cost."""
    tasks = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]

    # Estimate tokens (rough: 1 Chinese char ≈ 1 token for DeepSeek)
    system_tokens = len(_SYSTEM_PROMPT)  # ~chars ≈ tokens
    total_input_tokens = 0
    user_prompts = []

    for task in tasks:
        prompt = _format_user_prompt(task)
        user_prompts.append(prompt)
        total_input_tokens += system_tokens + len(prompt)

    # Estimate output: ~20 tokens per task (short JSON picks)
    est_output_tokens = len(tasks) * 20

    # With caching: system prompt cached after first call
    # Cache hit = 0.1x price, cache miss = 1x
    cached_input = system_tokens * (len(tasks) - 1)  # all but first
    uncached_input = total_input_tokens - cached_input
    effective_input = uncached_input + cached_input * 0.1

    pricing = _get_model_config(model)["pricing"]
    input_cost = effective_input / 1_000_000 * pricing["input"]
    output_cost = est_output_tokens / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost

    print(f"  Model: {model}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  System prompt: ~{system_tokens} tokens (cached after 1st call)")
    print(f"  Total input tokens: ~{total_input_tokens:,}")
    print(f"    With caching: ~{int(effective_input):,} effective tokens")
    print(f"  Est output tokens: ~{est_output_tokens:,}")
    print(f"  Est cost: ${total_cost:.2f} (input ${input_cost:.2f} + output ${output_cost:.2f})")
    print()

    # Dump samples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output_path.with_suffix(".dry_run.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(f"=== SYSTEM PROMPT ===\n\n{_SYSTEM_PROMPT}\n\n")
        f.write(f"{'='*60}\n\n")
        for i, task in enumerate(tasks[:n_samples]):
            prompt = _format_user_prompt(task)
            f.write(f"=== TASK {task['id']} ({task['source']}) ===\n\n")
            f.write(f"{prompt}\n\n")
            f.write(f"{'='*60}\n\n")
        # Also dump last few
        if len(tasks) > n_samples:
            f.write(f"\n... ({len(tasks) - n_samples*2} tasks omitted) ...\n\n")
            for task in tasks[-n_samples:]:
                prompt = _format_user_prompt(task)
                f.write(f"=== TASK {task['id']} ({task['source']}) ===\n\n")
                f.write(f"{prompt}\n\n")
                f.write(f"{'='*60}\n\n")

    print(f"  Sample prompts → {sample_path.name}")


# ---------------------------------------------------------------------------
# Async annotation
# ---------------------------------------------------------------------------


async def _call_deepseek(
    client: AsyncOpenAI, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    thinking: bool = False, max_tokens: int | None = None,
) -> tuple[str | None, dict | None]:
    """Call DeepSeek API. Returns (raw_content, usage_info) or (None, None)."""
    if max_tokens is None:
        max_tokens = MAX_OUTPUT_TOKENS_THINKING if thinking else MAX_OUTPUT_TOKENS
    extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    usage = resp.usage
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cache_hit": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        "cache_miss": getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
    }
    raw = resp.choices[0].message.content
    if not raw and verbose:
        choice = resp.choices[0]
        print(f"    [{task_id}] empty response — "
              f"finish_reason={choice.finish_reason}, "
              f"refusal={getattr(choice.message, 'refusal', None)}")
    finish_reason = resp.choices[0].finish_reason
    return raw, usage_info, finish_reason


async def _call_bedrock(
    bedrock_client, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    thinking: str | bool = False,
    effort: str = "high",
    max_tokens: int | None = None,
) -> tuple[str | None, dict | None]:
    """Call Anthropic via Bedrock. Returns (raw_content, usage_info) or (None, None)."""
    if max_tokens is None:
        max_tokens = MAX_OUTPUT_TOKENS_THINKING if thinking else MAX_OUTPUT_TOKENS

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    }

    if thinking == "adaptive":
        request["temperature"] = 1.0
        request["thinking"] = {"type": "adaptive"}
        request["output_config"] = {"effort": effort}
    else:
        request["temperature"] = 0.0

    body = json.dumps(request)

    # boto3 is sync — run in thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: bedrock_client.invoke_model(
            modelId=model_id, body=body, contentType="application/json"
        ),
    )

    result = json.loads(resp["body"].read())
    usage = result.get("usage", {})
    if verbose:
        print(f"    [{task_id}] usage: {usage}")
    usage_info = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "cache_hit": usage.get("cache_read_input_tokens", 0),
        "cache_miss": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
    }

    content_blocks = result.get("content", [])
    raw = None
    for block in content_blocks:
        if block.get("type") == "text":
            raw = block["text"]

    if not raw and verbose:
        print(f"    [{task_id}] empty Bedrock response — stop_reason={result.get('stop_reason')}")

    finish_reason = result.get("stop_reason")
    return raw, usage_info, finish_reason


async def _annotate_one(
    client,
    task: dict,
    model_config: dict,
    verbose: bool = False,
) -> tuple[dict | None, dict | None]:
    """Annotate a single task with retry. Returns (result, usage_info)."""
    text = task["text"]
    candidates = task["candidates"]
    user_prompt = _format_user_prompt(task)
    task_id = task["id"]
    provider = model_config["provider"]
    model_id = model_config["api_model"]

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"    [{task_id}] attempt {attempt+1}/{MAX_RETRIES}...")

            if provider == "deepseek":
                raw, usage_info, finish_reason = await _call_deepseek(
                    client, model_id, user_prompt, verbose, task_id,
                    thinking=model_config.get("thinking", False),
                )
            else:
                raw, usage_info, finish_reason = await _call_bedrock(
                    client, model_id, user_prompt, verbose, task_id,
                    thinking=model_config.get("thinking", False),
                    effort=model_config.get("effort", "high"),
                )

            if not raw:
                # Length truncation — reasoning used entire budget, don't retry
                if finish_reason in ("length", "max_tokens"):
                    if verbose:
                        print(f"    [{task_id}] FAIL truncated (finish_reason={finish_reason})")
                    return None, usage_info
                # Other empty response — transient glitch, retry
                await asyncio.sleep(0.5)
                continue

            parsed = json.loads(raw) if model_config["provider"] == "deepseek" else _extract_json(raw)
            picks = parsed.get("picks", {})

            # Validate picks against candidates — deterministic failure, don't retry
            for pos, word in picks.items():
                candidate_words = candidates.get(pos, [])
                if word not in candidate_words:
                    if verbose:
                        print(f"    [{task_id}] FAIL invalid pick: pos {pos} '{word}' not in {candidate_words}")
                    return None, usage_info

            segments = reconstruct_segments(text, picks, task.get("prefilled"))
            if not _validate_segments(text, segments):
                if verbose:
                    joined = "".join(segments)
                    print(f"    [{task_id}] FAIL segment mismatch: '{joined}' != '{text}'")
                return None, usage_info

            if verbose:
                print(f"    [{task_id}] ✓ {len(picks)} picks")
            return {
                "id": task["id"],
                "source": task["source"],
                "text": text,
                "segments": segments,
            }, usage_info

        except Exception as e:
            if verbose:
                print(f"    [{task_id}] error (attempt {attempt+1}): {e}")
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "ThrottlingException" in err_str:
                await asyncio.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])
                continue
            # Non-transient exception — don't retry
            return None, None

    return None, None


async def _annotate_batch(
    client,
    tasks: list[dict],
    model_config: dict,
    verbose: bool = False,
) -> list[tuple[dict | None, dict | None]]:
    """Annotate multiple tasks in a single API call. Returns list of (result, usage_info).

    Falls back to individual calls if batch parsing fails.
    """
    if len(tasks) == 1:
        result = await _annotate_one(client, tasks[0], model_config, verbose)
        return [result]

    user_prompt = _format_batch_prompt(tasks)
    batch_id = f"batch[{tasks[0]['id']}..{tasks[-1]['id']}]"
    provider = model_config["provider"]
    model_id = model_config["api_model"]

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"    [{batch_id}] attempt {attempt+1}/{MAX_RETRIES} ({len(tasks)} tasks)...")

            batch_max_tokens = MAX_OUTPUT_TOKENS * len(tasks)
            batch_max_tokens_thinking = MAX_OUTPUT_TOKENS_THINKING * len(tasks)

            if provider == "deepseek":
                raw, usage_info, finish_reason = await _call_deepseek(
                    client, model_id, user_prompt, verbose, batch_id,
                    thinking=model_config.get("thinking", False),
                    max_tokens=batch_max_tokens_thinking if model_config.get("thinking") else batch_max_tokens,
                )
            else:
                raw, usage_info, finish_reason = await _call_bedrock(
                    client, model_id, user_prompt, verbose, batch_id,
                    thinking=model_config.get("thinking", False),
                    effort=model_config.get("effort", "high"),
                    max_tokens=batch_max_tokens_thinking if model_config.get("thinking") else batch_max_tokens,
                )

            if not raw:
                if finish_reason in ("length", "max_tokens"):
                    if verbose:
                        print(f"    [{batch_id}] FAIL truncated")
                    return [(None, usage_info)] + [(None, None)] * (len(tasks) - 1)
                await asyncio.sleep(0.5)
                continue

            parsed = json.loads(raw) if provider == "deepseek" else _extract_json(raw)
            results_list = parsed.get("results", [])

            if len(results_list) != len(tasks):
                if verbose:
                    print(f"    [{batch_id}] FAIL got {len(results_list)} results for {len(tasks)} tasks")
                # Fall back to individual calls
                individual_results = []
                for t in tasks:
                    individual_results.append(await _annotate_one(client, t, model_config, verbose))
                return individual_results

            # Validate each result
            outputs = []
            # Split usage evenly across tasks for reporting
            per_task_usage = {k: v // len(tasks) if isinstance(v, int) else v
                             for k, v in (usage_info or {}).items()} if usage_info else None

            for i, (task, result_picks) in enumerate(zip(tasks, results_list)):
                picks = result_picks.get("picks", {})
                text = task["text"]
                candidates = task["candidates"]
                task_id = task["id"]

                # Validate picks
                valid = True
                for pos, word in picks.items():
                    candidate_words = candidates.get(pos, [])
                    if word not in candidate_words:
                        if verbose:
                            print(f"    [{task_id}] FAIL invalid pick in batch: pos {pos} '{word}'")
                        valid = False
                        break

                if not valid:
                    outputs.append((None, per_task_usage))
                    continue

                segments = reconstruct_segments(text, picks, task.get("prefilled"))
                if not _validate_segments(text, segments):
                    if verbose:
                        print(f"    [{task_id}] FAIL segment mismatch in batch")
                    outputs.append((None, per_task_usage))
                    continue

                if verbose:
                    print(f"    [{task_id}] ✓ {len(picks)} picks (batch)")
                outputs.append(({
                    "id": task_id,
                    "source": task["source"],
                    "text": text,
                    "segments": segments,
                }, per_task_usage))

            return outputs

        except Exception as e:
            if verbose:
                print(f"    [{batch_id}] error (attempt {attempt+1}): {e}")
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "ThrottlingException" in err_str:
                await asyncio.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])
                continue
            # Fall back to individual
            individual_results = []
            for t in tasks:
                individual_results.append(await _annotate_one(client, t, model_config, verbose))
            return individual_results

    return [(None, None)] * len(tasks)


async def _run_async(
    tasks: list[dict], output_path: Path, model: str, concurrency: int,
    verbose: bool = False, tasks_per_call: int = 1,
) -> dict:
    """Run all tasks using fork-and-join: spawn N calls, wait for all, repeat.

    Batch size adapts: doubles after a fully-successful batch, halves if
    any task in the batch hit a rate-limit (429/503) error.
    """
    model_config = _get_model_config(model)
    provider = model_config["provider"]
    pricing = model_config["pricing"]

    # Create the appropriate client
    if provider == "deepseek":
        client = AsyncOpenAI(
            api_key=_load_deepseek_api_key(), base_url=_DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT
        )
    else:
        # Scale timeout: base 60s per task, more for thinking
        per_task_timeout = 120 if model_config.get("thinking") else 60
        read_timeout = per_task_timeout * tasks_per_call
        client = _get_bedrock_client(read_timeout=read_timeout)
    stats = {"ok": 0, "failed": 0, "rate_limited": 0}
    tokens = {"input": 0, "input_cached": 0, "output": 0}
    start = time.time()
    batch_size = concurrency
    max_batch = concurrency
    min_batch = concurrency

    with open(output_path, "a", encoding="utf-8") as f:
        offset = 0
        while offset < len(tasks):
            # Each "slot" processes tasks_per_call tasks in one API call
            # We launch concurrency such slots in parallel
            call_batches = []
            for _ in range(batch_size):
                if offset >= len(tasks):
                    break
                end = min(offset + tasks_per_call, len(tasks))
                call_batches.append(tasks[offset:end])
                offset += end - offset

            results_nested = await asyncio.gather(
                *[_annotate_batch(client, tb, model_config, verbose=verbose) for tb in call_batches]
            )

            batch_rate_limited = False
            for batch_results in results_nested:
                for (result, usage_info) in batch_results:
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stats["ok"] += 1
                    else:
                        stats["failed"] += 1

                    if usage_info:
                        tokens["input"] += usage_info["cache_miss"]
                        tokens["input_cached"] += usage_info["cache_hit"]
                        tokens["output"] += usage_info["completion_tokens"]
                    elif result is None:
                        # No usage_info + failed = likely rate-limited
                        batch_rate_limited = True

            f.flush()

            # Adapt batch size
            if batch_rate_limited:
                batch_size = max(min_batch, batch_size // 2)
                if verbose:
                    print(f"  ↓ batch size → {batch_size}")
            else:
                old = batch_size
                batch_size = min(max_batch, batch_size * 2)
                if batch_size != old and verbose:
                    print(f"  ↑ batch size → {batch_size}")

            # Progress
            total = stats["ok"] + stats["failed"]
            if total % 50 < batch_size or verbose:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - total) / rate if rate > 0 else 0

                input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"]
                output_cost = tokens["output"] / 1_000_000 * pricing["output"]
                spent = input_cost + output_cost
                est_total = spent * len(tasks) / total if total > 0 else 0

                eta_min = remaining / 60
                print(
                    f"  [{total}/{len(tasks)}] "
                    f"ok={stats['ok']} fail={stats['failed']} "
                    f"| ${spent:.3f} spent (est ${est_total:.2f} total) "
                    f"| {rate:.1f} tasks/s, ETA {eta_min:.0f}m "
                    f"| in={tokens['input']:,} cached={tokens['input_cached']:,} "
                    f"out={tokens['output']:,}"
                )

    elapsed = time.time() - start
    rate = len(tasks) / elapsed if elapsed > 0 else 0
    input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"]
    output_cost = tokens["output"] / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost
    print(
        f"  Done: {stats['ok']} segmented, {stats['failed']} failed "
        f"({elapsed:.0f}s, {rate:.1f} tasks/s)\n"
        f"  Cost: ${total_cost:.3f} "
        f"(input={tokens['input']:,} cached={tokens['input_cached']:,} "
        f"output={tokens['output']:,})"
    )
    return {
        "ok": stats["ok"],
        "failed": stats["failed"],
        "elapsed": elapsed,
        "rate": rate,
        "cost": total_cost,
        "tokens": dict(tokens),
    }


# ---------------------------------------------------------------------------
# Anthropic Batch API
# ---------------------------------------------------------------------------


def _build_batch_request(task: dict, model_config: dict, tasks_per_call: int, task_group: list[dict] | None = None) -> dict:
    """Build a single batch request item for the Anthropic Batch API."""
    if task_group and len(task_group) > 1:
        user_prompt = _format_batch_prompt(task_group)
        # custom_id must match ^[a-zA-Z0-9_-]{1,64}$
        id_start = task_group[0]["id"].replace(":", "_").replace(".", "-")
        id_end = task_group[-1]["id"].replace(":", "_").replace(".", "-")
        custom_id = f"batch_{id_start}--{id_end}"[:64]
        max_tokens = MAX_OUTPUT_TOKENS * len(task_group)
    else:
        user_prompt = _format_user_prompt(task)
        custom_id = task["id"].replace(":", "_").replace(".", "-")[:64]
        max_tokens = MAX_OUTPUT_TOKENS

    thinking = model_config.get("thinking", False)
    if thinking:
        max_tokens = MAX_OUTPUT_TOKENS_THINKING * (len(task_group) if task_group else 1)

    params = {
        "model": model_config["api_model"],
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }

    if thinking == "adaptive":
        params["temperature"] = 1.0
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": model_config.get("effort", "high")}
    else:
        params["temperature"] = 0.0

    return {"custom_id": custom_id, "params": params}


def run_batch_api(
    tasks: list[dict],
    output_path: Path,
    model: str,
    tasks_per_call: int = 1,
    verbose: bool = False,
    poll_interval: int = 30,
) -> dict:
    """Submit all tasks via Anthropic Batch API, poll for completion, write results.

    Returns stats dict like _run_async.
    """
    import anthropic

    model_config = _get_model_config(model)
    if model_config["provider"] != "anthropic":
        raise ValueError(f"Batch API only supports 'anthropic' provider models. Got: {model} ({model_config['provider']})")

    pricing = model_config["pricing"]
    batch_discount = 0.5  # 50% off for batch API

    # Load API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and _ANTHROPIC_API_KEY_FILE.exists():
        api_key = _ANTHROPIC_API_KEY_FILE.read_text().strip()
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Group tasks by tasks_per_call
    task_groups = []
    for i in range(0, len(tasks), tasks_per_call):
        task_groups.append(tasks[i:i + tasks_per_call])

    print(f"  Building {len(task_groups)} batch requests ({len(tasks)} tasks, {tasks_per_call} per call)...")

    # Build batch requests
    requests = []
    for group in task_groups:
        if len(group) == 1:
            req = _build_batch_request(group[0], model_config, tasks_per_call)
        else:
            req = _build_batch_request(group[0], model_config, tasks_per_call, task_group=group)
        requests.append(req)

    print(f"  Submitting batch to Anthropic API...")
    start = time.time()
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")

    # Poll until done
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        elapsed = time.time() - start
        counts = batch.request_counts
        print(f"  [{elapsed:.0f}s] {batch.processing_status} — "
              f"succeeded={counts.succeeded} errored={counts.errored} "
              f"processing={counts.processing} canceled={counts.canceled}")

    elapsed = time.time() - start
    print(f"  Batch completed in {elapsed:.0f}s")

    # Process results
    stats = {"ok": 0, "failed": 0}
    tokens = {"input": 0, "input_cached": 0, "output": 0}

    # Map custom_id → task group for validation (must match sanitized IDs)
    group_map = {}
    for group in task_groups:
        if len(group) == 1:
            key = group[0]["id"].replace(":", "_").replace(".", "-")[:64]
            group_map[key] = group
        else:
            id_start = group[0]["id"].replace(":", "_").replace(".", "-")
            id_end = group[-1]["id"].replace(":", "_").replace(".", "-")
            key = f"batch_{id_start}--{id_end}"[:64]
            group_map[key] = group

    with open(output_path, "a", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            group = group_map.get(custom_id)

            if result.result.type == "errored":
                if verbose:
                    print(f"    [{custom_id}] ERROR: {result.result.error}")
                stats["failed"] += len(group) if group else 1
                continue

            msg = result.result.message
            usage = msg.usage

            tokens["input"] += getattr(usage, "input_tokens", 0)
            tokens["input_cached"] += getattr(usage, "cache_read_input_tokens", 0) or 0
            tokens["output"] += getattr(usage, "output_tokens", 0)

            # Extract text content
            raw = None
            for block in msg.content:
                if block.type == "text":
                    raw = block.text
                    break

            if not raw:
                if verbose:
                    print(f"    [{custom_id}] empty response")
                stats["failed"] += len(group) if group else 1
                continue

            try:
                parsed = _extract_json(raw)
            except Exception as e:
                if verbose:
                    print(f"    [{custom_id}] JSON parse error: {e}")
                stats["failed"] += len(group) if group else 1
                continue

            # Handle single vs multi-task results
            if group and len(group) > 1:
                results_list = parsed.get("results", [])
                if len(results_list) != len(group):
                    if verbose:
                        print(f"    [{custom_id}] got {len(results_list)} results for {len(group)} tasks")
                    stats["failed"] += len(group)
                    continue

                for task, result_picks in zip(group, results_list):
                    picks = result_picks.get("picks", {})
                    if _process_picks(task, picks, f, verbose):
                        stats["ok"] += 1
                    else:
                        stats["failed"] += 1
            else:
                task = group[0] if group else None
                picks = parsed.get("picks", {})
                if task and _process_picks(task, picks, f, verbose):
                    stats["ok"] += 1
                else:
                    stats["failed"] += 1

    # Cost (with 50% batch discount)
    input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"] * batch_discount
    output_cost = tokens["output"] / 1_000_000 * pricing["output"] * batch_discount
    total_cost = input_cost + output_cost

    print(f"  Done: {stats['ok']} segmented, {stats['failed']} failed ({elapsed:.0f}s)")
    print(f"  Cost: ${total_cost:.3f} (with 50% batch discount)")
    print(f"    input={tokens['input']:,} cached={tokens['input_cached']:,} output={tokens['output']:,}")

    return {
        "ok": stats["ok"],
        "failed": stats["failed"],
        "elapsed": elapsed,
        "cost": total_cost,
        "tokens": dict(tokens),
    }


def _process_picks(task: dict, picks: dict, f, verbose: bool) -> bool:
    """Validate picks and write result. Returns True on success."""
    text = task["text"]
    candidates = task["candidates"]
    task_id = task["id"]

    for pos, word in picks.items():
        candidate_words = candidates.get(pos, [])
        if word not in candidate_words:
            if verbose:
                print(f"    [{task_id}] FAIL invalid pick: pos {pos} '{word}' not in {candidate_words}")
            return False

    segments = reconstruct_segments(text, picks, task.get("prefilled"))
    if not _validate_segments(text, segments):
        if verbose:
            print(f"    [{task_id}] FAIL segment mismatch")
        return False

    result = {
        "id": task_id,
        "source": task["source"],
        "text": text,
        "segments": segments,
    }
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    if verbose:
        print(f"    [{task_id}] ✓ {len(picks)} picks")
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_cws(
    tasks_path: Path,
    output_path: Path,
    model: str = "deepseek-v4-pro",
    dry_run_mode: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    verbose: bool = False,
    tasks_per_call: int = 1,
    batch: bool = False,
) -> None:
    """Run LLM annotation on CWS tasks."""
    if dry_run_mode:
        dry_run(tasks_path, output_path, model)
        return

    # Load already-completed IDs for resume
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])

    tasks = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    pending = [t for t in tasks if t["id"] not in done_ids]

    print(f"  {len(tasks)} total, {len(done_ids)} done, {len(pending)} pending")
    if tasks_per_call > 1:
        print(f"  Batching {tasks_per_call} tasks per API call")

    if not pending:
        print("  Nothing to do.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch:
        print(f"  Using Anthropic Batch API (50% discount)")
        run_batch_api(pending, output_path, model, tasks_per_call=tasks_per_call, verbose=verbose)
    else:
        asyncio.run(_run_async(pending, output_path, model, concurrency, verbose=verbose,
                               tasks_per_call=tasks_per_call))


if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent.parent.parent  # ml/
    data = root / "data" / "dataset_v2"

    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    mode = args[0] if args else "dry-run"
    concurrency = int(args[1]) if len(args) > 1 else DEFAULT_CONCURRENCY

    if mode == "dry-run":
        annotate_cws(data / "ebooks_cws_tasks.jsonl", data / "ebooks_cws_results.jsonl", dry_run_mode=True)
    else:
        annotate_cws(data / "ebooks_cws_tasks.jsonl", data / "ebooks_cws_results.jsonl", concurrency=concurrency, verbose=verbose)
