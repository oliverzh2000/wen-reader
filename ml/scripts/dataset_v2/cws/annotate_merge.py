"""LLM-annotate ICWB2 merge tasks → segmented results.

Reads icwb2_cws_tasks.jsonl (merge-format), sends merge decisions to LLM,
reconstructs final segmentation by applying accepted merges to gold segments.

Supports:
  - OpenAI (gpt-4o-mini, gpt-4o)
  - DeepSeek
  - Anthropic via Bedrock
  - Resume (skips already-completed IDs)
  - Dry-run mode

Usage:
  from cws.annotate_merge import annotate_merges
  annotate_merges(tasks_path, output_path, model="gpt-4o-mini")
"""
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

_DIR = Path(__file__).parent
_SYSTEM_PROMPT = (_DIR / "system_prompt_merge.md").read_text(encoding="utf-8")
_API_KEY_FILE = _DIR.parent / "DEEPSEEK_API_DO_NOT_COMMIT.txt"
_OPENAI_API_KEY_FILE = _DIR.parent / "OPENAI_API_DO_NOT_COMMIT.txt"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Bedrock config
_BEDROCK_REGION = "us-east-1"
_BEDROCK_PROFILE = "olivezh-aws-profile"

DEFAULT_CONCURRENCY = 16
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]
REQUEST_TIMEOUT = 60
MAX_OUTPUT_TOKENS = 256  # merge responses are tiny
MAX_OUTPUT_TOKENS_THINKING = 4096  # thinking models need budget for reasoning

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "pricing": {"input": 0.075, "output": 0.30, "cached_input": 0.075},
    },
    "gpt-5-mini": {
        "provider": "openai",
        "api_model": "gpt-5-mini",
        "pricing": {"input": 0.125, "output": 1.00, "cached_input": 0.0125},
    },
    "gpt-4o": {
        "provider": "openai",
        "api_model": "gpt-4o",
        "pricing": {"input": 1.25, "output": 5.00, "cached_input": 1.25},
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "api_model": "gpt-4.1-mini",
        "pricing": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    },
    "gpt-4.1-nano": {
        "provider": "openai",
        "api_model": "gpt-4.1-nano",
        "pricing": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    },
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "api_model": "deepseek-v4-flash",
        "pricing": {"input": 0.14, "output": 0.28, "cached_input": 0.014},
    },
    "claude-haiku-4.5": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "pricing": {"input": 0.80, "output": 4.0, "cached_input": 0.08},
    },
    "claude-sonnet-4.6": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0, "cached_input": 0.30},
    },
    "claude-sonnet-4.6-thinking": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0, "cached_input": 0.30},
        "thinking": "adaptive",
        "effort": "medium",
    },
}


def _get_model_config(model: str) -> dict:
    if model not in _MODELS:
        available = ", ".join(sorted(_MODELS.keys()))
        raise ValueError(f"Unknown model '{model}'. Available: {available}")
    return _MODELS[model]


def _load_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    if _OPENAI_API_KEY_FILE.exists():
        return _OPENAI_API_KEY_FILE.read_text().strip()
    raise RuntimeError(f"No OpenAI API key. Set OPENAI_API_KEY or create {_OPENAI_API_KEY_FILE}")


def _load_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        return _API_KEY_FILE.read_text().strip()
    raise RuntimeError(f"No DeepSeek API key. Set DEEPSEEK_API_KEY or create {_API_KEY_FILE}")


def _get_bedrock_client(read_timeout: int = 120):
    import boto3
    from botocore.config import Config
    config = Config(read_timeout=read_timeout, connect_timeout=10, retries={"max_attempts": 0})
    session = boto3.Session(profile_name=_BEDROCK_PROFILE, region_name=_BEDROCK_REGION)
    return session.client("bedrock-runtime", config=config)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_user_prompt(task: dict) -> str:
    """Build user prompt for a single merge task."""
    lines = [f"S: {task['segments']}"]
    for group in task["groups"]:
        lines.append(f"G: {' / '.join(group)}")
    return "\n".join(lines)


def _format_batch_prompt(tasks: list[dict]) -> str:
    """Build batched user prompt for multiple merge tasks."""
    if len(tasks) == 1:
        return _format_user_prompt(tasks[0])

    lines = []
    for i, task in enumerate(tasks, 1):
        lines.append(f"[{i}]")
        lines.append(f"S: {task['segments']}")
        for group in task["groups"]:
            lines.append(f"G: {' / '.join(group)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------


def _apply_merges(
    segments_str: str, groups: list[list[str]], picks: list[str | None]
) -> list[str] | None:
    """Apply accepted merges to gold segments, return final segmentation.

    Returns None if validation fails.
    """
    segments = segments_str.split()

    # Build a set of merges to apply: find where each picked word occurs
    # as a concatenation of consecutive segments
    consumed = set()  # segment indices consumed by merges
    final_merges: list[tuple[int, int, str]] = []  # (start_idx, end_idx_exclusive, word)

    for pick in picks:
        if pick is None:
            continue

        # Find which consecutive segments form this word
        found = False
        for i in range(len(segments)):
            if i in consumed:
                continue
            concat = ""
            for j in range(i, len(segments)):
                if j in consumed:
                    break
                concat += segments[j]
                if concat == pick:
                    final_merges.append((i, j + 1, pick))
                    for k in range(i, j + 1):
                        consumed.add(k)
                    found = True
                    break
                if len(concat) > len(pick):
                    break
            if found:
                break

        if not found:
            return None  # pick doesn't match any segment sequence

    # Reconstruct: apply merges, keep other segments as-is
    result = []
    i = 0
    while i < len(segments):
        merged = False
        for start, end, word in final_merges:
            if i == start:
                result.append(word)
                i = end
                merged = True
                break
        if not merged:
            result.append(segments[i])
            i += 1

    return result


def _validate_picks(picks: list, groups: list[list[str]]) -> bool:
    """Validate that picks are consistent with groups (1-based indices or null)."""
    if len(picks) != len(groups):
        return False
    for pick, group in zip(picks, groups):
        if pick is not None:
            if not isinstance(pick, int) or pick < 1 or pick > len(group):
                return False
    return True


def _resolve_picks(picks: list, groups: list[list[str]]) -> list[str | None]:
    """Convert 1-based index picks to actual word strings."""
    resolved = []
    for pick, group in zip(picks, groups):
        if pick is None:
            resolved.append(None)
        else:
            resolved.append(group[pick - 1])
    return resolved


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


async def _call_openai(
    client: AsyncOpenAI, model_id: str, user_prompt: str, verbose: bool, task_id: str,
) -> tuple[str | None, dict | None, str | None]:
    """Call OpenAI API. Returns (raw_content, usage_info, finish_reason)."""
    # Newer models (gpt-5-*) require max_completion_tokens instead of max_tokens
    # and only support temperature=1
    use_new_api = "gpt-5" in model_id
    use_completion_tokens = use_new_api or "gpt-4.1" in model_id
    token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"

    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        token_param: MAX_OUTPUT_TOKENS_THINKING if use_new_api else MAX_OUTPUT_TOKENS,
    }
    if not use_new_api:
        kwargs["temperature"] = 0.0
    # Structured output: guarantees valid schema
    kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "merge_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "picks": {
                                    "type": "array",
                                    "items": {"type": ["integer", "null"]}
                                }
                            },
                            "required": ["picks"],
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            }
        }
    }

    resp = await client.chat.completions.create(**kwargs)
    usage = resp.usage
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": getattr(usage, "prompt_tokens_details", None),
    }
    # Extract cached tokens if available
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        usage_info["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
    else:
        usage_info["cached_tokens"] = 0

    raw = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    return raw, usage_info, finish_reason


async def _call_deepseek(
    client: AsyncOpenAI, model_id: str, user_prompt: str, verbose: bool, task_id: str,
) -> tuple[str | None, dict | None, str | None]:
    """Call DeepSeek API."""
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    usage = resp.usage
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
    }
    raw = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    return raw, usage_info, finish_reason


async def _call_bedrock(
    bedrock_client, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    thinking: str | bool = False, effort: str = "medium",
) -> tuple[str | None, dict | None, str | None]:
    """Call Anthropic via Bedrock."""
    max_tokens = 2048 if thinking else MAX_OUTPUT_TOKENS
    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    }
    if thinking == "adaptive":
        request["temperature"] = 1.0
        request["thinking"] = {"type": "adaptive"}
        request["output_config"] = {"effort": effort}
    else:
        request["temperature"] = 0.0

    body = json.dumps(request)
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: bedrock_client.invoke_model(modelId=model_id, body=body, contentType="application/json"),
    )
    result = json.loads(resp["body"].read())
    usage = result.get("usage", {})
    usage_info = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "cached_tokens": usage.get("cache_read_input_tokens", 0),
    }
    content_blocks = result.get("content", [])
    raw = None
    for block in content_blocks:
        if block.get("type") == "text":
            raw = block["text"]
    finish_reason = result.get("stop_reason")
    return raw, usage_info, finish_reason


def _extract_json(raw: str) -> dict:
    """Extract JSON from response, handling markdown fences."""
    import re
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
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
    raise json.JSONDecodeError("No JSON found", text, 0)


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------


async def _annotate_batch(
    client, tasks: list[dict], model_config: dict, verbose: bool = False,
) -> list[tuple[dict | None, dict | None]]:
    """Annotate a batch of merge tasks. Returns list of (result, usage_info)."""
    provider = model_config["provider"]
    model_id = model_config["api_model"]
    user_prompt = _format_batch_prompt(tasks)
    batch_id = tasks[0]["id"] if len(tasks) == 1 else f"batch[{tasks[0]['id']}..{tasks[-1]['id']}]"

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"    [{batch_id}] attempt {attempt+1} ({len(tasks)} tasks)")

            if provider == "openai":
                raw, usage_info, finish_reason = await _call_openai(
                    client, model_id, user_prompt, verbose, batch_id)
            elif provider == "deepseek":
                raw, usage_info, finish_reason = await _call_deepseek(
                    client, model_id, user_prompt, verbose, batch_id)
            else:
                raw, usage_info, finish_reason = await _call_bedrock(
                    client, model_id, user_prompt, verbose, batch_id,
                    thinking=model_config.get("thinking", False),
                    effort=model_config.get("effort", "medium"))

            if not raw:
                if finish_reason in ("length", "max_tokens"):
                    if verbose:
                        print(f"    [{batch_id}] FAIL truncated")
                    return [(None, usage_info)] + [(None, None)] * (len(tasks) - 1)
                await asyncio.sleep(0.5)
                continue

            parsed = json.loads(raw) if provider in ("openai", "deepseek") else _extract_json(raw)

            # Always expect {"results": [...]} format
            results_list = parsed.get("results", [])

            # Fallback: if model returned {"picks": [...]} directly (single task)
            if not results_list and "picks" in parsed and len(tasks) == 1:
                results_list = [parsed]

            if len(results_list) != len(tasks):
                if verbose:
                    print(f"    [{batch_id}] got {len(results_list)} results for {len(tasks)} tasks, falling back")
                # Fall back to individual calls
                if len(tasks) > 1:
                    individual = []
                    for t in tasks:
                        individual.append(await _annotate_batch(client, [t], model_config, verbose))
                    return [r[0] for r in individual]
                return [(None, usage_info)]

            # Process each result
            outputs = []
            per_task_usage = None
            if usage_info and len(tasks) > 1:
                per_task_usage = {k: v // len(tasks) if isinstance(v, (int, float)) else v
                                  for k, v in usage_info.items()}
            else:
                per_task_usage = usage_info

            for task, result_data in zip(tasks, results_list):
                picks = result_data.get("picks", [])
                groups = task["groups"]

                if not _validate_picks(picks, groups):
                    if verbose:
                        print(f"    [{task['id']}] FAIL invalid picks: {picks}")
                    outputs.append((None, per_task_usage))
                    continue

                # Resolve 1-based indices to actual word strings
                resolved_picks = _resolve_picks(picks, groups)

                segments = _apply_merges(task["segments"], groups, resolved_picks)
                if segments is None:
                    if verbose:
                        print(f"    [{task['id']}] FAIL merge application failed")
                    outputs.append((None, per_task_usage))
                    continue

                text = "".join(segments)
                if verbose:
                    accepted = sum(1 for p in picks if p is not None)
                    print(f"    [{task['id']}] ✓ {accepted}/{len(picks)} merges accepted")

                outputs.append(({
                    "id": task["id"],
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
            if len(tasks) > 1:
                # Fall back to individual on non-transient error
                individual = []
                for t in tasks:
                    individual.append(await _annotate_batch(client, [t], model_config, verbose))
                return [r[0] for r in individual]
            return [(None, None)]

    return [(None, None)] * len(tasks)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def _run_async(
    tasks: list[dict], output_path: Path, model: str, concurrency: int,
    verbose: bool = False, tasks_per_call: int = 5,
) -> dict:
    """Run all merge tasks with fork-and-join concurrency."""
    model_config = _get_model_config(model)
    provider = model_config["provider"]
    pricing = model_config["pricing"]

    if provider == "openai":
        client = AsyncOpenAI(
            api_key=_load_openai_api_key(), timeout=REQUEST_TIMEOUT)
    elif provider == "deepseek":
        client = AsyncOpenAI(
            api_key=_load_deepseek_api_key(), base_url=_DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT)
    else:
        client = _get_bedrock_client()

    stats = {"ok": 0, "failed": 0}
    tokens = {"input": 0, "input_cached": 0, "output": 0}
    start = time.time()

    with open(output_path, "a", encoding="utf-8") as f:
        offset = 0
        while offset < len(tasks):
            call_batches = []
            for _ in range(concurrency):
                if offset >= len(tasks):
                    break
                end = min(offset + tasks_per_call, len(tasks))
                call_batches.append(tasks[offset:end])
                offset = end

            results_nested = await asyncio.gather(
                *[_annotate_batch(client, tb, model_config, verbose) for tb in call_batches]
            )

            for batch_results in results_nested:
                for result, usage_info in batch_results:
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stats["ok"] += 1
                    else:
                        stats["failed"] += 1

                    if usage_info:
                        cached = usage_info.get("cached_tokens", 0)
                        prompt = usage_info.get("prompt_tokens", 0)
                        tokens["input"] += prompt - cached
                        tokens["input_cached"] += cached
                        tokens["output"] += usage_info.get("completion_tokens", 0)

            f.flush()

            # Progress
            total = stats["ok"] + stats["failed"]
            if total % 100 < concurrency * tasks_per_call or verbose:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0

                cached_discount = pricing["cached_input"]
                input_cost = (tokens["input"] / 1_000_000 * pricing["input"] +
                              tokens["input_cached"] / 1_000_000 * cached_discount)
                output_cost = tokens["output"] / 1_000_000 * pricing["output"]
                spent = input_cost + output_cost
                est_total = spent * len(tasks) / total if total > 0 else 0

                print(
                    f"  [{total}/{len(tasks)}] "
                    f"ok={stats['ok']} fail={stats['failed']} "
                    f"| ${spent:.4f} (est ${est_total:.3f}) "
                    f"| {rate:.1f}/s "
                    f"| in={tokens['input']:,} cached={tokens['input_cached']:,} out={tokens['output']:,}"
                )

    elapsed = time.time() - start
    cached_discount = pricing["cached_input"]
    input_cost = (tokens["input"] / 1_000_000 * pricing["input"] +
                  tokens["input_cached"] / 1_000_000 * cached_discount)
    output_cost = tokens["output"] / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost
    print(
        f"  Done: {stats['ok']} annotated, {stats['failed']} failed "
        f"({elapsed:.0f}s, {total / elapsed:.1f}/s)\n"
        f"  Cost: ${total_cost:.4f}"
    )
    return {
        "ok": stats["ok"], "failed": stats["failed"], "cost": total_cost,
        "elapsed": elapsed, "tokens": dict(tokens),
    }


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------


def _run_batch_api(
    tasks: list[dict],
    output_path: Path,
    model: str,
    tasks_per_call: int = 5,
    verbose: bool = False,
    poll_interval: int = 30,
) -> dict:
    """Submit all tasks via OpenAI Batch API, poll for completion, write results.

    50% discount, results typically within minutes. Failures are discarded.
    Automatically splits into multiple batches if >50K requests.
    """
    from openai import OpenAI

    MAX_BATCH_REQUESTS = 50_000
    MAX_ENQUEUED_TOKENS = 1_800_000  # stay under typical 2M org limit

    model_config = _get_model_config(model)
    if model_config["provider"] != "openai":
        raise ValueError(f"OpenAI Batch API only supports 'openai' provider. Got: {model}")

    pricing = model_config["pricing"]
    api_model = model_config["api_model"]
    client = OpenAI(api_key=_load_openai_api_key())

    # Group tasks
    task_groups = []
    for i in range(0, len(tasks), tasks_per_call):
        task_groups.append(tasks[i:i + tasks_per_call])

    print(f"  Model: {model}, {len(tasks)} tasks in {len(task_groups)} requests ({tasks_per_call}/call)")

    # Split into chunks respecting both request count and token limits
    # Estimate ~1000 tokens per request (system prompt + user content)
    est_tokens_per_request = len(_SYSTEM_PROMPT) // 3 + 50  # rough: chars/3 for system + user
    max_requests_by_tokens = MAX_ENQUEUED_TOKENS // est_tokens_per_request
    max_per_chunk = min(MAX_BATCH_REQUESTS, max_requests_by_tokens)

    chunks = [task_groups[i:i + max_per_chunk]
              for i in range(0, len(task_groups), max_per_chunk)]
    if len(chunks) > 1:
        print(f"  Splitting into {len(chunks)} batches (max {max_per_chunk} requests each, ~{MAX_ENQUEUED_TOKENS:,} token limit)")

    # Shared state across chunks
    total_stats = {"ok": 0, "failed": 0}
    total_tokens = {"input": 0, "input_cached": 0, "output": 0}
    total_elapsed = 0

    use_new_api = "gpt-5" in api_model
    use_completion_tokens = use_new_api or "gpt-4.1" in api_model
    token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"
    max_tokens = MAX_OUTPUT_TOKENS_THINKING if use_new_api else MAX_OUTPUT_TOKENS

    for chunk_idx, chunk_groups in enumerate(chunks):
        if len(chunks) > 1:
            print(f"\n  --- Batch {chunk_idx + 1}/{len(chunks)} ({len(chunk_groups)} requests) ---")

        # Build JSONL batch file
        batch_input_path = output_path.with_suffix(f".batch_input_{chunk_idx}.jsonl")

        group_map = {}
        with open(batch_input_path, "w", encoding="utf-8") as f:
            for group in chunk_groups:
                custom_id = group[0]["id"].replace(":", "_").replace(".", "-")
                if len(group) > 1:
                    custom_id += f"--{group[-1]['id'].replace(':', '_').replace('.', '-')}"
                custom_id = custom_id[:64]
                group_map[custom_id] = group

                user_prompt = _format_batch_prompt(group)

                body = {
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    token_param: max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "merge_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "results": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "picks": {
                                                    "type": "array",
                                                    "items": {"type": ["integer", "null"]}
                                                }
                                            },
                                            "required": ["picks"],
                                            "additionalProperties": False,
                                        }
                                    }
                                },
                                "required": ["results"],
                                "additionalProperties": False,
                            }
                        }
                    },
                }
                if not use_new_api:
                    body["temperature"] = 0.0

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"  Uploading batch input ({len(chunk_groups)} requests)...")
        with open(batch_input_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")

        print(f"  Creating batch...")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Batch ID: {batch.id}")

        # Poll
        start = time.time()
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch.id)
            elapsed = time.time() - start
            counts = batch.request_counts
            completed = counts.completed if counts else 0
            failed = counts.failed if counts else 0
            total_reqs = counts.total if counts else len(chunk_groups)
            print(f"  [{elapsed:.0f}s] {batch.status} — {completed}/{total_reqs} done, {failed} failed")

        chunk_elapsed = time.time() - start
        total_elapsed += chunk_elapsed

        if batch.status != "completed":
            print(f"  Batch ended with status: {batch.status}")
            if batch.errors:
                for err in batch.errors.data[:5]:
                    print(f"    {err.code}: {err.message}")
            total_stats["failed"] += sum(len(g) for g in chunk_groups)
            batch_input_path.unlink(missing_ok=True)
            continue

        # Download results
        print(f"  Downloading results...")
        output_file = client.files.content(batch.output_file_id)
        result_lines = output_file.text.strip().split("\n")

        with open(output_path, "a", encoding="utf-8") as f_out:
            for line in result_lines:
                if not line.strip():
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                group = group_map.get(custom_id)
                if not group:
                    total_stats["failed"] += 1
                    continue

                response = result.get("response", {})
                if response.get("status_code") != 200:
                    if verbose:
                        print(f"    [{custom_id}] HTTP {response.get('status_code')}")
                    total_stats["failed"] += len(group)
                    continue

                resp_body = response.get("body", {})
                usage = resp_body.get("usage", {})
                total_tokens["input"] += usage.get("prompt_tokens", 0)
                total_tokens["output"] += usage.get("completion_tokens", 0)
                details = usage.get("prompt_tokens_details") or {}
                total_tokens["input_cached"] += details.get("cached_tokens", 0)

                choices = resp_body.get("choices", [])
                if not choices:
                    total_stats["failed"] += len(group)
                    continue

                raw = choices[0].get("message", {}).get("content", "")
                if not raw:
                    total_stats["failed"] += len(group)
                    continue

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    total_stats["failed"] += len(group)
                    continue

                results_list = parsed.get("results", [])
                if not results_list and "picks" in parsed and len(group) == 1:
                    results_list = [parsed]

                if len(results_list) != len(group):
                    if verbose:
                        print(f"    [{custom_id}] got {len(results_list)} results for {len(group)} tasks")
                    total_stats["failed"] += len(group)
                    continue

                for task, result_data in zip(group, results_list):
                    picks = result_data.get("picks", [])
                    groups = task["groups"]

                    if not _validate_picks(picks, groups):
                        total_stats["failed"] += 1
                        continue

                    resolved_picks = _resolve_picks(picks, groups)
                    segments = _apply_merges(task["segments"], groups, resolved_picks)
                    if segments is None:
                        total_stats["failed"] += 1
                        continue

                    text = "".join(segments)
                    f_out.write(json.dumps({
                        "id": task["id"],
                        "source": task["source"],
                        "text": text,
                        "segments": segments,
                    }, ensure_ascii=False) + "\n")
                    total_stats["ok"] += 1

        # Cleanup batch input file
        batch_input_path.unlink(missing_ok=True)

    # Cost with 50% batch discount
    input_cost = total_tokens["input"] / 1_000_000 * pricing["input"] * 0.5
    cached_cost = total_tokens["input_cached"] / 1_000_000 * pricing["cached_input"] * 0.5
    output_cost = total_tokens["output"] / 1_000_000 * pricing["output"] * 0.5
    total_cost = input_cost + cached_cost + output_cost

    print(f"\n  Done: {total_stats['ok']} annotated, {total_stats['failed']} failed ({total_elapsed:.0f}s)")
    print(f"  Cost: ${total_cost:.4f} (50% batch discount applied)")
    print(f"    input={total_tokens['input']:,} cached={total_tokens['input_cached']:,} output={total_tokens['output']:,}")

    return {"ok": total_stats["ok"], "failed": total_stats["failed"],
            "elapsed": total_elapsed, "cost": total_cost}


def annotate_merges(
    tasks_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
    concurrency: int = DEFAULT_CONCURRENCY,
    verbose: bool = False,
    tasks_per_call: int = 5,
    dry_run_mode: bool = False,
    batch: bool = False,
) -> None:
    """Annotate ICWB2 merge tasks."""
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]

    if not tasks:
        print("  No tasks to annotate.")
        return

    # Resume: skip already-completed IDs
    done_ids = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    if done_ids:
        tasks = [t for t in tasks if t["id"] not in done_ids]
        print(f"  Resuming: {len(done_ids)} done, {len(tasks)} remaining")

    if not tasks:
        print("  All tasks already completed.")
        return

    if dry_run_mode:
        _dry_run(tasks, output_path, model)
        return

    if batch:
        print(f"  Using OpenAI Batch API (50% discount)")
        _run_batch_api(tasks, output_path, model, tasks_per_call=tasks_per_call, verbose=verbose)
    else:
        print(f"  Model: {model}, tasks: {len(tasks)}, concurrency: {concurrency}, batch: {tasks_per_call}")
        asyncio.run(_run_async(tasks, output_path, model, concurrency, verbose, tasks_per_call))


def _dry_run(tasks: list[dict], output_path: Path, model: str):
    """Estimate cost without making API calls."""
    pricing = _get_model_config(model)["pricing"]

    # Estimate tokens (Chinese chars ≈ 1 token each for most models)
    system_tokens = len(_SYSTEM_PROMPT)  # rough
    total_input_tokens = 0
    for task in tasks:
        prompt = _format_user_prompt(task)
        total_input_tokens += system_tokens + len(prompt)

    # Cached: system prompt repeated, 50% discount (OpenAI) or 90% (Anthropic)
    cached = system_tokens * (len(tasks) - 1)
    uncached = total_input_tokens - cached
    cached_rate = pricing["cached_input"]

    input_cost = uncached / 1_000_000 * pricing["input"] + cached / 1_000_000 * cached_rate
    # Output: ~15 tokens per task (tiny JSON)
    est_output = len(tasks) * 15
    output_cost = est_output / 1_000_000 * pricing["output"]

    print(f"  Model: {model}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Est input tokens: {total_input_tokens:,} (cached: {cached:,})")
    print(f"  Est output tokens: {est_output:,}")
    print(f"  Est cost: ${input_cost + output_cost:.4f}")

    # Dump samples
    sample_path = output_path.with_suffix(".dry_run.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(f"=== SYSTEM PROMPT ({len(_SYSTEM_PROMPT)} chars) ===\n\n{_SYSTEM_PROMPT}\n\n")
        for i, task in enumerate(tasks[:5]):
            f.write(f"=== TASK {task['id']} ===\n{_format_user_prompt(task)}\n\n")
    print(f"  Samples → {sample_path.name}")
