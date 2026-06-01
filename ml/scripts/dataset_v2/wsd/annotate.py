"""LLM-annotate WSD tasks → wsd_results.jsonl.

Reads wsd_tasks.jsonl, sends sentence + clusters to LLM, gets sense index back.

Reuses the same provider infrastructure from cws/annotate.py (DeepSeek, Bedrock,
Anthropic direct). Supports batching multiple WSD tasks per API call.

Usage:
  from wsd.annotate import annotate_wsd
  annotate_wsd(tasks_path, output_path, model="deepseek-v4-flash")
"""
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

_DIR = Path(__file__).parent
_SYSTEM_PROMPT = (_DIR / "system_prompt.md").read_text(encoding="utf-8")
_API_KEY_FILE = _DIR.parent / "DEEPSEEK_API_DO_NOT_COMMIT.txt"
_ANTHROPIC_API_KEY_FILE = _DIR.parent / "ANTHROPIC_API_DO_NOT_COMMIT.txt"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Bedrock config
_BEDROCK_REGION = "us-east-1"
_BEDROCK_PROFILE = "olivezh-aws-profile"

DEFAULT_CONCURRENCY = 16
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]
REQUEST_TIMEOUT = 120
MAX_OUTPUT_TOKENS = 4096  # WSD responses are small but batched sentences need room
MAX_OUTPUT_TOKENS_THINKING = 16384

# ---------------------------------------------------------------------------
# Model registry (same as cws/annotate.py)
# ---------------------------------------------------------------------------

_MODELS = {
    # DeepSeek
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "api_model": "deepseek-v4-flash",
        "pricing": {"input": 0.14, "output": 0.28},
        "thinking": False,
    },
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
    # Anthropic via Bedrock
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
        "effort": "medium",
    },
    "claude-haiku-4.5": {
        "provider": "bedrock",
        "api_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "pricing": {"input": 0.80, "output": 4.0},
        "thinking": False,
    },
    # OpenAI
    "gpt-4o": {
        "provider": "openai",
        "api_model": "gpt-4o",
        "pricing": {"input": 2.50, "output": 10.00},
        "thinking": False,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "pricing": {"input": 0.15, "output": 0.60},
        "thinking": False,
    },
    "gpt-5-mini": {
        "provider": "openai",
        "api_model": "gpt-5-mini",
        "pricing": {"input": 0.25, "output": 2.00},
        "thinking": True,
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "api_model": "gpt-4.1-mini",
        "pricing": {"input": 0.40, "output": 1.60},
        "thinking": False,
    },
    # Anthropic direct API (for Batch API support)
    "sonnet-4.6": {
        "provider": "anthropic",
        "api_model": "claude-sonnet-4-6",
        "pricing": {"input": 3.0, "output": 15.0},
        "thinking": False,
    },
}


def _get_model_config(model: str) -> dict:
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


def _load_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    _oai_file = _DIR.parent / "OPENAI_API_DO_NOT_COMMIT.txt"
    if _oai_file.exists():
        return _oai_file.read_text().strip()
    raise RuntimeError(f"No OpenAI API key. Set OPENAI_API_KEY or create {_oai_file}")


def _get_bedrock_client(read_timeout: int = 300):
    import boto3
    from botocore.config import Config
    config = Config(read_timeout=read_timeout, connect_timeout=10, retries={"max_attempts": 0})
    session = boto3.Session(profile_name=_BEDROCK_PROFILE, region_name=_BEDROCK_REGION)
    return session.client("bedrock-runtime", config=config)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_user_prompt(task: dict) -> str:
    """Build user prompt for a single sentence-level WSD task.

    Task schema: {"sentence": str, "words": [{"word", "pos", "clusters"}, ...]}
    """
    sentence = task["sentence"]
    words = task["words"]

    # Insert ★ markers around all target words (process right-to-left to preserve positions)
    marked = sentence
    for w in sorted(words, key=lambda x: x["pos"], reverse=True):
        pos = w["pos"]
        word = w["word"]
        marked = marked[:pos] + f"★{word}★" + marked[pos + len(word):]

    lines = [f"Sentence: {marked}"]
    lines.append("Words:")
    for wi, w in enumerate(words, 1):
        word = w["word"]
        pos = w["pos"]
        lines.append(f"  [{wi}] {word} (pos {pos})")
        for cluster in w["clusters"]:
            idx = cluster["idx"]
            pinyin = cluster.get("pinyin", "")
            senses_en = cluster.get("senses_en", [])
            en_str = "; ".join(senses_en) if senses_en else ""
            if pinyin and en_str:
                lines.append(f"    {idx}. ({pinyin}) {en_str}")
            elif en_str:
                lines.append(f"    {idx}. {en_str}")
            else:
                lines.append(f"    {idx}. ({pinyin})")

    return "\n".join(lines)


def _format_batch_prompt(tasks: list[dict]) -> str:
    """Build batched user prompt for multiple sentence-level WSD tasks."""
    if len(tasks) == 1:
        return _format_user_prompt(tasks[0])

    lines = []
    for i, task in enumerate(tasks, 1):
        lines.append(f"[{i}]")
        lines.append(_format_user_prompt(task))
        lines.append("")

    lines.append(f'Respond with JSON: {{"results": [{{"senses": [...]}}, ...]}}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API calls (reused from cws pipeline)
# ---------------------------------------------------------------------------


async def _call_deepseek(
    client: AsyncOpenAI, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    thinking: bool = False, max_tokens: int | None = None,
) -> tuple[str | None, dict | None, str | None]:
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
    finish_reason = resp.choices[0].finish_reason
    return raw, usage_info, finish_reason


async def _call_bedrock(
    bedrock_client, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    thinking: str | bool = False, effort: str = "medium",
    max_tokens: int | None = None,
) -> tuple[str | None, dict | None, str | None]:
    if max_tokens is None:
        max_tokens = MAX_OUTPUT_TOKENS_THINKING if thinking else MAX_OUTPUT_TOKENS

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": [
            {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
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
        "cache_hit": usage.get("cache_read_input_tokens", 0),
        "cache_miss": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
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


async def _call_openai(
    client: AsyncOpenAI, model_id: str, user_prompt: str, verbose: bool, task_id: str,
    max_tokens: int | None = None,
) -> tuple[str | None, dict | None, str | None]:
    """Call OpenAI API."""
    if max_tokens is None:
        max_tokens = MAX_OUTPUT_TOKENS

    use_new_api = "gpt-5" in model_id
    use_completion_tokens = use_new_api or "gpt-4.1" in model_id
    token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"

    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        token_param: max_tokens,
        "response_format": {"type": "json_object"},
    }
    if not use_new_api:
        kwargs["temperature"] = 0.0

    resp = await client.chat.completions.create(**kwargs)
    usage = resp.usage
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cache_hit": cached,
        "cache_miss": usage.prompt_tokens - cached,
    }
    raw = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    return raw, usage_info, finish_reason


# ---------------------------------------------------------------------------
# Annotation logic
# ---------------------------------------------------------------------------


async def _annotate_batch(
    client, tasks: list[dict], model_config: dict, verbose: bool = False,
) -> list[tuple[dict | None, dict | None]]:
    """Annotate a batch of WSD tasks. Returns list of (result, usage_info)."""
    provider = model_config["provider"]
    model_id = model_config["api_model"]
    user_prompt = _format_batch_prompt(tasks)
    batch_id = tasks[0]["id"] if len(tasks) == 1 else f"batch[{tasks[0]['id']}..{tasks[-1]['id']}]"

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"    [{batch_id}] attempt {attempt+1} ({len(tasks)} tasks)")

            batch_max = MAX_OUTPUT_TOKENS_THINKING if model_config.get("thinking") else min(MAX_OUTPUT_TOKENS * len(tasks), 16384)

            if provider == "deepseek":
                raw, usage_info, finish_reason = await _call_deepseek(
                    client, model_id, user_prompt, verbose, batch_id,
                    thinking=model_config.get("thinking", False),
                    max_tokens=batch_max)
            elif provider == "openai":
                raw, usage_info, finish_reason = await _call_openai(
                    client, model_id, user_prompt, verbose, batch_id,
                    max_tokens=batch_max)
            else:
                raw, usage_info, finish_reason = await _call_bedrock(
                    client, model_id, user_prompt, verbose, batch_id,
                    thinking=model_config.get("thinking", False),
                    effort=model_config.get("effort", "medium"),
                    max_tokens=batch_max)

            if not raw:
                if finish_reason in ("length", "max_tokens"):
                    if verbose:
                        print(f"    [{batch_id}] FAIL truncated")
                    return [(None, usage_info)] + [(None, None)] * (len(tasks) - 1)
                await asyncio.sleep(0.5)
                continue

            parsed = json.loads(raw) if provider in ("deepseek", "openai") else _extract_json(raw)

            # Handle single vs batch response
            if len(tasks) == 1:
                results_list = [parsed] if "senses" in parsed else parsed.get("results", [])
            else:
                results_list = parsed.get("results", [])

            if len(results_list) != len(tasks):
                if verbose:
                    print(f"    [{batch_id}] got {len(results_list)} results for {len(tasks)} tasks")
                # Fall back to individual
                if len(tasks) > 1:
                    individual = []
                    for t in tasks:
                        individual.extend(await _annotate_batch(client, [t], model_config, verbose))
                    return individual
                return [(None, usage_info)]

            # Process results
            outputs = []
            per_task_usage = None
            if usage_info and len(tasks) > 1:
                per_task_usage = {k: v // len(tasks) if isinstance(v, (int, float)) else v
                                  for k, v in usage_info.items()}
            else:
                per_task_usage = usage_info

            for task, result_data in zip(tasks, results_list):
                senses = result_data.get("senses", [])
                words = task["words"]

                # Validate: must have one sense per word
                if len(senses) != len(words):
                    if verbose:
                        print(f"    [{task['id']}] FAIL got {len(senses)} senses for {len(words)} words")
                    outputs.append((None, per_task_usage))
                    continue

                # Validate each sense index
                valid = True
                for sense_idx, w in zip(senses, words):
                    n_clusters = len(w["clusters"])
                    if not isinstance(sense_idx, int) or sense_idx < 1 or sense_idx > n_clusters:
                        if verbose:
                            print(f"    [{task['id']}] FAIL invalid sense {sense_idx} for {w['word']} (has {n_clusters} clusters)")
                        valid = False
                        break

                if not valid:
                    outputs.append((None, per_task_usage))
                    continue

                if verbose:
                    word_senses = ", ".join(f"{w['word']}={s}" for w, s in zip(words, senses))
                    print(f"    [{task['id']}] ✓ {word_senses}")

                outputs.append(({
                    "id": task["id"],
                    "source": task["source"],
                    "sentence": task["sentence"],
                    "words": [
                        {
                            "word": w["word"],
                            "pos": w["pos"],
                            "sense": s,
                            "n_clusters": len(w["clusters"]),
                        }
                        for w, s in zip(words, senses)
                    ],
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
                individual = []
                for t in tasks:
                    individual.extend(await _annotate_batch(client, [t], model_config, verbose))
                return individual
            return [(None, None)]

    return [(None, None)] * len(tasks)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def _run_async(
    tasks: list[dict], output_path: Path, model: str, concurrency: int,
    verbose: bool = False, tasks_per_call: int = 10,
) -> dict:
    """Run all WSD tasks with fork-and-join concurrency."""
    model_config = _get_model_config(model)
    provider = model_config["provider"]
    pricing = model_config["pricing"]

    if provider == "deepseek":
        client = AsyncOpenAI(
            api_key=_load_deepseek_api_key(), base_url=_DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT)
    elif provider == "openai":
        client = AsyncOpenAI(
            api_key=_load_openai_api_key(), timeout=REQUEST_TIMEOUT)
    else:
        per_task_timeout = 60 if not model_config.get("thinking") else 120
        client = _get_bedrock_client(read_timeout=per_task_timeout * tasks_per_call)

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
                        tokens["input"] += usage_info.get("cache_miss", 0)
                        tokens["input_cached"] += usage_info.get("cache_hit", 0)
                        tokens["output"] += usage_info.get("completion_tokens", 0)

            f.flush()

            # Progress
            total = stats["ok"] + stats["failed"]
            if total % 200 < concurrency * tasks_per_call or verbose:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0

                input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"]
                output_cost = tokens["output"] / 1_000_000 * pricing["output"]
                spent = input_cost + output_cost
                est_total = spent * len(tasks) / total if total > 0 else 0

                print(
                    f"  [{total}/{len(tasks)}] "
                    f"ok={stats['ok']} fail={stats['failed']} "
                    f"| ${spent:.3f} (est ${est_total:.2f}) "
                    f"| {rate:.1f}/s "
                    f"| in={tokens['input']:,} cached={tokens['input_cached']:,} out={tokens['output']:,}"
                )

    elapsed = time.time() - start
    input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"]
    output_cost = tokens["output"] / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost
    print(
        f"  Done: {stats['ok']} labeled, {stats['failed']} failed "
        f"({elapsed:.0f}s, {(stats['ok']+stats['failed'])/elapsed:.1f}/s)\n"
        f"  Cost: ${total_cost:.3f}"
    )
    return {"ok": stats["ok"], "failed": stats["failed"], "cost": total_cost,
            "elapsed": elapsed, "tokens": dict(tokens)}


# ---------------------------------------------------------------------------
# Anthropic Batch API
# ---------------------------------------------------------------------------


def _sanitize_custom_id(raw_id: str) -> str:
    """Sanitize a task ID to match ^[a-zA-Z0-9_-]{1,64}$"""
    import hashlib
    # Just use a hash — simple, deterministic, always valid
    return hashlib.md5(raw_id.encode()).hexdigest()


def _build_wsd_batch_request(task_group: list[dict], model_config: dict) -> dict:
    """Build a single batch request item for the Anthropic Batch API (WSD)."""
    user_prompt = _format_batch_prompt(task_group)

    # Build custom_id (must match ^[a-zA-Z0-9_-]{1,64}$)
    if len(task_group) == 1:
        custom_id = _sanitize_custom_id(task_group[0]["id"])
    else:
        combined = task_group[0]["id"] + "|" + task_group[-1]["id"]
        custom_id = _sanitize_custom_id(combined)

    max_tokens = MAX_OUTPUT_TOKENS * len(task_group)

    params = {
        "model": model_config["api_model"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "system": [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }

    return {"custom_id": custom_id, "params": params}


def run_batch_api(
    tasks: list[dict],
    output_path: Path,
    model: str,
    tasks_per_call: int = 5,
    verbose: bool = False,
    poll_interval: int = 30,
) -> dict:
    """Submit all WSD tasks via Anthropic Batch API, poll for completion, write results.

    Returns stats dict.
    """
    import anthropic

    model_config = _get_model_config(model)
    if model_config["provider"] != "anthropic":
        raise ValueError(f"Batch API only supports 'anthropic' provider models. Got: {model} ({model_config['provider']})")

    pricing = model_config["pricing"]
    batch_discount = 0.5

    # Load API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and _ANTHROPIC_API_KEY_FILE.exists():
        api_key = _ANTHROPIC_API_KEY_FILE.read_text().strip()
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Group tasks
    task_groups = []
    for i in range(0, len(tasks), tasks_per_call):
        task_groups.append(tasks[i:i + tasks_per_call])

    print(f"  Model: {model}, {len(tasks)} tasks, {tasks_per_call} per call → {len(task_groups)} requests")
    print(f"  Building batch requests...")

    # Build requests
    requests = []
    for group in task_groups:
        req = _build_wsd_batch_request(group, model_config)
        requests.append(req)

    # Map custom_id → task group
    group_map = {}
    for group in task_groups:
        if len(group) == 1:
            key = _sanitize_custom_id(group[0]["id"])
        else:
            combined = group[0]["id"] + "|" + group[-1]["id"]
            key = _sanitize_custom_id(combined)
        group_map[key] = group

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

    with open(output_path, "a", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            group = group_map.get(custom_id)

            if not group:
                if verbose:
                    print(f"    [{custom_id}] unknown custom_id, skipping")
                continue

            if result.result.type == "errored":
                if verbose:
                    print(f"    [{custom_id}] ERROR: {result.result.error}")
                stats["failed"] += len(group)
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
                stats["failed"] += len(group)
                continue

            try:
                parsed = _extract_json(raw)
            except Exception as e:
                if verbose:
                    print(f"    [{custom_id}] JSON parse error: {e}")
                stats["failed"] += len(group)
                continue

            # Handle single vs multi-task results
            if len(group) == 1:
                results_list = [parsed] if "senses" in parsed else parsed.get("results", [])
            else:
                results_list = parsed.get("results", [])

            if len(results_list) != len(group):
                if verbose:
                    print(f"    [{custom_id}] got {len(results_list)} results for {len(group)} tasks")
                stats["failed"] += len(group)
                continue

            # Validate and write each task result
            for task, result_data in zip(group, results_list):
                senses = result_data.get("senses", [])
                words = task["words"]

                if len(senses) != len(words):
                    if verbose:
                        print(f"    [{task['id']}] FAIL got {len(senses)} senses for {len(words)} words")
                    stats["failed"] += 1
                    continue

                valid = True
                for sense_idx, w in zip(senses, words):
                    n_clusters = len(w["clusters"])
                    if not isinstance(sense_idx, int) or sense_idx < 1 or sense_idx > n_clusters:
                        if verbose:
                            print(f"    [{task['id']}] FAIL invalid sense {sense_idx} for {w['word']}")
                        valid = False
                        break

                if not valid:
                    stats["failed"] += 1
                    continue

                out = {
                    "id": task["id"],
                    "source": task["source"],
                    "sentence": task["sentence"],
                    "words": [
                        {
                            "word": w["word"],
                            "pos": w["pos"],
                            "sense": s,
                            "n_clusters": len(w["clusters"]),
                        }
                        for w, s in zip(words, senses)
                    ],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["ok"] += 1

    # Cost (with 50% batch discount)
    input_cost = (tokens["input"] + tokens["input_cached"] * 0.1) / 1_000_000 * pricing["input"] * batch_discount
    output_cost = tokens["output"] / 1_000_000 * pricing["output"] * batch_discount
    total_cost = input_cost + output_cost

    print(f"  Done: {stats['ok']} labeled, {stats['failed']} failed ({elapsed:.0f}s)")
    print(f"  Cost: ${total_cost:.3f} (with 50% batch discount)")
    print(f"    input={tokens['input']:,} cached={tokens['input_cached']:,} output={tokens['output']:,}")

    return {"ok": stats["ok"], "failed": stats["failed"], "cost": total_cost,
            "elapsed": elapsed, "tokens": dict(tokens)}


def annotate_wsd(
    tasks_path: Path,
    output_path: Path,
    model: str = "deepseek-v4-flash",
    concurrency: int = DEFAULT_CONCURRENCY,
    verbose: bool = False,
    tasks_per_call: int = 10,
    dry_run_mode: bool = False,
    batch: bool = False,
    limit: int | None = None,
) -> None:
    """Annotate WSD tasks with LLM.

    Args:
        tasks_path: Input JSONL (wsd_tasks.jsonl)
        output_path: Output JSONL (wsd_results.jsonl)
        model: Model short name
        concurrency: Parallel API calls
        verbose: Print per-task details
        tasks_per_call: How many WSD tasks to batch per API call
        dry_run_mode: If True, just estimate cost
        batch: If True, use Anthropic Batch API (50% discount)
        limit: If set, only process first N tasks (for testing)
    """
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]

    # Resume: skip already-completed IDs
    done_ids = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])

    remaining = [t for t in tasks if t["id"] not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} done, {len(remaining)} remaining")

    if limit and len(remaining) > limit:
        remaining = remaining[:limit]
        print(f"  Limited to {limit} tasks (testing)")

    if not remaining:
        print("  All tasks already completed!")
        return

    if dry_run_mode:
        _dry_run(remaining, model)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch:
        print(f"  Using Anthropic Batch API (50% discount)")
        run_batch_api(remaining, output_path, model, tasks_per_call=tasks_per_call, verbose=verbose)
    else:
        print(f"  Model: {model}, {len(remaining)} tasks, concurrency={concurrency}, "
              f"batch={tasks_per_call}/call")
        asyncio.run(_run_async(remaining, output_path, model, concurrency, verbose, tasks_per_call))


def _dry_run(tasks: list[dict], model: str, n_samples: int = 5):
    """Estimate cost without making API calls."""
    config = _get_model_config(model)
    pricing = config["pricing"]

    system_tokens = len(_SYSTEM_PROMPT)
    total_input = 0
    total_words = 0
    for task in tasks:
        prompt = _format_user_prompt(task)
        total_input += system_tokens + len(prompt)
        total_words += len(task["words"])

    est_output = total_words * 8  # {"senses": [N, ...]} is tiny per word

    # With caching
    cached_input = system_tokens * (len(tasks) - 1)
    uncached_input = total_input - cached_input
    effective_input = uncached_input + cached_input * 0.1

    input_cost = effective_input / 1_000_000 * pricing["input"]
    output_cost = est_output / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost

    print(f"  Model: {model}")
    print(f"  Sentence tasks: {len(tasks)} ({total_words:,} word disambiguations)")
    print(f"  Est input: ~{int(effective_input):,} effective tokens")
    print(f"  Est output: ~{est_output:,} tokens")
    print(f"  Est cost: ${total_cost:.2f} (in ${input_cost:.2f} + out ${output_cost:.2f})")

    # Sample prompts
    print(f"\n  === Sample prompts ===")
    for task in tasks[:n_samples]:
        n_words = len(task["words"])
        word_list = ", ".join(w["word"] for w in task["words"])
        print(f"\n  --- {task['id']} ({n_words} words: {word_list}) ---")
        print(f"  {_format_user_prompt(task)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-annotate WSD tasks")
    parser.add_argument("--input", type=Path, required=True, help="WSD tasks JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output results JSONL")
    parser.add_argument("--model", "-m", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--tasks-per-call", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API (50%% discount)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N tasks (for testing)")
    args = parser.parse_args()

    annotate_wsd(args.input, args.output, model=args.model,
                 concurrency=args.concurrency, verbose=args.verbose,
                 tasks_per_call=args.tasks_per_call, dry_run_mode=args.dry_run,
                 batch=args.batch, limit=args.limit)
