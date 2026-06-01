"""
Smoke test: DeepSeek V4 API for CWS. Tests both models, JSON mode, and caching.
Usage: python smoke_test_api.py
"""

import json
import os
from pathlib import Path
from openai import OpenAI

API_KEY_FILE = Path(__file__).parents[1] / "DEEPSEEK_API_DO_NOT_COMMIT.txt"
BASE_URL = "https://api.deepseek.com"
MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]

SYSTEM_PROMPT = """\
你是一个专业的中文分词专家。给定一个中文句子，请将其准确分词。

规则：
- 优先识别多字词（如"研究生"、"人工智能"），避免过度切分
- 成语、固定搭配、人名、地名作为整体
- 遇到歧义时根据上下文语义选择最合理的切分
- 标点符号单独作为一个token
- 重叠词保持完整（如"高高兴兴"、"看看"）

输出严格的JSON格式，包含一个"segments"字段。

示例：
输入：今天天气真好，适合出去走走。
输出：{"segments": ["今天", "天气", "真", "好", "，", "适合", "出去", "走走", "。"]}

输入：南京市长江大桥于1968年建成通车。
输出：{"segments": ["南京市", "长江大桥", "于", "1968年", "建成", "通车", "。"]}"""


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    raise RuntimeError(f"No API key. Set DEEPSEEK_API_KEY or create {API_KEY_FILE}")


def call_cws(client: OpenAI, model: str, sentence: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
        max_tokens=512,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    usage = resp.usage
    return {
        "output": resp.choices[0].message.content,
        "cache_hit": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        "cache_miss": getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
        "prompt_tokens": usage.prompt_tokens,
    }


def main():
    client = OpenAI(api_key=load_api_key(), base_url=BASE_URL)

    sentences = [
        "他在研究生命的意义和研究生的区别。",
        "南京市长江大桥于一九六八年建成通车。",
        "这个门把手坏了，需要换一个新的。",
        "我们今天下午三点在会议室开会讨论方案。",
    ]

    for model in MODELS:
        print(f"\n{'='*50}\n{model}\n{'='*50}")
        for i, sent in enumerate(sentences, 1):
            for attempt in range(3):
                result = call_cws(client, model, sent)
                try:
                    parsed = json.loads(result["output"])
                    segs = " / ".join(parsed["segments"])
                    break
                except (json.JSONDecodeError, KeyError, TypeError):
                    if attempt == 2:
                        segs = f"FAILED (raw: {result['output']!r})"
                    continue
            print(
                f"  [{i}] {sent}\n"
                f"      → {segs}\n"
                f"      cache: hit={result['cache_hit']} miss={result['cache_miss']}"
            )


if __name__ == "__main__":
    main()
