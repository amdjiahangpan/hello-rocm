#!/usr/bin/env python3
"""对 FastFlowLM OpenAI 兼容接口发一条短问。默认 Gemma 4 E4B @ :8219。"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8219")
MODEL = os.environ.get("MODEL", "gemma4-it:e4b")
PROMPT = os.environ.get("PROMPT", "用一句话介绍你自己。")


def main() -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是简洁的助手。"},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": 128,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
