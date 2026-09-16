#!/usr/bin/env python3
"""Gemma 4 E4B 图文请求示例。把图片编成 data URL 发给 FLM。"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8219")
MODEL = os.environ.get("MODEL", "gemma4-it:e4b")


def data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=pathlib.Path)
    parser.add_argument("prompt", nargs="?", default="用中文描述这张图。")
    args = parser.parse_args()
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_url(args.image)}},
                ],
            }
        ],
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
