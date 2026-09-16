#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8219}"
MODEL="${MODEL:-gemma4-it:e4b}"
PROMPT="${PROMPT:-只回答一个词：ok}"
export MODEL PROMPT

case "${BASE}" in
  */v1) API_ROOT="${BASE}" ;;
  *) API_ROOT="${BASE}/v1" ;;
esac

echo "GET ${API_ROOT}/models"
curl -sS "${API_ROOT}/models"
echo

echo "POST ${API_ROOT}/chat/completions model=${MODEL}"
curl -sS "${API_ROOT}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "max_tokens": 16, "temperature": 0}))')"
echo
