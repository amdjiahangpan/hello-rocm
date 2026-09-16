#!/usr/bin/env bash
set -euo pipefail

if ! command -v lemonade >/dev/null 2>&1; then
  echo "CHECK_LEMONADE_FAIL: lemonade not in PATH. See docs/zh/00-environment/lemonade.md" >&2
  exit 1
fi

lemonade --version
lemonade backends || true
lemonade status || true
if command -v ss >/dev/null 2>&1; then
  ss -tlnp | grep 13305 || echo "13305 not listening (start lemond if you need the server)"
fi
echo "CHECK_LEMONADE_OK"
