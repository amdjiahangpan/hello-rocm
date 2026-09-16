#!/usr/bin/env bash
set -euo pipefail

unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH || true

if ! command -v flm >/dev/null 2>&1; then
  echo "CHECK_FLM_FAIL: flm not in PATH. See docs/zh/00-environment/fastflowlm.md" >&2
  exit 1
fi

if [[ ! -e /dev/accel/accel0 ]]; then
  echo "CHECK_FLM_FAIL: /dev/accel/accel0 missing" >&2
  exit 1
fi

echo "ulimit -l: $(ulimit -l)"
ls -l /dev/accel/accel0
flm validate
echo "CHECK_FLM_OK"
