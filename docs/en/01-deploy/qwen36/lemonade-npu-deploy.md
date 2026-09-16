## Lemonade NPU deployment: Qwen3.6-35B-A3B

Same caveat as Gemma 4: systemd `lemond` cannot see the user FLM cache. Use [FastFlowLM Qwen3.6](./fastflowlm-npu-deploy.md).

```bash
lemonade unload 2>/dev/null || true
pkill -x flm 2>/dev/null || true
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve qwen3.6-moe:35b-a3b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

Resident ~29 GiB — stop GPU-side large models first. `lemonade load qwen3.6-moe:35b-a3b` is only worth trying if `lemond` runs as your login user without ProtectHome and `flm:npu` is installed.
