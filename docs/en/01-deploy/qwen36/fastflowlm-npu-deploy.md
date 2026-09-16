## FastFlowLM NPU deployment: Qwen3.6-35B-A3B

Deploy **Qwen3.6-35B-A3B** on Ubuntu 24.04 + **XDNA2** with FastFlowLM.

> Prerequisite: [FastFlowLM environment](/00-environment/fastflowlm.md). Model notes: [Qwen3.6](./qwen36_model.md). GPU: [Lemonade GPU](./lemonade-gpu-deploy.md). Gemma 4 counterpart: [Gemma 4 FastFlowLM](/01-deploy/gemma4/fastflowlm-npu-deploy.md).

| Item | Value |
|:---|:---|
| Tag | `qwen3.6-moe:35b-a3b` |
| Format | NPU2 (not GGUF) |
| API | `http://127.0.0.1:8219/v1` |
| Resident | ~**29 GiB** |
| Throughput | ~**14 tok/s** long-form on the validation host |

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm pull qwen3.6-moe:35b-a3b --modelscope 1
flm check qwen3.6-moe:35b-a3b

lemonade unload 2>/dev/null || true
pkill -x flm 2>/dev/null || true
flm serve qwen3.6-moe:35b-a3b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

Smoke: `BASE=http://127.0.0.1:8219 MODEL=qwen3.6-moe:35b-a3b python3 src/npu/examples/chat_gemma4.py`. Stop: `pkill -x flm`. Queue overflow returns 503 — lower concurrency or raise `--q-len`.
