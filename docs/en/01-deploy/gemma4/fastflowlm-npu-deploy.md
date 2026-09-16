## FastFlowLM NPU deployment (Ubuntu 24.04 + XDNA2)

This page deploys Gemma 4 E4B-it on Ubuntu 24.04 + **XDNA2 NPU** with **FastFlowLM (`flm`)** and an OpenAI-compatible API. It is not vLLM / Ollama and does not use the ROCm GPU stack.

> Prerequisite: [FastFlowLM environment](/00-environment/fastflowlm.md) (`flm validate` must pass).
>
> Model background: [Gemma 4 introduction](./gemma4_model.md). GPU counterparts: [LM Studio](./lm-studio-rocm7-deploy.md) · [Lemonade GPU](./lemonade-gpu-deploy.md). Wrap with Lemonade: [Lemonade NPU](./lemonade-npu-deploy.md). Measured throughput: [results](./fastflowlm-npu-results.md).

- **Audience**: you already have XDNA2 and want a local multimodal model
- **Difficulty**: ⭐⭐
- **Time**: weights ~8.7 GiB; smoke test ~1 minute after load
- **Validated on**: GMKtec EVO-X2 / Ryzen AI MAX+ 395, FLM 1.0.5

---

### What you get

| Item | Value |
|:---|:---|
| Tag | `gemma4-it:e4b` |
| Quant | NPU2 / Q4_1 (text + vision + audio), **not** GGUF |
| Listen | `http://0.0.0.0:8219/v1` |
| Context | 32768 default (max 128k) |
| Resident | ~9 GiB |

Do not feed GPU `Q4_K_M.gguf` to `flm`.

---

### 1. Prerequisite

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
```

Firmware should be **1.1.x**. See [NPU troubleshooting](/00-environment/npu-troubleshooting.md) on failure. Never `source /opt/rocm` before `flm`.

---

### 2. Pull weights

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm pull gemma4-it:e4b --modelscope 1
flm check gemma4-it:e4b
flm list --filter installed
```

Drop `--modelscope` for Hugging Face ([FastFlowLM/Gemma4-E4B-IT-NPU2](https://huggingface.co/FastFlowLM/Gemma4-E4B-IT-NPU2)). Cache lives in `~/.config/flm/models/`.

---

### 3. Serve

One LLM on NPU at a time:

```bash
lemonade unload 2>/dev/null || true
sudo systemctl stop lemond 2>/dev/null || true
pkill -x flm 2>/dev/null || true

unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve gemma4-it:e4b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

| Flag | Meaning |
|:---|:---|
| `--ctx-len` | context; rounded up to a power of two, min 512 |
| `--pmode` | `powersaver` / `balanced` / `performance` / `turbo` |
| `--q-len` | request queue; full → 503 |

---

### 4. Smoke test

```bash
curl -s http://127.0.0.1:8219/v1/models

python3 src/npu/examples/chat_gemma4.py
python3 src/npu/examples/chat_gemma4_image.py ./photo.png "What is in this image?"
```

FLM has no llama.cpp-style `/health`. Use `/v1/models` plus one chat.

---

### 5. Optional CLI / bench

`flm run` and `flm bench` must not share the NPU with `flm serve`:

```bash
flm run gemma4-it:e4b
# /input "/path/a.jpg" Describe this image

pkill -x flm 2>/dev/null || true
flm bench gemma4-it:e4b --bench-iterations 2 --pmode performance
```

Validated decode on 2026-09-16: **11.75 → 6.61 tok/s** from 1k to 32k. Full table: [results](./fastflowlm-npu-results.md).

---

### 6. Frontends and stop

Base URL `http://127.0.0.1:8219/v1`, any API key, model `gemma4-it:e4b`. Stop with `pkill -x flm`.
