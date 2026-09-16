## Lemonade GPU deployment (Ubuntu 24.04 + llama.cpp)

Deploy Gemma 4 E4B-it GGUF with **Lemonade Server + llamacpp** on the Ryzen AI iGPU. Not FastFlowLM, not vLLM.

> Prerequisite: [Lemonade environment](/00-environment/lemonade.md). NPU: [FastFlowLM](./fastflowlm-npu-deploy.md) · [Lemonade NPU](./lemonade-npu-deploy.md).

---

| Item | Value |
|:---|:---|
| Lemonade id | `Gemma-4-E4B-it-GGUF` (confirm with `lemonade list`) |
| Weights | `unsloth/gemma-4-E4B-it-GGUF` (often Q4_K_M) |
| Backend | `--llamacpp vulkan` (try `rocm` if the kernel qualifies) |
| API | `http://127.0.0.1:13305/api/v1` |
| Device | gfx1151 iGPU, **does not** occupy `/dev/accel` |

```bash
lemonade backends install llamacpp:vulkan
lemonade pull Gemma-4-E4B-it-GGUF --source modelscope
lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan
lemonade status
```

If `llamacpp:rocm` reports *Linux kernel missing support*, stay on Vulkan. See [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html).

Smoke:

```bash
BASE=http://127.0.0.1:13305/api/v1 MODEL=Gemma-4-E4B-it-GGUF \
  python3 src/npu/examples/chat_gemma4.py
```

Frontend: Base URL `http://127.0.0.1:13305/api/v1`, API key `lemonade`. Stop: `lemonade unload` and, if needed, `sudo systemctl stop lemond`.
