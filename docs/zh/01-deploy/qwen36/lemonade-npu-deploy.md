## Lemonade NPU 部署 Qwen3.6-35B-A3B

与 Gemma 4 相同：系统 `lemond` 当前跑不了用户目录里的 FLM 权重。用 [FastFlowLM Qwen3.6](./fastflowlm-npu-deploy.md)。

> 环境：[Lemonade](/zh/00-environment/lemonade.md) · [FastFlowLM](/zh/00-environment/fastflowlm.md)。GPU：[Lemonade GPU](./lemonade-gpu-deploy.md)。

```bash
lemonade unload 2>/dev/null || true
pkill -x flm 2>/dev/null || true
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve qwen3.6-moe:35b-a3b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

冒烟：

```bash
BASE=http://127.0.0.1:8219 MODEL=qwen3.6-moe:35b-a3b \
  python3 src/npu/examples/chat_gemma4.py
```

常驻约 29 GiB，先停 GPU 大模型。若你改用登录用户前台跑 `lemond` 且 `flm:npu` 已装，可再试 `lemonade load qwen3.6-moe:35b-a3b`。
