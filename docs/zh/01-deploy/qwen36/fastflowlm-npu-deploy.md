## FastFlowLM 零基础 NPU 部署：Qwen3.6-35B-A3B

本节介绍在 Ubuntu 24.04 + **XDNA2 NPU** 上，使用 FastFlowLM 部署 **Qwen3.6-35B-A3B**。

> 前置条件：[FastFlowLM 环境](/zh/00-environment/fastflowlm.md)。模型差异见 [Qwen3.6 说明](./qwen36_model.md)。GPU / Lemonade：[Lemonade GPU](./lemonade-gpu-deploy.md)。Gemma 4 对照：[Gemma 4 FastFlowLM](/zh/01-deploy/gemma4/fastflowlm-npu-deploy.md)。

- **适合人群**：统一内存充裕（建议 64 GB+）、已跑通过 Gemma 4 NPU
- **难度等级**：⭐⭐⭐
- **预计时间**：权重下载视网速；常驻约 29 GiB

---

### 你将得到

| 项 | 值 |
|:---|:---|
| tag | `qwen3.6-moe:35b-a3b` |
| 格式 | NPU2（不是 GGUF） |
| API | `http://127.0.0.1:8219/v1` |
| 常驻 | 约 **29 GiB** |
| 吞吐参考 | 验证机单路长生成约 **14 tok/s** |

---

### 1. 拉权重

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm pull qwen3.6-moe:35b-a3b --modelscope 1
flm check qwen3.6-moe:35b-a3b
flm list --filter installed
```

已装可跳过 `pull`。

---

### 2. 起服务

先停 Lemonade 和已有 `flm`（NPU 同时一路）。建议同时停 GPU 上的大模型，避免统一内存 OOM。

```bash
lemonade unload 2>/dev/null || true
sudo systemctl stop lemond 2>/dev/null || true
pkill -x flm 2>/dev/null || true

unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve qwen3.6-moe:35b-a3b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

---

### 3. 冒烟

```bash
curl -s http://127.0.0.1:8219/v1/models

BASE=http://127.0.0.1:8219 MODEL=qwen3.6-moe:35b-a3b \
  python3 src/npu/examples/chat_gemma4.py
```

前端模型名填 `qwen3.6-moe:35b-a3b`。并发靠 `--q-len` 排队，满了 503。

---

### 4. 停止

```bash
pkill -x flm
```
