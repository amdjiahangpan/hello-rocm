## FastFlowLM 零基础 NPU 部署（Ubuntu 24.04 + XDNA2）

本节介绍在 Ubuntu 24.04 + **XDNA2 NPU** 上，使用 **FastFlowLM（`flm`）** 部署 Gemma 4 E4B-it，并暴露 OpenAI 兼容接口。这不是 vLLM / Ollama，也不走 ROCm GPU。

> 前置条件：已完成 [FastFlowLM 环境](/zh/00-environment/fastflowlm.md)（驱动、IOMMU、`flm validate`）。
>
> 模型背景见 [Gemma 4 模型介绍](./gemma4_model.md)。GPU 对照：[LM Studio](./lm-studio-rocm7-deploy.md) · [Lemonade GPU](./lemonade-gpu-deploy.md)。NPU 包一层 Lemonade：[Lemonade NPU](./lemonade-npu-deploy.md)。实测吞吐：[运行结果](./fastflowlm-npu-results.md)。

- **适合人群**：本机已有 XDNA2，想先跑通一个端侧多模态模型
- **难度等级**：⭐⭐
- **预计时间**：权重约 8.7 GiB（下载视网速）；加载后冒烟约 1 分钟
- **验证机**：GMKtec EVO-X2 / Ryzen AI MAX+ 395，FLM 1.0.5

---

### 你将得到

| 项 | 值 |
|:---|:---|
| 模型 tag | `gemma4-it:e4b` |
| 量化 | NPU2 / Q4_1（文本 + 视觉 + 音频），**不是** GGUF |
| 监听 | `http://0.0.0.0:8219/v1` |
| 默认上下文 | 32768（最大 128k） |
| 常驻 | 约 9 GiB |

不要把 GPU 那套 `Q4_K_M.gguf` 塞给 `flm`。

---

### 1. 前置（30 秒）

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
```

应看到 NPU 固件 **1.1.x**。失败先看 [NPU 排错](/zh/00-environment/npu-troubleshooting.md)。

不要 `source /opt/rocm` 再跑 `flm`。NPU 走 Lemonade XRT，和 GPU 的 ROCm 混用会变成 `No such device with index '0'`。

可选：`bash src/npu/scripts/check_flm_npu.sh`

---

### 2. 拉权重

国内默认 ModelScope：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm pull gemma4-it:e4b --modelscope 1
flm check gemma4-it:e4b
flm list --filter installed
```

约 8.7 GiB，落到 `~/.config/flm/models/`（可用 `FLM_MODEL_PATH` 改位置）。Hugging Face 源：去掉 `--modelscope`，仓库为 [FastFlowLM/Gemma4-E4B-IT-NPU2](https://huggingface.co/FastFlowLM/Gemma4-E4B-IT-NPU2)。

---

### 3. 起服务

NPU 同时只能一个 LLM。若 8219 上还是别的模型，或 Lemonade 占着 NPU，先停：

```bash
lemonade unload 2>/dev/null || true
sudo systemctl stop lemond 2>/dev/null || true
pkill -x flm 2>/dev/null || true
```

手动启动：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve gemma4-it:e4b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

常用参数：

| 参数 | 含义 |
|:---|:---|
| `--ctx-len` | 上下文；非 2 的幂向上取整，最小 512 |
| `--pmode` | `powersaver` / `balanced` / `performance` / `turbo` |
| `--q-len` | NPU 请求队列，满了 503 |
| `--prefill-chunk-len` | prefill 分块，默认 4096 |

覆盖示例：`--ctx-len 8192 --pmode turbo`。

---

### 4. 冒烟

另开终端：

```bash
curl -s http://127.0.0.1:8219/v1/models

curl -s http://127.0.0.1:8219/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-it:e4b",
    "messages": [{"role":"user","content":"只回答一个词：ok"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

FLM **没有** llama.cpp 那种 `/health`。用 `/v1/models` 和一次 chat 判断是否就绪。

Python 示例（标准库，不必装 openai）：

```bash
python3 src/npu/examples/chat_gemma4.py
```

图文：

```bash
python3 src/npu/examples/chat_gemma4_image.py ./photo.png "图里有什么？"
```

使用官方 SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8219/v1", api_key="flm")
print(client.chat.completions.create(
    model="gemma4-it:e4b",
    messages=[{"role": "user", "content": "用一句话介绍你自己。"}],
).choices[0].message.content)
```

`api_key` 任意非空即可。

---

### 5. CLI 交互（可选）

不要和 `flm serve` 同时抢 NPU。适合确认权重后再开 server：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm run gemma4-it:e4b
```

进 CLI 后：`/?` 帮助，`/show` 模型信息，`/bye` 退出。图和音频：

```text
/input "/path/to/image.jpg" 用中文描述这张图
/input "/path/to/audio.mp3" 总结这段音频
```

只给路径加引号，提示词不要加引号。

---

### 6. 1k–32k 阶梯（可选）

`flm serve` 和 `flm bench` **不能同时占 NPU**。测阶梯前先停服务：

```bash
pkill -x flm 2>/dev/null || true
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm bench gemma4-it:e4b --bench-iterations 2 --pmode performance
```

会在当前目录写 CSV。本机 2026-09-16、2 次迭代、performance：

| 上下文 | Decode (tok/s) | Prefill (tok/s) | TTFT (s) |
|:---|---:|---:|---:|
| 1k | 11.75 | 449.5 | 2.18 |
| 2k | 11.38 | 559.7 | 3.48 |
| 4k | 10.94 | 629.1 | 6.18 |
| 8k | 9.95 | 664.0 | 11.69 |
| 16k | 8.54 | 638.3 | 24.29 |
| 32k | 6.61 | 546.1 | 56.76 |

短 chat 的 decode（约 12.5 tok/s）是空 KV；要比长上下文请用这条阶梯。完整对照见 [运行结果](./fastflowlm-npu-results.md)。

---

### 7. 给前端用

Open WebUI / 其它 OpenAI 兼容客户端：

- Base URL：`http://127.0.0.1:8219/v1`（容器内 `http://host.docker.internal:8219/v1`）
- API Key：任意
- 模型：`gemma4-it:e4b`

视觉 token 预算可用自定义参数 `image-max-tokens`（常见值 70 / 140 / 280 / 560 / 1120）。关 Ollama 连接，避免抢模型列表。

NPU 串行执行，多客户端是排队不是多槽并行。打满 `--q-len` 会 503。

---

### 8. 停止

```bash
pkill -x flm
```

只杀 `flm`，不动 GPU 上的 vLLM / Ollama / llama.cpp。

---

### 下一步

- [运行结果与性能分析](./fastflowlm-npu-results.md)
- [Lemonade 包一层 NPU](./lemonade-npu-deploy.md)
- [Lemonade GPU](./lemonade-gpu-deploy.md)
- [NPU 排错](/zh/00-environment/npu-troubleshooting.md)
