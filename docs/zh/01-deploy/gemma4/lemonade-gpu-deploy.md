## Lemonade GPU 零基础部署（Ubuntu 24.04 + llama.cpp）

本节介绍用 **Lemonade Server + llamacpp** 在 Ryzen AI iGPU 上部署 Gemma 4 E4B-it GGUF。这不是 FastFlowLM，也不是 vLLM。

> 前置条件：已完成 [Lemonade 环境](/zh/00-environment/lemonade.md)。NPU 对照：[FastFlowLM](./fastflowlm-npu-deploy.md) · [Lemonade NPU](./lemonade-npu-deploy.md)。hello-rocm 上的 vLLM / Ollama 仍见同目录其它教程。

- **适合人群**：希望一个端口（13305）+ Web UI 跑 GGUF
- **难度等级**：⭐⭐
- **预计时间**：权重约 4–6 GiB；加载后冒烟约 1 分钟

---

### 你将得到

| 项 | 值 |
|:---|:---|
| Lemonade id | `Gemma-4-E4B-it-GGUF`（以 `lemonade list` 为准） |
| 权重 | Hugging Face `unsloth/gemma-4-E4B-it-GGUF`（常见 Q4_K_M） |
| 后端 | `--llamacpp vulkan`（kernel 满足时再试 `rocm`） |
| API | `http://127.0.0.1:13305/api/v1` |
| 设备 | gfx1151 iGPU，**不占** `/dev/accel` |

---

### 1. 环境

```bash
lemonade --version
lemonade status
lemonade backends
```

没有 `llamacpp:vulkan` 时：

```bash
lemonade backends install llamacpp:vulkan
```

`llamacpp:rocm` 若报 *Linux kernel missing support*，继续用 Vulkan。见 [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html)。

---

### 2. 拉模并加载

国内推荐 ModelScope：

```bash
lemonade pull Gemma-4-E4B-it-GGUF --source modelscope
lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan
lemonade status
```

`run` 会确保 Server 在 13305，并尝试打开浏览器：

```bash
lemonade run Gemma-4-E4B-it-GGUF --llamacpp vulkan
```

只要 API、不要交互时用 `load`。id 以 `lemonade list` 为准。

---

### 3. 冒烟

```bash
curl -s http://127.0.0.1:13305/api/v1/models

BASE=http://127.0.0.1:13305/api/v1 MODEL=Gemma-4-E4B-it-GGUF \
  python3 src/npu/examples/chat_gemma4.py
```

若客户端只认 `/v1`，把 Base 改成 `http://127.0.0.1:13305/v1` 再试。

E4B GGUF 一般带视觉；以当前 llama.cpp 是否编译进 mmproj 为准。NPU 多模态更稳的是 [FLM 路径](./fastflowlm-npu-deploy.md)。

前端：Base URL `http://127.0.0.1:13305/api/v1`，API Key `lemonade`，模型 `Gemma-4-E4B-it-GGUF`。

---

### 4. 停止

```bash
lemonade unload
sudo systemctl stop lemond   # 若由 systemd 拉起
```

不要和其它大 GGUF、vLLM 同时常驻统一内存。
