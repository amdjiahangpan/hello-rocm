## Lemonade GPU 部署 Qwen3.6-35B-A3B

用 Lemonade 的 **llamacpp** 在 iGPU 上跑 Qwen3.6 GGUF。

> 前置条件：[Lemonade 环境](/zh/00-environment/lemonade.md)。模型说明：[Qwen3.6](./qwen36_model.md)。NPU：[FastFlowLM](./fastflowlm-npu-deploy.md) · [Lemonade NPU](./lemonade-npu-deploy.md)。

---

| 项 | 值 |
|:---|:---|
| id | `Qwen3.6-35B-A3B-GGUF`（以 `lemonade list` 为准） |
| 后端 | `--llamacpp vulkan`（kernel 满足时再试 `rocm`） |
| API | `http://127.0.0.1:13305/api/v1` |
| 内存 | 建议 32 GB+ 统一内存；先停其它大模型 |

```bash
lemonade backends install llamacpp:vulkan
lemonade pull Qwen3.6-35B-A3B-GGUF --source modelscope
lemonade load Qwen3.6-35B-A3B-GGUF --llamacpp vulkan
lemonade status
```

冒烟：

```bash
BASE=http://127.0.0.1:13305/api/v1 MODEL=Qwen3.6-35B-A3B-GGUF \
  python3 src/npu/examples/chat_gemma4.py
```

Open WebUI：Base `http://127.0.0.1:13305/api/v1`，模型填 `Qwen3.6-35B-A3B-GGUF`。

`llamacpp:rocm` 不可用时保持 `vulkan`。停止：`lemonade unload`。不要和 FastFlowLM 8219 上的 29 GiB NPU 模型同时常驻。
