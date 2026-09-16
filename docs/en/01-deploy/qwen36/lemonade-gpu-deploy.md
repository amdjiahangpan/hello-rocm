## Lemonade GPU deployment: Qwen3.6-35B-A3B

Run Qwen3.6 GGUF on the iGPU through Lemonade llamacpp.

> Prerequisite: [Lemonade environment](/00-environment/lemonade.md). Model: [Qwen3.6](./qwen36_model.md). NPU: [FastFlowLM](./fastflowlm-npu-deploy.md).

```bash
lemonade backends install llamacpp:vulkan
lemonade pull Qwen3.6-35B-A3B-GGUF --source modelscope
lemonade load Qwen3.6-35B-A3B-GGUF --llamacpp vulkan
lemonade status
```

Id must contain **3.6**. Use Vulkan if `llamacpp:rocm` is unsupported. Smoke with `BASE=http://127.0.0.1:13305/api/v1 MODEL=Qwen3.6-35B-A3B-GGUF python3 src/npu/examples/chat_gemma4.py`. Do not keep this resident together with the ~29 GiB NPU model on 8219.
