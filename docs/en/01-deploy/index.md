<div align=center>
  <h1>01-Deploy</h1>
  <strong>🚀 ROCm LLM Deployment in Practice</strong>
</div>

<div align="center">

*Get started with LLM deployment on AMD GPUs from scratch*

[Back to Home](/en/) · [中文](/zh/01-deploy/)

</div>

## Introduction

&emsp;&emsp;This module provides comprehensive tutorials for deploying large language models on AMD GPUs. Whether you are a beginner or an experienced developer, you can quickly learn how to deploy and run LLMs on the ROCm platform through these tutorials.

&emsp;&emsp;Since ROCm 7.10.0, ROCm supports seamless installation in Python virtual environments just like CUDA, significantly lowering the barrier for LLM deployment on AMD GPUs.

&emsp;&emsp;This module uses **Google Gemma 4** (`gemma-4-E4B-it` primarily) as the example model by default, and also provides parallel tutorials for **Qwen3 / Qwen3.5 / Qwen3.6**. The Ryzen AI **XDNA2 NPU** path uses FastFlowLM and Lemonade; environment pages live in [00-Environment](/00-environment/). The directory structure is as follows:

```
01-Deploy/
└── models/
    ├── Gemma4/           # Deployment tutorials with Gemma 4 as the primary model (recommended)
    ├── Qwen3/            # Qwen3 series deployment tutorials (reference/comparison)
    ├── Qwen3.5/          # Qwen3.5 series deployment tutorials (new architecture reference)
    └── Qwen3.6/          # Qwen3.6 FastFlowLM / Lemonade
```

## Tutorial List

### Ubuntu 24.04 + ROCm 10 Environment Setup Tutorial

&emsp;&emsp;This tutorial walks you through installing and verifying **ROCm 10.0.0** on Ubuntu 24.04 / Windows 11, including cleaning an old stack, installing PyTorch 2.13.0 with `uv pip`, validating the official vLLM 0.27.0 image, and the optional `rocm install sdk` path. Complete this before any deploy tutorial, or start from the [00-Environment baseline](/00-environment/) and the [ROCm 10.0.0 release notes](/00-environment/rocm-10-0-0-release-notes).

- **Target Audience**: Users setting up a ROCm environment on an AMD GPU for the first time
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the Environment Setup Tutorial (Gemma4)](/en/01-deploy/gemma4/env-prepare-ubuntu24-rocm7.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/env-prepare-ubuntu24-rocm7.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/env-prepare-ubuntu24-rocm7.md)

---

### Gemma 4 Model Introduction

&emsp;&emsp;Before starting deployment, it is recommended to read the Gemma 4 model introduction to understand the architecture characteristics, capability differences, and hardware selection recommendations for the four versions: Gemma 4 E2B / E4B / 31B / 26B A4B, so you can choose the right model for your environment.

- **Target Audience**: Users trying Gemma 4 for the first time
- **Difficulty Level**: ⭐
- **Estimated Time**: 15 minutes

📖 [Read the Gemma 4 Model Introduction](/en/01-deploy/gemma4/gemma4_model.md)

---

### LM Studio LLM Deployment from Scratch

&emsp;&emsp;LM Studio is a user-friendly desktop application that supports running large language models locally. This tutorial uses **Gemma 4 E4B-it Q4_K_M** as an example to guide you through deploying and running LLMs on AMD GPUs using LM Studio with the ROCm version of the llama.cpp backend.

- **Target Audience**: Beginners and users who want to quickly experience LLMs
- **Difficulty Level**: ⭐
- **Estimated Time**: 30 minutes

📖 [Start the LM Studio Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/lm-studio-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md)

---

### vLLM LLM Deployment from Scratch

&emsp;&emsp;vLLM is a high-performance LLM inference and serving framework that supports efficient PagedAttention and continuous batching. This tutorial uses **Gemma 4 E4B-it** as an example, covering both a quick start method using the official ROCm vLLM Docker image, and an advanced method for manually compiling Triton / FlashAttention / vLLM from source.

- **Target Audience**: Developers who need to set up inference services
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the vLLM Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/vllm-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/vllm-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/vllm-rocm7-deploy.md)

---

### Ollama LLM Deployment from Scratch

&emsp;&emsp;Ollama is a framework for quickly serving large language models and vision-language models with an efficient backend runtime. This tutorial uses **Gemma 4 E4B-it Q4_K_M** as an example to guide you through deploying LLMs on AMD GPUs using Ollama (ROCm version llama.cpp backend), with tokens/s benchmark examples included.

- **Target Audience**: Developers who want to spin up a local inference service with a single command
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour

📖 [Start the Ollama Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/ollama-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/ollama-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/ollama-rocm7-deploy.md)

---

### llama.cpp LLM Deployment from Scratch

&emsp;&emsp;llama.cpp is a lightweight and high-performance inference backend that supports multiple model formats including GGUF, with optimized versions available for ROCm. This tutorial uses **Gemma 4 E4B-it Q4_K_M (GGUF)** as an example, showing how to deploy mainstream models on Ubuntu 24.04 + ROCm 7+ using both pre-built binaries and Docker.

- **Target Audience**: Developers who want to freely orchestrate inference workflows via CLI / REST API
- **Difficulty Level**: ⭐⭐⭐
- **Estimated Time**: 1.5 hours

📖 [Start the llama.cpp Deployment Tutorial (Gemma4)](/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md)  
📎 Reference: [Qwen3 Version](/en/01-deploy/qwen3/llamacpp-rocm7-deploy.md) · [Qwen3.5 Version](/en/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md)

---

### FastFlowLM NPU Deployment from Scratch

&emsp;&emsp;FastFlowLM (`flm`) is a native runtime for AMD **XDNA2 NPUs** and uses NPU2 weights rather than GGUF. This tutorial serves **Gemma 4 E4B-it** as an OpenAI-compatible API on Ubuntu 24.04 (default **8219**) and includes 1k–32k measurements. Qwen3.6-35B-A3B is the larger MoE counterpart.

- **Target Audience**: Ryzen AI MAX / AI 300 users with XDNA2 who want on-device NPU inference
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: download depends on the network; Gemma 4 smoke test ~1 minute after load

📖 [Start the FastFlowLM Tutorial (Gemma4)](/en/01-deploy/gemma4/fastflowlm-npu-deploy.md)  
📎 Reference: [Qwen3.6 Version](/en/01-deploy/qwen36/fastflowlm-npu-deploy.md) · [NPU results](/en/01-deploy/gemma4/fastflowlm-npu-results.md) · [Environment](/00-environment/fastflowlm.md)

---

### Lemonade LLM Deployment from Scratch

&emsp;&emsp;Lemonade Server provides a Web UI and OpenAI-compatible API (default **13305**). GPU uses llama.cpp (this tutorial validated `llamacpp:vulkan` on kernel 6.17); NPU is still FastFlowLM underneath. Because systemd `lemond` sets ProtectHome, **the stable NPU path is native FastFlowLM on 8219**.

- **Target Audience**: users who want one GPU port plus a Web UI
- **Difficulty Level**: ⭐⭐
- **Estimated Time**: 1 hour including environment setup

📖 [Start the Lemonade GPU Tutorial (Gemma4)](/en/01-deploy/gemma4/lemonade-gpu-deploy.md)  
📎 Reference: [Lemonade NPU (Gemma4)](/en/01-deploy/gemma4/lemonade-npu-deploy.md) · [Qwen3.6 GPU](/en/01-deploy/qwen36/lemonade-gpu-deploy.md) · [Environment](/00-environment/lemonade.md)

---

## Requirements

### Hardware Requirements

- AMD GPU (ROCm-supported GPUs such as RX 7000 / 9000 series, Ryzen AI MAX / AI 300, Instinct MI series, etc.)
- At least 8GB VRAM recommended (Gemma 4 E4B Q4_K_M quantized version can run with 8GB VRAM; for native bfloat16 inference or larger models, please refer to the VRAM recommendations in the corresponding tutorials)
- FastFlowLM / Lemonade NPU additionally requires **XDNA2** (see [hardware](/00-environment/xdna2-npu.md))

### Software Requirements

- Operating System: Linux (Ubuntu 22.04+) or Windows 11
- ROCm 10.0.0 or later (this module's environment guides target 10.0.0)
- Python 3.10+

## FAQ

<details>
<summary>Q: How do I check if my AMD GPU supports ROCm?</summary>

Please refer to the [ROCm official support list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) to check supported GPU models.

</details>

<details>
<summary>Q: What should I do if I encounter a "HIP error" during deployment?</summary>

1. Confirm that ROCm is properly installed
2. Check that environment variables are correctly set
3. Try restarting the system and running again

</details>

<details>
<summary>Q: I get a permission denied error when downloading Gemma 4?</summary>

Gemma series models require you to first click **Agree & Access** on the corresponding model page on Hugging Face (e.g., <a href="https://huggingface.co/google/gemma-4-E4B-it">google/gemma-4-E4B-it</a>), then log in with a Hugging Face Token that has `read` permissions or inject it via `HF_TOKEN` into the container / process.

</details>

## Reference Resources

- [ROCm 10.0.0 Official Documentation](https://rocm.docs.amd.com/en/latest/)
- [vLLM Official Documentation](https://docs.vllm.ai/)
- [Ollama Official Documentation](https://docs.ollama.com/)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [FastFlowLM](https://fastflowlm.com/docs/)
- [Lemonade Server](https://lemonade-server.ai/)
- [Hugging Face Gemma 4 Model Collection](https://huggingface.co/collections/google/gemma-4)

---

<div align="center">

**Contributions for more deployment tutorials are welcome!** 🎉

[Submit an Issue](https://github.com/datawhalechina/hello-rocm/issues) | [Submit a PR](https://github.com/datawhalechina/hello-rocm/pulls)

</div>
