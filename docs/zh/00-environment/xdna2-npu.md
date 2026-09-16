## Ryzen AI XDNA2 硬件对照

本节说明 **FastFlowLM / Lemonade NPU** 路径需要的硬件。NPU 推理走 **XDNA2**，与 [ROCm GPU 栈](/zh/00-environment/) 相互独立：GPU 教程用 `gfx1151` 等 iGPU；NPU 教程用 `/dev/accel/accel0`。

> 对照：[Lemonade Linux + FLM](https://lemonade-server.ai/flm_npu_linux.html) · [FastFlowLM Linux 安装](https://fastflowlm.com/docs/install_lin/)

---

### 1. 支持的处理器

FastFlowLM 在 Linux 上跑 NPU LLM **只要 XDNA2**。XDNA1 不能加载本教程的 NPU2 权重。

| 系列 | 代号 | NPU | 本教程 |
|:---|:---|:---|:---|
| Ryzen AI Max 300 | Strix Halo（如 MAX+ 395，gfx1151） | XDNA2 | **主推**，实测示例机为 GMKtec EVO-X2 |
| Ryzen AI 300 | Kraken Point、Strix Point | XDNA2 | 支持；官方 Gemma 4 基准多用 Kraken |
| Ryzen AI 400 | Gorgon Point | XDNA2 | 支持 |
| Z2 Extreme | 掌机 | XDNA2 | 支持 |
| Ryzen AI 7000 / 8000 / 200 | Phoenix / Hawk Point 等 | **XDNA1** | **不支持** NPU LLM |

Strix Halo 的 iGPU 是 **gfx1151**。Lemonade 的 `llamacpp:rocm` / `llamacpp:vulkan` 走这块 iGPU，**不占用** `/dev/accel`。

---

### 2. 运行时栈

| 项 | 要求 |
|:---|:---|
| NPU 固件 | **1.1.0.0 或更高**（本教程实测 1.1.2.65） |
| 内核 + 驱动 | 内核 7.0+ 自带 `amdxdna`，或 Ubuntu 24.04 用 `amdxdna-dkms` |
| 用户态 | `libxrt-npu2` + FastFlowLM（`flm`） |
| memlock | `ulimit -l` 为 `unlimited` |
| 设备节点 | `/dev/accel/accel0`，用户在 `render`（建议同时加入 `video`）组 |
| IOMMU | cmdline 同时有 `amd_iommu=on` 与 `iommu=pt` |

---

### 3. 查本机

```bash
lspci -nn | grep -iE 'processing|display|npu'
ls -l /dev/accel/accel0
```

Strix Halo 上 NPU PCI 常见为 `1022:17f0`。装好 `flm` 后再跑：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
```

期望看到 NPU 设备、固件 **1.1.x**、memlock 为 infinity / unlimited。

---

### 4. 与 ROCm GPU 的关系

- **可以共存**：同一台 Ryzen AI MAX 上，ROCm 跑 GPU（vLLM / Ollama / llama.cpp），FastFlowLM 跑 NPU。
- **不要混环境变量**：`source /opt/rocm` 后再跑 `flm`，常见报错是 `No such device with index '0'`。每次跑 NPU 前先 `unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH`。
- **统一内存互斥**：NPU 与大 GGUF 抢同一块系统内存。Gemma 4 E4B NPU 约 9 GiB；Qwen3.6-35B-A3B NPU 约 29 GiB。同时只常驻一路大模型。
- **NPU 同时只能一路**：Lemonade 若已经 `load` 了 FLM 模型，就不要再 `flm serve`。

下一步：[FastFlowLM 环境](./fastflowlm.md) · [Lemonade 环境](./lemonade.md)
