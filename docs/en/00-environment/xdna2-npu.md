## Ryzen AI XDNA2 Hardware

This page covers the hardware needed for **FastFlowLM / Lemonade NPU**. NPU inference uses **XDNA2** and is independent of the [ROCm GPU stack](/00-environment/): GPU tutorials use the iGPU (`gfx1151` on Strix Halo); NPU tutorials use `/dev/accel/accel0`.

> See also: [Lemonade Linux + FLM](https://lemonade-server.ai/flm_npu_linux.html) · [FastFlowLM Linux](https://fastflowlm.com/docs/install_lin/)

---

### 1. Supported processors

FastFlowLM NPU LLMs on Linux require **XDNA2**. XDNA1 cannot load the NPU2 weights used in these tutorials.

| Family | Codename | NPU | This tutorial |
|:---|:---|:---|:---|
| Ryzen AI Max 300 | Strix Halo (e.g. MAX+ 395, gfx1151) | XDNA2 | **Primary**; sample host is GMKtec EVO-X2 |
| Ryzen AI 300 | Kraken Point, Strix Point | XDNA2 | Supported; official Gemma 4 benches often use Kraken |
| Ryzen AI 400 | Gorgon Point | XDNA2 | Supported |
| Z2 Extreme | Handheld | XDNA2 | Supported |
| Ryzen AI 7000 / 8000 / 200 | Phoenix / Hawk Point | **XDNA1** | **Not supported** for NPU LLMs |

The Strix Halo iGPU is **gfx1151**. Lemonade `llamacpp:rocm` / `llamacpp:vulkan` use that iGPU and **do not** occupy `/dev/accel`.

---

### 2. Runtime stack

| Item | Requirement |
|:---|:---|
| NPU firmware | **1.1.0.0 or later** (validated at 1.1.2.65) |
| Kernel + driver | Kernel 7.0+ with in-tree `amdxdna`, or `amdxdna-dkms` on Ubuntu 24.04 |
| Userspace | `libxrt-npu2` + FastFlowLM (`flm`) |
| memlock | `ulimit -l` = `unlimited` |
| Device node | `/dev/accel/accel0`; user in `render` (also add `video`) |
| IOMMU | cmdline has both `amd_iommu=on` and `iommu=pt` |

---

### 3. Check the machine

```bash
lspci -nn | grep -iE 'processing|display|npu'
ls -l /dev/accel/accel0
```

On Strix Halo the NPU PCI ID is commonly `1022:17f0`. After `flm` is installed:

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
```

You should see the NPU device, firmware **1.1.x**, and memlock infinity / unlimited.

---

### 4. Relationship to ROCm GPU

- **They can coexist**: ROCm for GPU (vLLM / Ollama / llama.cpp), FastFlowLM for NPU.
- **Do not mix env vars**: `source /opt/rocm` then `flm` often becomes `No such device with index '0'`. Unset `ROCM_PATH HIP_PATH LD_LIBRARY_PATH` first.
- **Unified memory**: NPU and large GGUF share system RAM. Gemma 4 E4B NPU is ~9 GiB; Qwen3.6-35B-A3B NPU is ~29 GiB. Keep one large model resident.
- **One NPU client at a time**: if Lemonade has loaded an FLM model, do not also `flm serve`.

Next: [FastFlowLM environment](./fastflowlm.md) · [Lemonade environment](./lemonade.md)
