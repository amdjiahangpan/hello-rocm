## FastFlowLM environment (Ubuntu 24.04 + XDNA2)

This page installs **FastFlowLM (`flm`)** and the XDNA2 NPU driver on Ubuntu 24.04 so later chapters can `flm pull` / `flm serve` without Lemonade. Hardware: [XDNA2](./xdna2-npu.md). Troubleshooting: [NPU troubleshooting](./npu-troubleshooting.md).

> Prerequisite: the machine is **XDNA2** (Strix Halo / Kraken / Strix Point / Gorgon).
>
> Official docs: [FastFlowLM Linux](https://fastflowlm.com/docs/install_lin/) · [Lemonade Linux + FLM](https://lemonade-server.ai/flm_npu_linux.html) (same driver source; this page uses `flm` only)

---

### Validated baseline (2026-09-16)

| Item | Value |
|:---|:---|
| `flm` | **1.0.5** (`/usr/bin/flm`) |
| XRT | `libxrt-npu2` 2.25.0 |
| Driver | `amdxdna-dkms` |
| Device | `/dev/accel/accel0`, group `render` |
| Firmware | **1.1.2.65** |
| IOMMU | `amd_iommu=on iommu=pt` |
| memlock | unlimited |
| Groups | `render` `video` |
| Validated weights | `gemma4-it:e4b` · `qwen3.6-moe:35b-a3b` |

Install the **latest** `.deb` from [FastFlowLM Releases](https://github.com/ROCm/FastFlowLM/releases); you do not have to pin 1.0.5.

---

### 1. Install driver and FLM

```bash
sudo add-apt-repository -y ppa:lemonade-team/stable
sudo apt update
sudo apt install -y libxrt-npu2 amdxdna-dkms
```

Download the matching `.deb` (Ubuntu 24.04 → `ubuntu24.04`):

```bash
sudo apt install ./fastflowlm*_ubuntu24.04_amd64.deb
sudo reboot
```

**Reboot is required** so DKMS and 1.1 firmware take effect. Do not swap firmware without the matching DKMS driver.

---

### 2. IOMMU

```bash
tr ' ' '\n' < /proc/cmdline | grep iommu
```

You need both `amd_iommu=on` and `iommu=pt`. Otherwise `flm` hits SVA bind **-19**. If missing, add them to `GRUB_CMDLINE_LINUX_DEFAULT`, then:

```bash
sudo update-grub
sudo reboot
```

> ⚠️ Some “large GTT” recipes set `amd_iommu=off`, which is **incompatible** with NPU.

---

### 3. Permissions and memlock

```bash
sudo usermod -aG render,video "$USER"   # once; re-login
```

`ulimit -l` must be `unlimited`. If not, add to `/etc/security/limits.conf`:

```text
*    soft    memlock    unlimited
*    hard    memlock    unlimited
```

Re-login, then:

```bash
ls -l /dev/accel/accel0
ulimit -l
```

---

### 4. Validate

Clear ROCm userspace paths before every `flm` invocation:

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
flm list --filter installed
```

Expected `flm validate` output includes `/dev/accel/accel0`, firmware **1.1.x**, and unlimited memlock.

Optional:

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
bash src/npu/scripts/check_flm_npu.sh
```

`source /opt/rocm` then `flm` typically becomes `No such device with index '0'`.

---

### 5. Ports and exclusivity

| Item | Value |
|:---|:---|
| `flm serve` | this tutorial uses **8219** |
| Concurrent LLMs | **1** on NPU |
| vs Lemonade | `lemonade unload` before `flm serve` if Lemonade loaded an FLM model |

Next: [Gemma 4 FastFlowLM](/01-deploy/gemma4/fastflowlm-npu-deploy.md) · [Qwen3.6 FastFlowLM](/01-deploy/qwen36/fastflowlm-npu-deploy.md) · [Lemonade environment](./lemonade.md)
