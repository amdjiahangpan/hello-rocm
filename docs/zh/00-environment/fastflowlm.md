## FastFlowLM 环境配置（Ubuntu 24.04 + XDNA2）

本节介绍在 Ubuntu 24.04 上安装 **FastFlowLM（`flm`）** 与 XDNA2 NPU 驱动，使后续章节可以直接 `flm pull` / `flm serve`。不经过 Lemonade。硬件对照见 [XDNA2](./xdna2-npu.md)；排错见 [NPU 排错](./npu-troubleshooting.md)。

> 前置条件：本机是 **XDNA2**（Strix Halo / Kraken / Strix Point / Gorgon）。XDNA1 请停在硬件对照页。
>
> 官方文档：[FastFlowLM Linux](https://fastflowlm.com/docs/install_lin/) · [Lemonade Linux + FLM](https://lemonade-server.ai/flm_npu_linux.html)（驱动来源相同，本节只用 `flm`）

---

### 本教程验证基线（2026-09-16）

| 项 | 值 |
|:---|:---|
| `flm` | **1.0.5**（`/usr/bin/flm`） |
| XRT | `libxrt-npu2` 2.25.0 |
| 驱动 | `amdxdna-dkms` |
| 设备 | `/dev/accel/accel0`，组 `render` |
| 固件 | **1.1.2.65** |
| IOMMU | `amd_iommu=on iommu=pt` |
| memlock | unlimited |
| 用户组 | `render` `video` |
| 已验证权重 | `gemma4-it:e4b` · `qwen3.6-moe:35b-a3b` |

安装时请到 [FastFlowLM Releases](https://github.com/ROCm/FastFlowLM/releases) 取**当前最新** `.deb`，不必钉死 1.0.5。

---

### 1. 安装驱动与 FLM

```bash
sudo add-apt-repository -y ppa:lemonade-team/stable
sudo apt update
sudo apt install -y libxrt-npu2 amdxdna-dkms
```

从 [Releases](https://github.com/ROCm/FastFlowLM/releases) 下载对应发行版的 `.deb`（Ubuntu 24.04 选 `ubuntu24.04`）：

```bash
sudo apt install ./fastflowlm*_ubuntu24.04_amd64.deb
sudo reboot
```

**必须 reboot**，才会切到 DKMS 模块和 1.1 固件。不要只换 firmware、不换 DKMS。

---

### 2. 配置 IOMMU

```bash
tr ' ' '\n' < /proc/cmdline | grep iommu
```

必须同时有 `amd_iommu=on` 和 `iommu=pt`。否则 `flm` 会出现 SVA bind **-19**。缺了就改 GRUB：

```bash
sudo nano /etc/default/grub
```

在 `GRUB_CMDLINE_LINUX_DEFAULT` 中加入 `amd_iommu=on iommu=pt`，然后：

```bash
sudo update-grub
sudo reboot
```

> ⚠️ 部分「拉大 GTT」配方会写 `amd_iommu=off`，和 NPU **互斥**。要跑 FastFlowLM 就必须用 `iommu=pt`。

---

### 3. 权限与 memlock

```bash
sudo usermod -aG render,video "$USER"   # 一次；重新登录后生效
```

`ulimit -l` 必须是 `unlimited`。若不是，写入 `/etc/security/limits.conf`：

```text
*    soft    memlock    unlimited
*    hard    memlock    unlimited
```

重新登录后再查：

```bash
ls -l /dev/accel/accel0
ulimit -l
```

---

### 4. 校验

每次跑 `flm` 前先清掉 ROCm 用户态路径：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
flm list --filter installed
```

`flm validate` 期望类似：

```text
[Linux]  NPU: /dev/accel/accel0
[Linux]  NPU FW Version: 1.1.2.65
[Linux]  Memlock Limit: infinity
```

仓库脚本（可选）：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
bash src/npu/scripts/check_flm_npu.sh
```

`source /opt/rocm` 后再跑 `flm` 会变成 `No such device with index '0'`。

---

### 5. 端口与互斥

| 项 | 值 |
|:---|:---|
| `flm serve` | 本教程用 **8219**（官方默认端口可能不同） |
| 同时 LLM | **1 路**（NPU 串行） |
| 与 Lemonade | Lemonade 若 `load` 了 FLM 模型，先 `lemonade unload` 再 `flm serve` |

下一步：

- [Gemma 4 FastFlowLM 部署](/zh/01-deploy/gemma4/fastflowlm-npu-deploy.md)
- [Qwen3.6 FastFlowLM 部署](/zh/01-deploy/qwen36/fastflowlm-npu-deploy.md)
- [Lemonade 环境](./lemonade.md)（要 Web UI / 统一 13305 端口时）
