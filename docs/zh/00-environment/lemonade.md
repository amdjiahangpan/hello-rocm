## Lemonade 环境配置（Ubuntu 24.04）

本节介绍在 Ubuntu 24.04 上安装 [Lemonade Server](https://github.com/lemonade-sdk/lemonade)，提供 OpenAI 兼容 API 和 Web UI。GPU 走 llama.cpp，NPU 走 FastFlowLM。本页只装环境和后端；拉模见 [01-Deploy](/zh/01-deploy/)。

> 官方文档：[Ubuntu 安装](https://lemonade-server.ai/docs/guide/install/ubuntu/) · [Linux NPU / FLM](https://lemonade-server.ai/flm_npu_linux.html)
>
> NPU 驱动与 IOMMU 的细账在 [FastFlowLM 环境](./fastflowlm.md)。Lemonade 只是调用已装好的 `flm`。

---

### 本教程验证基线（2026-09-16）

| 项 | 值 |
|:---|:---|
| 包 | `lemonade-server` **11.9.0~24.04** |
| CLI | `lemonade version 11.9.0` |
| 服务 | `/usr/bin/lemond`，默认 **13305** |
| OpenAI | `http://127.0.0.1:13305/api/v1`（部分客户端也可试 `/v1`） |
| GPU 后端 | **`llamacpp:vulkan`**（实测机 kernel 6.17 上 `llamacpp:rocm` 报 *Linux kernel missing support*） |
| NPU 后端 | 见下方注意：本教程 **NPU 推荐走原生 FastFlowLM 8219** |
| 系统服务 | `lemond.service`（enabled），工作目录 `/var/lib/lemonade` |

---

### 1. 安装 Server

```bash
sudo add-apt-repository -y ppa:lemonade-team/stable
sudo apt update
sudo apt install -y lemonade-server
```

```bash
lemonade --version
lemonade status
```

应看到 `lemonade version 11.9.0`（或更新）。Server 在跑时：`Server is running on port 13305`。浏览器打开 `http://127.0.0.1:13305`。

APT 安装通常会启用 **`lemond.service`**（开机自启、用户 `lemonade`、`ProtectHome=yes`）。用完可停：

```bash
sudo systemctl stop lemond
```

不要开机自启：

```bash
sudo systemctl disable --now lemond
```

> ⚠️ 单位名是 **`lemond`**，不是 `lemonade-server`。

---

### 2. 安装推理后端

```bash
lemonade backends
lemonade backends install llamacpp:vulkan   # 已装可跳过
lemonade backends
```

`llamacpp:rocm` 需要内核满足 [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html)。不满足时用 Vulkan，不要卡在 ROCm 后端。

**关于 `flm:npu`：**

- `lemonade backends install flm:npu` 会去 GitHub 拉 FLM 包，国内常超时。
- 系统里若已有 apt / deb 安装的 `flm`，systemd 的 `lemond.service` 以用户 `lemonade` 运行且 **ProtectHome=yes**，**读不到** `~/.config/flm`。
- 因此本教程 NPU 请走 [FastFlowLM 环境](./fastflowlm.md) + `flm serve`（端口 **8219**）。
- 若你改用登录用户前台跑 `lemond`（不要 systemd ProtectHome），并成功装上 `flm:npu`，再尝试 `lemonade load <flm-tag>`。

不要 `source /opt/rocm` 再跑 Lemonade 的 NPU 路径。

---

### 3. 体检

```bash
lemonade --version
lemonade status
lemonade backends
ss -tlnp | grep 13305
```

可选脚本：

```bash
bash src/npu/scripts/check_lemonade.sh
```

NPU 还要另做 FastFlowLM 校验：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
bash src/npu/scripts/check_flm_npu.sh
```

---

### 4. 常用命令

| 命令 | 作用 |
|:---|:---|
| `lemonade backends` | 已装 / 可装后端 |
| `lemonade status` | Server 是否在 13305、已 load 的模型 |
| `lemonade list` | 目录；`--downloaded` 只看本地 |
| `lemonade pull NAME` | 下载（国内可加 `--source modelscope`） |
| `lemonade load NAME` | 加载到 Server |
| `lemonade unload` | 卸模型 |
| `lemonade run NAME` | load + 打开 Web UI |
| `lemonade chat` | 终端 REPL |

GPU 指定后端：

```bash
lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan
```

---

### 5. 与本教程其它端口

| 端口 | 用途 |
|:---|:---|
| **13305** | Lemonade Server |
| **8219** | 原生 `flm serve`（与 Lemonade NPU **互斥**） |
| 11434 / 8000 等 | Ollama / vLLM 等 GPU 教程，统一内存紧张时不要同时常驻 |

下一步：

- [Gemma 4 Lemonade GPU](/zh/01-deploy/gemma4/lemonade-gpu-deploy.md)
- [Gemma 4 Lemonade NPU](/zh/01-deploy/gemma4/lemonade-npu-deploy.md)
- [Qwen3.6 Lemonade](/zh/01-deploy/qwen36/lemonade-gpu-deploy.md)
- [NPU 排错](./npu-troubleshooting.md)
