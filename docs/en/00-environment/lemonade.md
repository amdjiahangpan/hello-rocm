## Lemonade environment (Ubuntu 24.04)

This page installs [Lemonade Server](https://github.com/lemonade-sdk/lemonade) on Ubuntu 24.04: OpenAI-compatible API plus Web UI. GPU uses llama.cpp; NPU uses FastFlowLM. Model pull lives in [01-Deploy](/01-deploy/).

> Official docs: [Ubuntu install](https://lemonade-server.ai/docs/guide/install/ubuntu/) · [Linux NPU / FLM](https://lemonade-server.ai/flm_npu_linux.html)
>
> Driver and IOMMU details: [FastFlowLM environment](./fastflowlm.md). Lemonade only calls an already-installed `flm`.

---

### Validated baseline (2026-09-16)

| Item | Value |
|:---|:---|
| Package | `lemonade-server` **11.9.0~24.04** |
| CLI | `lemonade version 11.9.0` |
| Server | `/usr/bin/lemond`, default **13305** |
| OpenAI | `http://127.0.0.1:13305/api/v1` (some clients also try `/v1`) |
| GPU backend | **`llamacpp:vulkan`** (on kernel 6.17, `llamacpp:rocm` reported *Linux kernel missing support*) |
| NPU backend | this tutorial recommends **native FastFlowLM on 8219** (see below) |
| systemd | `lemond.service`, working directory `/var/lib/lemonade` |

---

### 1. Install the server

```bash
sudo add-apt-repository -y ppa:lemonade-team/stable
sudo apt update
sudo apt install -y lemonade-server
```

```bash
lemonade --version
lemonade status
```

You should see `lemonade version 11.9.0` (or newer). When the server is up: `Server is running on port 13305`. Open `http://127.0.0.1:13305`.

APT usually enables **`lemond.service`** (user `lemonade`, `ProtectHome=yes`). To stop:

```bash
sudo systemctl stop lemond
sudo systemctl disable --now lemond   # optional: no autostart
```

> ⚠️ The unit name is **`lemond`**, not `lemonade-server`.

---

### 2. Install inference backends

```bash
lemonade backends
lemonade backends install llamacpp:vulkan   # skip if already installed
lemonade backends
```

`llamacpp:rocm` needs a kernel that meets [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html). Otherwise use Vulkan.

**About `flm:npu`:**

- `lemonade backends install flm:npu` downloads FLM from GitHub and often times out in mainland China.
- Even with a system `flm`, `lemond.service` runs as `lemonade` with **ProtectHome=yes**, so it **cannot read** `~/.config/flm`.
- This tutorial therefore runs NPU via [FastFlowLM](./fastflowlm.md) + `flm serve` on **8219**.
- If you run `lemond` in the foreground as your login user (no ProtectHome) and successfully install `flm:npu`, you can try `lemonade load <flm-tag>`.

Do not `source /opt/rocm` on Lemonade’s NPU path.

---

### 3. Health check

```bash
lemonade --version
lemonade status
lemonade backends
ss -tlnp | grep 13305
```

Optional: `bash src/npu/scripts/check_lemonade.sh`. For NPU also run `bash src/npu/scripts/check_flm_npu.sh` after unsetting ROCm paths.

---

### 4. Common commands

| Command | Role |
|:---|:---|
| `lemonade backends` | installed / available backends |
| `lemonade status` | server on 13305, loaded model |
| `lemonade list` | catalog; `--downloaded` for local only |
| `lemonade pull NAME` | download (`--source modelscope` in China) |
| `lemonade load NAME` | load into the server |
| `lemonade unload` | unload |
| `lemonade run NAME` | load + open Web UI |
| `lemonade chat` | terminal REPL |

GPU backend pin:

```bash
lemonade load Gemma-4-E4B-it-GGUF --llamacpp vulkan
```

---

### 5. Ports in this tutorial

| Port | Use |
|:---|:---|
| **13305** | Lemonade Server |
| **8219** | native `flm serve` (**exclusive** with Lemonade NPU) |

Next: [Gemma 4 Lemonade GPU](/01-deploy/gemma4/lemonade-gpu-deploy.md) · [Gemma 4 Lemonade NPU](/01-deploy/gemma4/lemonade-npu-deploy.md) · [NPU troubleshooting](./npu-troubleshooting.md)
