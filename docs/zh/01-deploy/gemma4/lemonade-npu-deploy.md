## Lemonade NPU 零基础部署（Ubuntu 24.04 + FastFlowLM）

Lemonade 的 NPU 后端就是 **FastFlowLM**。权重与原生 `flm` 相同：`gemma4-it:e4b`。差别只是 API 理论上走 **13305**，而不是 8219。

> 前置条件：[Lemonade 环境](/zh/00-environment/lemonade.md) · [FastFlowLM 环境](/zh/00-environment/fastflowlm.md)。完整 FLM 操作与 1k–32k 数字：[FastFlowLM 部署](./fastflowlm-npu-deploy.md)。

---

### 本教程建议

APT 安装的 `lemond.service` 以用户 `lemonade` 运行，且 **ProtectHome=yes**，读不到 `~/.config/flm`。`lemonade backends install flm:npu` 还会去 GitHub 拉 FLM 包，国内常超时。

**因此本教程 NPU 的稳定路径是原生 FastFlowLM 8219**，不要卡在 `lemonade load gemma4-it:e4b`。

| 项 | 值 |
|:---|:---|
| FLM tag | `gemma4-it:e4b` |
| 推荐 API | `http://127.0.0.1:8219/v1` |
| 设备 | XDNA2 `/dev/accel/accel0` |

---

### 1. 互斥

```bash
lemonade unload 2>/dev/null || true
pkill -x flm 2>/dev/null || true
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm validate
```

`flm validate` 通过后再决定走原生 serve，还是尝试 Lemonade load。

---

### 2. 推荐：原生 FLM（8219）

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm pull gemma4-it:e4b --modelscope 1   # 已装可跳过
flm serve gemma4-it:e4b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

前端指到 `http://127.0.0.1:8219/v1`，模型 `gemma4-it:e4b`。步骤展开见 [FastFlowLM 部署](./fastflowlm-npu-deploy.md)。

---

### 3. 可选：前台 Lemonade 调 FLM（13305）

仅当你满足以下全部条件时再试：

1. 不用 systemd `lemond`（或关掉 ProtectHome）
2. 已成功 `lemonade backends install flm:npu`
3. Lemonade 进程能读到与 `flm list` 相同的模型缓存

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
lemonade pull gemma4-it:e4b
lemonade load gemma4-it:e4b --flm-args '--pmode performance --q-len 20'
lemonade status
```

`lemonade pull` 不认该 id 时，只用第 2 节原生路径。

---

### 4. 停止

```bash
lemonade unload 2>/dev/null || true
sudo systemctl stop lemond 2>/dev/null || true
pkill -x flm 2>/dev/null || true
```

Lemonade 停干净后若 8219 仍在，说明底层 `flm` 还在，必须停 `flm`。NPU 同时只能一路。
