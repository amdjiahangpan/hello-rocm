## FastFlowLM / Lemonade 排错

NPU 与 Lemonade 环境、部署中的常见问题。GPU 的 ROCm / HIP 问题仍见 [00-Environment 校验](/zh/00-environment/#三校验安装)。

| 现象 | 处理 |
|:---|:---|
| 没有 `flm` | 按 [FastFlowLM 环境](./fastflowlm.md) 装 deb，然后 **reboot** |
| `No NPU device found` | 查 `/dev/accel/accel0` 与 `render` 组；`sudo usermod -aG render,video $USER` 后重新登录 |
| SVA bind `-19` | cmdline 必须 `amd_iommu=on iommu=pt`，改 GRUB 后 reboot |
| `No such device with index '0'` | 当前 shell **不要**带 ROCm XRT：`unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH`；再查 `xrt-smi examine` |
| 只换 1.1 固件、NPU 消失 | 驱动+固件必须成对；用 PPA 的 `amdxdna-dkms`，不要只刷 firmware |
| `modinfo amdxdna` 不在 `updates/dkms` | 未 reboot，或仍在用内核树内旧模块 |
| `ulimit -l` 不是 unlimited | 写入 `limits.conf` 后重新登录；当前会话可试 `ulimit -l unlimited` |
| NPU 与大 GTT 配方冲突 | GTT 常用 `amd_iommu=off`，和 FLM 互斥 |
| 没有 `lemonade` | `sudo apt install lemonade-server` |
| `status` 连不上 13305 | 看进程 `/usr/bin/lemond`；`ss -tlnp \| grep 13305`；`sudo systemctl start lemond` |
| 13305 停不掉 | `sudo systemctl stop lemond`（单位名不是 lemonade-server） |
| `llamacpp:rocm` = kernel missing support | 改用 `llamacpp:vulkan`；ROCm 见 [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html) |
| `flm:npu` 未装 / GitHub 超时 | 国内拉 FLM tar.gz 常失败。NPU 用原生 [FastFlowLM](./fastflowlm.md) |
| `load gemma4-it:e4b` 404 | systemd `lemond` 的 ProtectHome 看不到用户 FLM 缓存；走 8219 |
| Lemonade 和 `flm serve` 抢 NPU | 先 `lemonade unload` + `sudo systemctl stop lemond` 或 `pkill -x flm`，只留一路 |
| 503 | NPU 队列满，降并发或加大 `--q-len` |
| 和 GPU 大模型一起 OOM | 统一内存；先停 vLLM / Ollama / 大 GGUF，再只起 FLM |
| `flm validate` 在沙箱里失败 | 无 `/dev/accel` 属正常；在真实终端跑 |

日志位置：

```bash
# FastFlowLM 前台日志，或你重定向的文件
journalctl -u lemond -n 80 --no-pager
```
