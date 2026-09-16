## FastFlowLM / Lemonade troubleshooting

Common NPU and Lemonade issues. For ROCm / HIP GPU problems, see [00-Environment verification](/00-environment/#3-verify-installation).

| Symptom | Fix |
|:---|:---|
| No `flm` | Follow [FastFlowLM environment](./fastflowlm.md), then **reboot** |
| `No NPU device found` | Check `/dev/accel/accel0` and `render` group; `sudo usermod -aG render,video $USER` then re-login |
| SVA bind `-19` | cmdline must have `amd_iommu=on iommu=pt`; update GRUB and reboot |
| `No such device with index '0'` | unset `ROCM_PATH HIP_PATH LD_LIBRARY_PATH`; check `xrt-smi examine` |
| NPU gone after firmware-only swap | driver + firmware must match; use PPA `amdxdna-dkms` |
| `modinfo amdxdna` not under `updates/dkms` | no reboot yet, or still on in-tree module |
| `ulimit -l` not unlimited | write `limits.conf`, re-login |
| NPU vs large-GTT recipe | GTT often uses `amd_iommu=off`, incompatible with FLM |
| No `lemonade` | `sudo apt install lemonade-server` |
| `status` cannot reach 13305 | process `/usr/bin/lemond`; `ss -tlnp \| grep 13305`; `sudo systemctl start lemond` |
| 13305 will not stop | `sudo systemctl stop lemond` (unit is not lemonade-server) |
| `llamacpp:rocm` = kernel missing support | use `llamacpp:vulkan`; see [gfx1151 Linux](https://lemonade-server.ai/gfx1151_linux.html) |
| `flm:npu` missing / GitHub timeout | use native [FastFlowLM](./fastflowlm.md) |
| `load gemma4-it:e4b` 404 | systemd `lemond` ProtectHome hides user FLM cache; use 8219 |
| Lemonade and `flm serve` fight over NPU | `lemonade unload` + stop `lemond`, or `pkill -x flm`; keep one client |
| 503 | NPU queue full; lower concurrency or raise `--q-len` |
| OOM with a GPU model | unified memory; stop vLLM / Ollama / large GGUF first |
| `flm validate` fails in a sandbox | no `/dev/accel` is expected; run on a real terminal |
