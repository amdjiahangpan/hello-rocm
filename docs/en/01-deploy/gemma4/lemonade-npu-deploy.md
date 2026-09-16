## Lemonade NPU deployment (Ubuntu 24.04 + FastFlowLM)

Lemonade’s NPU backend **is** FastFlowLM. The weight tag is the same (`gemma4-it:e4b`). The API would be **13305** instead of 8219 — if Lemonade can see the FLM cache.

> Prerequisites: [Lemonade environment](/00-environment/lemonade.md) · [FastFlowLM environment](/00-environment/fastflowlm.md). Full FLM steps and 1k–32k numbers: [FastFlowLM deploy](./fastflowlm-npu-deploy.md).

APT `lemond.service` runs as user `lemonade` with **ProtectHome=yes**, so it cannot read `~/.config/flm`. `lemonade backends install flm:npu` also downloads FLM from GitHub and often times out.

**The stable NPU path in this tutorial is native FastFlowLM on 8219.**

```bash
lemonade unload 2>/dev/null || true
pkill -x flm 2>/dev/null || true
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm serve gemma4-it:e4b --host 0.0.0.0 --port 8219 --ctx-len 32768 --q-len 20 --socket 20 --cors 1
```

Point frontends at `http://127.0.0.1:8219/v1`, model `gemma4-it:e4b`.

Only try `lemonade load gemma4-it:e4b` if you run `lemond` in the foreground as your login user (no ProtectHome) **and** `flm:npu` is installed. Stop both clients when switching: `lemonade unload` and `pkill -x flm`.
