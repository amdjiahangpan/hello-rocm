# NPU 示例与脚本

配合 `docs/zh/00-environment/fastflowlm.md`、`docs/zh/00-environment/lemonade.md` 以及 `docs/zh/01-deploy/gemma4/`、`docs/zh/01-deploy/qwen36/`。

命令均在**仓库根目录**执行。跑 FastFlowLM 前先：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
```

## 示例

```bash
# 默认 Gemma 4 E4B @ :8219
python3 src/npu/examples/chat_gemma4.py

# Lemonade GPU
BASE=http://127.0.0.1:13305/api/v1 MODEL=Gemma-4-E4B-it-GGUF \
  python3 src/npu/examples/chat_gemma4.py

# Qwen3.6 NPU
BASE=http://127.0.0.1:8219 MODEL=qwen3.6-moe:35b-a3b \
  python3 src/npu/examples/chat_gemma4.py

python3 src/npu/examples/chat_gemma4_image.py ./photo.png "图里有什么？"
```

## 脚本

```bash
bash src/npu/scripts/check_flm_npu.sh
bash src/npu/scripts/check_lemonade.sh
bash src/npu/scripts/smoke_openai.sh
BASE=http://127.0.0.1:13305/api/v1 MODEL=Gemma-4-E4B-it-GGUF \
  bash src/npu/scripts/smoke_openai.sh
```
