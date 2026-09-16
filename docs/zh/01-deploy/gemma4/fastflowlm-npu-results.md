## FastFlowLM NPU 运行结果：Gemma 4 E4B

本页数字来自 **2026-09-16** 在 GMKtec EVO-X2（Ryzen AI MAX+ 395 / XDNA2）上的实测。运行时 FastFlowLM **1.0.5**，`--pmode performance`。部署步骤见 [FastFlowLM 部署](./fastflowlm-npu-deploy.md)。

对照官方 [Gemma 4 NPU 基准](https://fastflowlm.com/docs/benchmarks/gemma4_results/)（Kraken Point / 32 GB，FLM 0.9.40）。原始表：[bench CSV](./assets/bench_gemma4-it_e4b_20260916.csv)。

---

### 环境

NPU `/dev/accel/accel0`，固件 **1.1.2.65**，memlock unlimited。本地已装 `gemma4-it:e4b`。

---

### 服务跑通（OpenAI 兼容 8219）

短问「只回答一个词：ok」返回 `ok`。

| 项 | 值 |
|:---|:---|
| 客户端时延 | 1.52 s |
| TTFT | 1.35 s |
| 进程 RSS | **9275 MiB（约 9.06 GiB）** |

视觉探测（红圆 + 蓝矩形）判定正确，说明 `vision_weight.q4nx` 已加载。

---

### 1k–32k 阶梯（`flm bench`）

停掉 `flm serve` 后：

```bash
unset ROCM_PATH HIP_PATH LD_LIBRARY_PATH
flm bench gemma4-it:e4b --bench-iterations 2 --pmode performance
```

本机 2 次迭代结果：

| 上下文 | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | 官方 Kraken decode | 官方 Kraken prefill |
|:---|---:|---:|---:|---:|---:|
| 1k | 2.176 ± 0.005 | 449.5 ± 1.1 | **11.75 ± 0.02** | 12.6 | 441 |
| 2k | 3.479 ± 0.001 | 559.7 ± 0.1 | **11.38 ± 0.03** | 12.3 | 572 |
| 4k | 6.175 ± 0.023 | 629.1 ± 2.4 | **10.94 ± 0.00** | 11.6 | 668 |
| 8k | 11.685 ± 0.030 | 664.0 ± 1.7 | **9.95 ± 0.02** | 10.6 | 720 |
| 16k | 24.293 ± 0.030 | 638.3 ± 0.8 | **8.54 ± 0.01** | 9.0 | 695 |
| 32k | 56.764 ± 0.175 | 546.1 ± 1.7 | **6.61 ± 0.01** | 6.8 | 586 |

**怎么读：**

1. **Decode 全阶梯贴着官方 Kraken，略低 3–7%。** 1k 为 11.75 vs 12.6；32k 为 6.61 vs 6.8。曲线形状一样：上下文越长越慢。Halo 没有数量级优势。
2. **Prefill 1k 与官方几乎重合（450 vs 441），中段偏低。** 峰值在 8k（664 vs 官方 720），32k 掉到 546（官方 586）。
3. **TTFT 近似随长度线性变长。** 1k≈2.2 s，32k≈56.8 s。`--prefill-chunk-len` 默认 4096。
4. **API 短问 decode 约 12.5 tok/s 高于 bench@1k 的 11.75。** 短问 KV 几乎为空；bench@1k 是先灌 1k 再 decode。比 decode 请用这条阶梯。
5. **短问算出来的 prefill 无效。** 那是几十个 token 上的固定开销，不是稳态带宽。

---

### 并发（serve 模式）

短问 `ok`，并发 1 / 2 / 4，全部成功、无 503：

| 并发 | 成功 | 整波墙钟 | 平均时延 | 最大时延 |
|:---|:---|:---|:---|:---|
| 1 | 1/1 | 1.39 s | 1.39 s | 1.39 s |
| 2 | 2/2 | 3.12 s | 2.42 s | 3.12 s |
| 4 | 4/4 | 6.64 s | 4.27 s | 6.63 s |

墙钟约 **1× / 2.2× / 4.8×**。NPU 一次只跑一路，其余进 `--q-len` 队列。

---

### 结论

- 部署路径：`flm pull` → `flm serve` → `/v1/chat/completions`。
- 文本、视觉都通；常驻约 **9 GiB**。
- 1k–32k decode **11.75 → 6.61 tok/s**，与官方 Kraken E4B 同级略低。
- 要比长上下文 prefill，用 `flm bench`，不要用短 chat 的 TTFT 反推。
