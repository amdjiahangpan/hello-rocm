## FastFlowLM NPU results: Gemma 4 E4B

Numbers from **2026-09-16** on a GMKtec EVO-X2 (Ryzen AI MAX+ 395 / XDNA2), FastFlowLM **1.0.5**, `--pmode performance`. Deploy steps: [FastFlowLM deploy](./fastflowlm-npu-deploy.md).

Official reference: [Gemma 4 NPU benches](https://fastflowlm.com/docs/benchmarks/gemma4_results/) (Kraken Point / 32 GB, FLM 0.9.40). Raw table: [bench CSV](./assets/bench_gemma4-it_e4b_20260916.csv).

---

### Service smoke (8219)

Short prompt “reply with one word: ok” returned `ok`. Client latency 1.52 s, TTFT 1.35 s, RSS **~9.06 GiB**. Vision probe (red circle + blue rectangle) was correct.

---

### 1k–32k ladder (`flm bench`)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Official Kraken decode | Official Kraken prefill |
|:---|---:|---:|---:|---:|---:|
| 1k | 2.176 | 449.5 | **11.75** | 12.6 | 441 |
| 2k | 3.479 | 559.7 | **11.38** | 12.3 | 572 |
| 4k | 6.175 | 629.1 | **10.94** | 11.6 | 668 |
| 8k | 11.685 | 664.0 | **9.95** | 10.6 | 720 |
| 16k | 24.293 | 638.3 | **8.54** | 9.0 | 695 |
| 32k | 56.764 | 546.1 | **6.61** | 6.8 | 586 |

Decode tracks official Kraken, about 3–7% slower. Short-chat decode (~12.5 tok/s) is empty-KV; use this ladder for long-context decode. Short-chat prefill is not a steady-state bandwidth number.

Concurrency 1 / 2 / 4 on short `ok` all succeeded (no 503); wall clock ~1× / 2.2× / 4.8× because NPU runs one request at a time and queues the rest.
