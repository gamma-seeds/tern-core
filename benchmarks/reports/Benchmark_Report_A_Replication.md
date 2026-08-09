# BUILD 6 Track A — TurboQuant KV Replication Report

> **Hardware:** Apple M4 Pro · 64 GB · Darwin 25.5.0 · PyTorch 2.7.0
> **Date:** 2026-08-08
> **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22 layers, 4 KV heads GQA, 64 head_dim)

---

## Verdict

**The 5.3× figure is the theoretical bits-per-coordinate ratio (16/3 = 5.33×). The measured byte ratio with the actual TurboQuant encoding is 3.88×.** Encoding overhead (norms, codebook indices, rotation matrices, QJL state) accounts for the gap. The theoretical ratio reproduces exactly; the measured ratio is the operationally honest number. Both are legitimate, at different levels of abstraction.

---

## Original vs Replication

| Metric | Original (2026-03-30) | Replication (2026-08-08) | Tag |
|--------|:---------------------:|:------------------------:|:---:|
| Theoretical ratio (16/b_mse bits) | 5.33× | 5.33× | Derived |
| **Measured byte ratio** | *not separately reported* | **3.88×** | **Measured** |
| Prefill encode overhead | 12.4 ms | 68.7 ms | Measured |
| Per-token encode | 0.38 ms | 22.1 ms | Measured |
| Uncompressed KV bytes | — | 2,523,136 | Measured |
| Compressed KV bytes | — | 650,496 | Measured |

### Timing difference analysis

The original reported 12.4 ms prefill / 0.38 ms per-token; replication shows 68.7 ms / 22.1 ms. Likely causes:
- **Float32 vs Float16 model precision**: original may have loaded in FP16; replication uses float32
- **Library version drift**: torch 2.7.0 vs original torch version; turboquant internal changes
- **Thermal state and background load**: not controlled in either run

The timing figures are part of the overhead characterisation, not the compression claim.

### PPL difference

| | Original | Replication |
|---|:---:|:---:|
| Baseline PPL | 7.82 | 5.54 |
| Method | "5 sentences, teacher-forcing" | WikiText-2, 2048-token teacher-forcing |

Different evaluation methodology explains the PPL gap. The R7-B autoregressive methodology (from the R12 sweep) produced a baseline of 7.94, closer to the original. The PPL figures in both cases are **model baseline** (no KV compression effect) because the measurement is **open-loop**.

---

## Architecture finding: n_heads clarification

The original config reported `n_heads=32`. TinyLlama uses Grouped Query Attention:
- **32 attention heads** (query projections)
- **4 KV heads** (key/value projections)

The KV cache has 4 head slots per layer, not 32. The compressor operates on KV heads. The original config's `n_heads=32` referred to attention heads; the IncrementalTQCompressor's per-layer detection reads the correct shape from the live cache regardless.

---

## Open-loop scope — what the measurement covers

The TurboQuant compression is **open-loop**: `compressor.append()` encodes the KV cache as a side effect, but the model's next forward pass uses the **original uncompressed** `past_key_values`. This means:

- **Compression ratio**: genuinely measured — the compressed bytes are real TurboQuant output
- **Encoding overhead**: genuinely measured — the time cost of running TurboQuant encode
- **Quality impact**: NOT measured — the PPL is the model's own baseline, independent of KV compression

The R12 KV-cache-compression diagnostic (2026-05-18) attempted **closed-loop** round-trip (encode → decode → substitute compressed KV back into the model). All four completed sweep points (b_mse=6,5,4,3) returned **NaN perplexity**. The round-trip decode produces numerically unstable output. This remains an open engineering gap.

---

## Run-by-run

| Run | Compression | Prefill encode | Per-token encode |
|:---:|:-----------:|:--------------:|:----------------:|
| 1 | 3.88× | 76.5 ms | 20.9 ms |
| 2 | 3.88× | 68.7 ms | 22.1 ms |
| 3 | 3.88× | 68.5 ms | 22.4 ms |

Consistent across all 3 runs — the ratio is deterministic (same model, same config, same precision).

---

## Config

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- b_mse: 3
- Method: IncrementalTQCompressor open-loop
- Prompt: "The capital of France is"
- Generated tokens: 50
- Runs: 3 (median reported)
- Device: cpu, dtype: float32

---

## Recommendation for deck v4.0

The published 5.3× should be reframed:

| What to say | Source | Tag |
|---|---|---|
| 5.3× theoretical KV coordinate compression (16→3 bits) | Codebook encoding math | Derived |
| 3.9× measured byte compression (encoding overhead included) | This replication | Measured |
| Quality preservation under closed-loop: **undemonstrated** | R12 sweep NaN result | Gap |

If the deck carries the 5.3× figure, it should be tagged "Derived (bits-per-coordinate)" and the measured byte ratio noted alongside. If only one number ships, the measured 3.9× is the defensible one.
