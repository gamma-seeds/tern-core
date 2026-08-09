# BUILD 6.1 — Track A Accounting Reconciliation

> **Date:** 2026-08-09
> **Input:** Track A replication (TinyLlama, 3.88×), recovered Llama-3.2-1B bench (0.39×/2.20×/6.74×)
> **Purpose:** Reconcile Track A's 3.88× under the opt-A and opt-B accountings from the recovered bits_per_slot formula, so Rob + Fable can rule once on the public figure.

## The bits_per_slot formula (from recovered Llama-3.2-1B bench)

```
bytes_per_slot = d × (b_mse + 1) / 8 + 6
             = 64 × (3 + 1) / 8 + 6
             = 32 + 6 = 38 bytes
```

Breakdown: PQ b_mse-bit indices (24 B) + QJL 1-bit signs (8 B) + PQ FP16 norm (2 B) + QJL FP32 r_norm (4 B)

Regenerable shared state per (layer, head, K/V): qjl.S (d×d FP32 = 16,384 B) + PQ rotation signs (d FP32 = 256 B) = **16,640 B**

## TinyLlama: three accountings at 56 tokens

Track A parameters: 22 layers, 4 KV heads, 64 head_dim, float32 KV, b_mse=3, seq_len=56

| Accounting | Compressed bytes | Ratio | Description | Tag |
|---|---:|---:|---|---|
| Track A raw (`_count_object_bytes`) | 650,496 | **3.88×** | In-memory tensor walk of compressor objects | Measured |
| Opt A (formula-packed, all bytes) | 3,303,168 | **0.76×** (expansion) | Packed per-token + regenerable shared | Derived |
| Opt B (formula-packed, shared excluded) | 374,528 | **6.74×** | Packed per-token only | Derived |

Uncompressed baseline: 2,523,136 bytes (float32 KV at 45,056 B/token × 56 tokens). Measured, Track B2.

## Why 3.88× matches neither opt A nor opt B

Track A's `_count_object_bytes` walks the compressor's Python objects and counts tensor storage. This is a third, distinct accounting:

- **Smaller than opt A** because shared state (qjl.S, rotation) is stored once per compressor instance, not duplicated across every slot the formula accounts for.
- **Larger than opt B** because per-token indices and signs are held as float32 tensors in RAM, not packed to formula-optimal bit widths.

The Llama-3.2-1B bench used a different raw-state measurement and got 0.39× (expansion) — demonstrating that raw in-memory accounting is model-specific and method-specific. It measures implementation overhead, not encoding efficiency.

## Scaling: how opt A converges on opt B

Shared state (2,928,640 B for TinyLlama) is fixed per model; per-token cost scales linearly. At serving scale, shared amortises toward zero.

| Sequence length | Opt A ratio | Opt B ratio | Shared as % of opt A |
|---:|---:|---:|---:|
| 56 | 0.76× (expansion) | 6.74× | 88.7% |
| 256 | 3.64× | 6.74× | 62.4% |
| 1,024 | 4.72× | 6.74× | 30.0% |
| 4,096 | 6.08× | 6.74× | 9.7% |
| 16,384 | 6.57× | 6.74× | 2.5% |

## The published 5.33× — a fourth accounting

The original 5.33× is the coding-efficiency upper bound: 16 bits FP16 / 3 bits b_mse = 5.33×. It excludes:

- QJL sign bits (1 bit/coordinate)
- Per-slot norms (6 bytes)
- Regenerable shared state
- The float32 vs FP16 reference-frame choice

It is also measured against an **FP16 reference**, while Track A and Track B2 operate on float32 KV. Against float32 the same pure-coding ratio is 32/3 = 10.67×.

## Complete ratio map (d=64, b_mse=3)

| Accounting | vs float32 KV | vs FP16 KV | Overhead included |
|---|---:|---:|---|
| Pure coding (32/b_mse or 16/b_mse) | 10.67× | 5.33× | None |
| Packed + QJL signs | 8.00× | 4.00× | QJL 1-bit/coord |
| Opt B (formula, shared excluded) | 6.74× | 3.37× | + norms (6 B/slot) |
| Opt A at 4096 tokens | 6.08× | 3.04× | + shared, amortised |
| Track A raw (TinyLlama, 56 tokens) | 3.88× | — | Implementation-specific |
| Opt A at 56 tokens | 0.76× | — | + shared, dominant |

## Recommendation for ClaimSheet

1. **Opt B (6.74× vs float32)** is the operationally honest metric for NPU serving — shared state is regenerable (recomputed from model weights + RNG seed, never stored to disk or transferred), and amortises to negligible cost at serving sequence lengths. Defensible statement: "KV cache compressed to 38 bytes per slot vs 256 bytes uncompressed (float32), 6.7× at serving scale."

2. **Opt A at stated seq_len** is the honest metric for a single short session — it shows the actual memory footprint including regenerable state. At typical serving lengths (1K+) it converges on opt B.

3. **5.33× retires** from the claim set — it omits QJL signs and norms (real per-token overhead), and measures against an FP16 reference that the model doesn't use. It served as a design-stage compass, not a measured result.

4. **3.88× retires** from the claim set — it's an implementation artifact of how the Python compressor stores tensors in RAM, not the packed output format.

Both opt A and opt B carry **no quality verdict**. Both Track A and the recovered Llama-3.2-1B bench are open-loop (compressor encodes as side effect, model uses uncompressed KV). R12 closed-loop produced NaN. B3 native-ternary closed-loop produced 1–15% argmax match. **No TurboQuant or native-ternary KV approach has demonstrated quality at any compression level.**

## Cross-model verification

| | TinyLlama (22L, 4H, d=64, fp32) | Llama-3.2-1B (16L, 8H, d=64, fp32) |
|---|---|---|
| Opt B | 6.74× | 6.74× |
| Opt A @ 56/106 tokens | 0.76× | 2.20× |

Opt B is identical because it depends only on d and b_mse, which are the same. Opt A differs because the shared-state-to-token ratio varies (TinyLlama: 22×4=88 head-units at 56 tokens; Llama-3.2-1B: 16×8=128 head-units at 106 tokens).
