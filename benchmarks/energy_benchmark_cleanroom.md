# Clean-Room ANE Energy Benchmark

> GPU FP16 → ANE FP16 → ANE Ternary → ANE Sparse Ternary
> Apple M4 Pro · 2026-03-27

## Conditions

| | |
|---|---|
| Hardware | Apple M4 Pro |
| Input shape | (1, 64, 2048) (seq_len=64 tokens/pass) |
| Models | CoreML .mlpackage, ANE routed (CPU_AND_NE) |
| FP16 model | 1804.1 MB |
| Ternary 2-bit model | 225.6 MB |
| Compression | 8.0x |
| Power sampling | 500ms intervals, powermetrics |
| Samplers | ane_power, cpu_power, gpu_power |
| Environment | Clean-room (apps closed, displays off) |

## Baseline (Idle)

| Subsystem | Power |
|-----------|------:|
| ANE | 0.000 W |
| CPU | 0.349 W |
| GPU | 0.028 W |

## ANE Power: FP16 vs Ternary

| Metric | FP16 (ANE) | Ternary 2-bit (ANE) | Delta | % Change |
|--------|:----------:|:-------------------:|:-----:|:--------:|
| ANE Power (W) | 5.548 | 6.899 | -1.352 | -24.4% |
| Latency (ms) | 15.06 | 7.38 | -7.68 | -51.0% |
| Tokens/sec | 4250 | 8677 | +4428 | +104.2% |
| **Tokens/Watt** | **766** | **1258** | **+492** | **+64.2%** |

## Detailed Power Breakdown

| Backend | ANE (W) | CPU (W) | GPU (W) | Latency (ms) | Tok/s | Tok/W |
|---------|:-------:|:-------:|:-------:|:------------:|:-----:|:-----:|
| FP16 ANE | 5.548 | 0.349 | 0.031 | 15.06 | 4250 | 766 |
| Ternary 2-bit ANE | 6.899 | 0.287 | 0.031 | 7.38 | 8677 | 1258 |
| FP16 GPU | 0.000 | 0.479 | 18.600 | 24.09 | 2657 | 138 |

## Sparse Channel-Pruned Results

Structured channel pruning removes the lowest-importance output channels from
the MLP intermediate dimension (5632 → 2816) and attention internal dimension
(2048 → 1229), building physically smaller Linear layers for the ANE.

| Config | Latency (ms) | Speedup vs Dense | Model Size | Params |
|--------|:------------:|:----------------:|:----------:|:------:|
| Dense ternary 2-bit | 7.30 | 1.00x | 225.6 MB | 100% |
| Sparse 20% MLP / 10% attn | 6.10 | 1.20x | 184.9 MB | 82% |
| Sparse 30% MLP / 20% attn | 5.30 | 1.38x | 162.4 MB | 73% |
| Sparse 40% MLP / 30% attn | 5.40 | 1.35x | 139.8 MB | 63% |
| **Sparse 50% MLP / 40% attn** | **4.11** | **1.78x** | **117.2 MB** | **53%** |

## Compounded Optimisation: Full Pipeline

Each stage compounds on the previous — from standard GPU FP16 to sparse
ternary on ANE, the full pipeline delivers **5.86x latency reduction** and
an estimated **9x+ energy efficiency gain**.

| Stage | Latency | Tok/s | Model Size | vs GPU FP16 | Technique |
|-------|:-------:|:-----:|:----------:|:-----------:|-----------|
| **GPU FP16** | 24.09 ms | 2,657 | 1,804 MB | 1.00x | Baseline MPS |
| **ANE FP16** | 15.06 ms | 4,250 | 1,804 MB | 1.60x | Route to Neural Engine |
| **ANE Ternary 2-bit** | 7.38 ms | 8,677 | 225.6 MB | 3.26x | Ternary quantization + 2-bit palette |
| **ANE Sparse Ternary** | 4.11 ms | 15,581 | 117.2 MB | **5.86x** | + Channel pruning (50% MLP / 40% attn) |

### Compounded gains breakdown

```
GPU FP16 → ANE FP16:           1.60x  (hardware routing)
ANE FP16 → ANE Ternary:        2.04x  (quantization + palettization)
ANE Ternary → ANE Sparse:      1.78x  (structured channel pruning)
                                ─────
GPU FP16 → ANE Sparse Ternary: 5.86x  (compounded)
```

### Energy efficiency (measured + estimated)

| Stage | Power (W) | Tok/s | Tok/W | vs GPU FP16 |
|-------|:---------:|:-----:|:-----:|:-----------:|
| GPU FP16 | 18.60 | 2,657 | 138 | 1.0x |
| ANE FP16 | 5.55 | 4,250 | 766 | 5.6x |
| ANE Ternary 2-bit | 6.90 | 8,677 | 1,258 | 9.1x |
| ANE Sparse Ternary | ~5.5* | 15,581 | ~2,833* | **~20.5x*** |

*\*Sparse power estimated from reduced compute (47% fewer parameters → proportionally lower ANE utilisation). Clean-room power measurement pending.*

### Model compression

```
FP16 dense:         1,804.1 MB   (100%)
Ternary FP16:       1,804.1 MB   (ternary values stored in FP16)
Ternary 2-bit:        225.6 MB   (8.0x compression)
Sparse ternary 2-bit: 117.2 MB   (15.4x compression)
```

## Summary

Starting from a standard GPU FP16 baseline at 24.09 ms and 138 tokens/watt,
the full ternary optimisation pipeline achieves:

- **5.86x faster** inference (24.09 ms → 4.11 ms)
- **15,581 tokens/sec** (up from 2,657)
- **15.4x model compression** (1,804 MB → 117 MB)
- **~20x energy efficiency** gain (estimated, pending clean-room power measurement)

Three independent techniques compound multiplicatively:
1. **ANE routing** (1.6x) — Apple's Neural Engine vs GPU for matrix multiply
2. **Ternary 2-bit quantization** (2.0x) — {-α, 0, +α} weights in 2-bit palette
3. **Structured channel pruning** (1.78x) — remove low-importance channels entirely

The 2-bit palette maps exactly to ternary {-α, 0, +α} weights: 4 palette entries,
3 used. Channel pruning converts unstructured weight-level sparsity (43% zeros
spread uniformly) into structured sparsity (entire channels removed), which the
ANE can exploit through physically smaller matrix multiplications.

---
*Clean-room ANE energy benchmark · Terncore · Cubey/Synapticode · 2026-03-27*
