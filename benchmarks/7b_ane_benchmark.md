# 7B-Scale ANE Benchmark — Ternary vs FP16

> Llama 2 7B dimensions: 4096 hidden, 11008 intermediate, 32 blocks
> 6.48B linear params · Apple M4 Pro · 2026-03-27

## Results

| Backend | Mean (ms) | Min (ms) | vs MPS FP16 | vs Dense ANE |
|---------|:---------:|:--------:|:-----------:|:------------:|
| MPS FP16 | 147.83 | 147.38 | 1.00x | 0.27x |
| CoreML FP16 (ALL) | 120.49 | 119.88 | 1.23x | 0.33x |
| CoreML FP16 (ANE) | 144.72 | 140.27 | 1.02x | 0.27x |
| MPS Ternary | 147.83 | 147.40 | 1.00x | 0.27x |
| Ternary 2-bit (ALL) | 39.22 | 39.07 | 3.77x | 1.00x |
| Ternary 2-bit (ANE) | 39.20 | 39.06 | 3.77x | 1.00x |
| Ternary 2-bit (GPU) | 186.96 | 183.03 | 0.79x | 0.21x |
| Sparse 30% MLP / 20% attn (ALL) | 29.76 | 29.37 | 4.97x | 1.32x |
| Sparse 30% MLP / 20% attn (ANE) | 29.86 | 29.30 | 4.95x | 1.31x |
| Sparse 40% MLP / 30% attn (ALL) | 24.88 | 24.85 | 5.94x | 1.58x |
| Sparse 40% MLP / 30% attn (ANE) | 24.89 | 24.86 | 5.94x | 1.58x |

## Model Sizes

| Format | Size | vs FP16 |
|--------|-----:|--------:|
| FP16 CoreML | 10304.1 MB | 1.0x |
| Ternary 2-bit | 1288.1 MB | 8.0x |
| Sparse 30% MLP / 20% attn | 927.4 MB | 11.1x |
| Sparse 40% MLP / 30% attn | 798.6 MB | 12.9x |

## Scale Comparison — 7B vs 270M

| Metric | TinyLlama (270M) | Llama 7B (6.5B) |
|--------|:----------------:|:----------------------------:|
| MPS FP16 | 26.22 ms | 147.83 ms |
| Dense 2-bit ANE | 7.23 ms | 39.20 ms |
| Dense vs MPS | 3.63x | 3.77x |
| Best sparse ANE | 4.68 ms | 24.89 ms |
| Best sparse vs MPS | 5.60x | 5.94x |

## Method

Linear stack matching Llama 2 7B: 32 blocks × 7 linear layers = 224 matmuls.
Same methodology as TinyLlama benchmark — coremltools cannot convert full
transformer ops (RoPE/attention), so we benchmark the linear layers which
are the bottleneck and where ternary provides the speedup.

Channel pruning targets MLP intermediate (11008 → pruned) and attention
internal dimension (4096 → pruned). k/v projections kept at full size.

---
*7B ANE benchmark · Terncore · Cubey/Synapticode · 2026-03-27*
