# Sparse Channel-Pruned ANE Benchmark

> Structured channel pruning for faster ternary inference on ANE
> Apple M4 Pro · 2026-03-27 13:31

## Method

The ANE executes dense matrix multiplications — it cannot skip individual zero
weights. To exploit ternary sparsity (7.23 ms dense baseline),
we use **structured channel pruning**:

1. Quantize to ternary {-α, 0, +α} (43% weight sparsity)
2. Score each channel by L1 importance (geometric mean for MLP gate/up pairs)
3. Remove lowest-importance channels entirely from MLP intermediate and attention dims
4. Build physically smaller Linear layers → smaller matmuls on ANE
5. Apply 2-bit palettization → CoreML → ANE dispatch

**Pruning targets:**
- MLP intermediate (gate/up/down_proj): 5632 → varies — dominates compute
- Attention dim (q/o_proj): 2048 → varies — secondary target
- k/v_proj: not pruned (already 256-dim)

## Configuration

| | |
|---|---|
| Hardware | Apple M4 Pro |
| Blocks | 22 |
| Input | (1, 64, 2048) (seq=64 tokens) |
| Warmup | 10 |
| Measured runs | 50 |
| Compute units | CPU_AND_NE (ANE) |
| Baseline | Dense ternary 2-bit = 7.23 ms |

## Results

| Config | Latency (ms) | Min (ms) | Speedup | Model Size | Params |
|--------|:------------:|:--------:|:-------:|:----------:|:------:|
| Dense Ternary 2-bit (ANE) | 7.23 | 7.15 | 1.00x | 225.6 MB | 100% |
| Sparse 20% MLP / 10% attn (ANE) | 6.29 | 6.19 | 1.15x | 184.9 MB | 82% |
| Sparse 30% MLP / 20% attn (ANE) | 5.33 | 5.24 | 1.36x | 162.4 MB | 73% |
| Sparse 40% MLP / 30% attn (ANE) | 4.68 | 4.60 | 1.55x | 139.8 MB | 63% |
| Sparse 50% MLP / 40% attn (ANE) | 4.02 | 3.96 | 1.80x | 117.2 MB | 53% |

## Best Configuration

**Sparse 50% MLP / 40% attn (ANE)** achieves the best latency:

- **4.02 ms** vs **7.23 ms** dense baseline
- **1.80x speedup** from channel pruning
- **117.2 MB** model size (vs 225.6 MB dense)
- **46.9% parameter reduction**

## Tokens per Second

| Config | Tok/s | vs Dense |
|--------|:-----:|:--------:|
| Dense Ternary 2-bit (ANE) | 8849 | 1.00x |
| Sparse 20% MLP / 10% attn (ANE) | 10181 | 1.15x |
| Sparse 30% MLP / 20% attn (ANE) | 12012 | 1.36x |
| Sparse 40% MLP / 30% attn (ANE) | 13672 | 1.55x |
| Sparse 50% MLP / 40% attn (ANE) | 15906 | 1.80x |

## Analysis

Structured channel pruning converts unstructured ternary sparsity (where 43%
of individual weights are zero but no full channels are zero) into structured
sparsity by removing the least-important channels entirely. The ANE then
processes physically smaller weight matrices, reducing both latency and energy.

The MLP layers (gate/up/down_proj at 5632 intermediate dim) account for ~70%
of total FLOPs per block, making them the highest-impact pruning target.

---
*Sparse channel-pruned ANE benchmark · Terncore · Cubey/Synapticode · 2026-03-27*
