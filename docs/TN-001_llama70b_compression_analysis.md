# TN-001: Ternary Compression Analysis — Meta-Llama-3.1-70B

**Classification:** Gamma Seeds Pte Ltd — Internal Technical Note  
**Author:** tern-core v0.5.0 streaming pipeline  
**Date:** 2026-04-10  
**Status:** KSGC-grade empirical result  
**Patent relevance:** Patent 4 (Progressive Compression), Patent 12 (Auto conversion)

---

## 1. Summary

We ran the first full ternary compression analysis on a 70B-parameter
frontier model (Meta-Llama-3.1-70B) using the v0.5.0 streaming pipeline.
The analysis quantifies the compression/quality tradeoff and identifies the
architectural bottleneck that limits ternary compression on GQA models.

**Key finding:** A flat quantisation threshold has no meaningful leverage on
70B GQA architecture. Adaptive per-layer thresholds improve compression by
only 0.01x (1.60x to 1.61x). The path to 3x+ compression on frontier
models is mixed ternary/INT4 quantisation, not threshold tuning.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Meta-Llama-3.1-70B (GQA, 80 blocks, 8192 hidden) |
| Safetensors shards | 30 files, 131 GB on disk |
| Hardware | Mac Mini M4 Pro, 64 GB RAM |
| Pipeline | tern-core v0.5.0 streaming (28 GB peak RAM) |
| Baseline PPL | 6.06 (FP16 calibration text, 120 tokens) |
| Eligible layers | 560 (7 Linear per block x 80 blocks) |
| Protected by pattern | embed_tokens, lm_head, model.norm, all LayerNorm |

---

## 3. PPL Headroom Sweep

Threshold fixed at 0.7, headroom varied.

| Headroom | PPL ceiling | Layers converted | % | Compression | Pred PPL |
|----------|-------------|-----------------|---|-------------|----------|
| 5% | 6.36 | 31/560 | 5.5% | 1.10x | 6.08 (+0.3%) |
| 20% | 7.27 | 125/560 | 22.3% | 1.60x | 6.30 (+4.0%) |

Verdict at 20%: "Good -- minimal quality loss, suitable for production."

---

## 4. Threshold Sweep (20% headroom)

All five thresholds tested across the valid range.

| Threshold | Layers converted | % | Compression | Pred PPL |
|-----------|-----------------|---|-------------|----------|
| 0.5 | 123/560 | 22.0% | 1.59x | 6.30 |
| 0.6 | 125/560 | 22.3% | 1.60x | 6.30 |
| **0.7** | **125/560** | **22.3%** | **1.60x** | **6.30** |
| 0.8 | 126/560 | 22.5% | 1.61x | 6.30 |
| 0.9 | 123/560 | 22.0% | 1.59x | 6.30 |

**The threshold has almost no effect.** Across the full 0.5-0.9 range,
convertible layers vary by 3 (123-126), compression moves 0.02x, and
predicted PPL is flat at 6.30.

---

## 5. Per-Layer Adaptive Threshold Analysis

For each of the 560 eligible layers, we swept 9 threshold values
(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99) and selected the
threshold that minimises reconstruction error.

### Optimal threshold distribution

| Optimal t | Layers | % of total |
|-----------|--------|-----------|
| 0.70 | 16 | 2.9% |
| **0.80** | **469** | **83.8%** |
| 0.90 | 62 | 11.1% |
| 0.95 | 10 | 1.8% |
| 0.99 | 3 | 0.5% |

84% of all eligible layers prefer t=0.80. The reconstruction error
curve has a U-shape that bottoms out around 0.8 for nearly every
layer in the 70B model.

### Adaptive vs flat compression

| Strategy | Layers | Params converted | Compression |
|----------|--------|-----------------|-------------|
| Flat t=0.7 | 125/560 | 29.4B | 1.60x |
| Adaptive | 126/560 | 29.6B | 1.61x |
| **Delta** | **+1** | **+235M** | **+0.01x** |

**Adaptive threshold is a dead end on 70B GQA.** The per-layer error
improvement from threshold tuning (~1.2%) is dwarfed by the gap between
"tolerant" and "sensitive" layers (50%+ in relative error).

---

## 6. Sensitivity Ranking

### Most tolerant layers (convert first)

All are MLP up_proj/down_proj in deep transformer blocks (layers 26-77):

1. `model.layers.70.mlp.up_proj` — error 0.4152, sparsity 0.4241
2. `model.layers.74.mlp.up_proj` — error 0.4153, sparsity 0.4242
3. `model.layers.77.mlp.up_proj` — error 0.4154, sparsity 0.4254

### Least tolerant layers (never convert to ternary)

All are layer-0 weights — first transformer block:

1. `model.layers.0.mlp.gate_proj` — error 0.9964, sparsity 0.5130
2. `model.layers.0.mlp.up_proj` — error 0.9964, sparsity 0.5093
3. `model.layers.0.self_attn.q_proj` — error 0.9925, sparsity 0.7852

Layer 0 has ~0.99 reconstruction error regardless of threshold — nearly
total information loss under ternary quantisation. This is consistent with
the literature: first-layer weights encode positional/tokenisation structure
that is destroyed by aggressive quantisation.

### Error gap analysis

The tolerant layers cluster at error ~0.41-0.44, then the error curve
jumps sharply to ~0.49+ with no intermediate plateau. This step function
means the 70B model has a natural "ternary-safe" zone of ~22% of layers.
Beyond that zone, quality degrades rapidly regardless of threshold.

---

## 7. Architectural Explanation

Llama-3.1-70B uses **Grouped Query Attention (GQA)** with:
- Q projections: 8192 x 8192 (67M params)
- K/V projections: 1024 x 8192 (8M params, 8x smaller)
- MLP projections: 28672 x 8192 (235M params)

The small K/V projections have higher relative reconstruction error
because they carry concentrated information in fewer parameters. The
large MLP projections are more tolerant because their information is
spread across more parameters — redundancy enables compression.

This differs fundamentally from smaller models like Mistral-7B where
the K/V projections are proportionally larger relative to hidden_size,
giving the threshold more leverage.

---

## 8. Path to 3x+ Compression

The bottleneck is the 78% of layers stuck in FP16. Threshold tuning
cannot move them. The options:

| Strategy | Expected ratio | Notes |
|----------|---------------|-------|
| **Mixed ternary/INT4** | **2.5-3.5x** | Sensitive layers at 0.5 B/param instead of 2 B/param |
| Mixed ternary/INT8 | 1.8-2.2x | Conservative, higher quality |
| GPTQ-style ternary | 2.0-2.5x | Hessian-weighted, not per-layer mean |
| TFH fine-tune then convert | 3.0-5.0x | Model learns to tolerate ternary |

**Recommended v0.6.0 path: mixed ternary/INT4.**

Predicted compression: 22% of params at 0.25 B/param (ternary) + 78%
at 0.5 B/param (INT4) + embed/norm at 2 B/param (~0.5%) = **~2.8x**
before any further optimisation.

This represents a model size reduction from 131 GB to ~47 GB, fitting
comfortably on a 64 GB machine for inference.

---

## 9. Pipeline Performance

| Metric | Value |
|--------|-------|
| Streaming scan time | 202s (3.4 min) |
| Threshold sweep (5 points) | 1158s (19.3 min) |
| Adaptive analysis (560 layers x 9 thresholds) | ~20 min |
| Peak RAM (scan) | 28 GB |
| Peak RAM (conversion) | OOM at 60/80 blocks — writer needs streaming (v0.5.1) |

---

## 10. Reproducibility

All results generated by:

```bash
# Baseline + sensitivity scan
python -m terncore.autoscan --model ./llama70b --perplexity-gate 0.20 \
    --streaming --baseline-ppl 6.06 --no-cache

# Threshold sweep
for t in 0.5 0.6 0.7 0.8 0.9; do
  python -m terncore.autoscan --model ./llama70b --perplexity-gate 0.20 \
      --streaming --baseline-ppl 6.06 --no-cache -t $t
done
```

Commit: `d7d436b` (tern-core v0.5.0)  
Model: meta-llama/Meta-Llama-3.1-70B (30 safetensors shards, SHA in index)

---

*Copyright (c) 2025 Gamma Seeds Pte Ltd. All rights reserved.*
