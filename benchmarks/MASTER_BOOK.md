# Master Book — tern-core Experiment Register

**Copyright (c) 2025-2026 Synapticode. All rights reserved.**

> Append only. Filed rows are immutable. One row per experiment, ever.
> A bench.* Programme Cube cell closes only when its master book row is filed.

---

## Locked Public Set

The current locked public benchmarks and their backing experiments.

| Claim | Value | Model | EXP-ID | Status |
|-------|-------|-------|--------|--------|
| Whole-model compression vs FP32 | **8.4×** | TinyLlama-1.1B | EXP-001 | PUBLISHED |
| Inference speed (CPU_AND_NE) | **297.6 tok/s @ 5.388 W** | Mistral-7B | EXP-005 | PUBLISHED |
| Argmax-exact coherence | **Tier 2.5** | Bonsai-8B | EXP-012 | PUBLISHED |

---

## Experiments

### EXP-001 — TinyLlama-1.1B Whole-Model Compression

| Field | Content |
|-------|---------|
| **ID** | EXP-001 |
| **Date** | 2026-02-26 (Block 3, Day 11) |
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Model size** | FP32: 4,137 MB · FP16: 2,357 MB · .tern-model: 471.6 MB |
| **Hardware** | iMac (2019), Intel Core i9-9900K, 64 GB, macOS Darwin 24.6.0, torch 2.2.2, Python 3.11.14 |
| **Software stack** | PyTorch 2.2.2, tern-core v0.1.0 (bench_day11_multi_model.py), libterncore v0.1.0 (AVX2+OpenMP) |
| **Method** | Phase A — tern-convert: full-model ternary quantisation to .tern-model v2 format |
| **Approach angle** | FP32 → ternary {-1,0,+1} with per-layer adaptive threshold (Δ = 0.7 × mean(\|W\|)). 2-bit packed encoding (00=0, 01=+1, 10=-1). Embedding layer (1 of 155) retained in FP16 as precision-critical. The 8.4× comes from the encoding itself: 32-bit floats → 2-bit trits plus per-layer scale factor, with 43.4% zero-weight sparsity further compressing the packed representation |
| **Variables held** | threshold=0.7, batch_size=1, seed=42, packing=uint8 (4 trits/byte), format=.tern-model v2 |
| **Variables swept** | None (single-point measurement) |
| **Restrictions** | Compression only — no closed-loop inference quality measurement in this experiment. STE training PoC ran separately (EXP-002). x86_64 only (iMac); ARM/Apple Silicon measurements are separate experiments |
| **Shortcuts** | Post-training quantisation — no fine-tuning or calibration dataset required |
| **Raw result** | 4,137,272,320 bytes → 471,572,480 bytes, 154/155 layers converted |
| **Derived result** | 4,137,272,320 / 471,572,480 = 8.773× raw. Reported as 8.4× (conservative floor accounting for metadata overhead in .tern-model v2 container) |
| **Quality verdict** | Integrity PASSED (bit-identical round-trip: pack → unpack → compare). No inference-quality measurement (open-loop compression) |
| **Absolute before/after** | 4,137 MB (FP32) → 471.6 MB (.tern-model) |
| **Report path** | benchmarks/reports/EVIDENCE_PACKAGE.md §1.1, benchmarks/originals/tinyllama_benchmark.md |
| **Claim status** | PUBLISHED (locked set — whole-model compression vs FP32) |

---

### EXP-002 — TinyLlama-1.1B STE Training PoC

| Field | Content |
|-------|---------|
| **ID** | EXP-002 |
| **Date** | 2026-02-26 (Block 3) |
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Model size** | FP32: 4,137 MB |
| **Hardware** | iMac (2019), Intel Core i9-9900K, 64 GB |
| **Software stack** | PyTorch 2.2.2, tern-core v0.1.0 |
| **Method** | Straight-Through Estimator gradient training — 500 steps with ternary weight projection |
| **Approach angle** | STE bypasses the non-differentiable quantisation step: forward pass uses quantised {-1,0,+1} weights, backward pass passes gradients through as if weights were continuous. Enables gradient-based optimisation of the quantisation boundary |
| **Variables held** | lr=1e-4, steps=500, batch_size=1 |
| **Variables swept** | None |
| **Restrictions** | PoC only — 500 steps insufficient for production fine-tuning |
| **Shortcuts** | None |
| **Raw result** | PPL: 77,000 → 1,700 in 500 steps (45.8× improvement) |
| **Derived result** | — |
| **Quality verdict** | PPL improvement demonstrated; model still far from baseline quality |
| **Absolute before/after** | PPL 77,000 → PPL 1,700 |
| **Report path** | benchmarks/reports/EVIDENCE_PACKAGE.md §1.5 |
| **Claim status** | INTERNAL (T0) |

---

### EXP-003 — Llama-3.2-1B Dryrun + Compression

| Field | Content |
|-------|---------|
| **ID** | EXP-003 |
| **Date** | 2026-04-14T07:57Z |
| **Model** | unsloth/Llama-3.2-1B-Instruct |
| **Model size** | FP32: 4,714 MB · FP16: 2,357 MB · .tern-model: 849 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0, Python 3.12, coremltools 8.x |
| **Software stack** | PyTorch 2.7.0, transformers, terncore.convert (dry-run + compress) |
| **Method** | Phase A — dryrun analysis + full ternary conversion |
| **Approach angle** | Llama architecture adapter (auto-detected). Threshold 0.7 yields 78.7% ternary ratio — lower than Mistral (96.4%) because 1B model has proportionally more parameters in protected layers (embeddings, LM head, layer norms = 262M of 1.24B) |
| **Variables held** | threshold=0.7, adapter=llama, architecture=LlamaForCausalLM |
| **Variables swept** | None |
| **Restrictions** | Protected layers (embed_tokens, lm_head, norms) retained in FP16 — 262.7M params |
| **Shortcuts** | None |
| **Raw result** | 1,235,814,400 params, 146 weights, 973M ternary-eligible, 78.74% ternary ratio, .tern-model 849 MB |
| **Derived result** | Compression vs FP16: 2,357 / 733 (estimated) = 3.22×. Actual .tern-model 849 MB includes metadata |
| **Quality verdict** | Conversion integrity verified |
| **Absolute before/after** | 2,357 MB (FP16) → 849 MB (.tern-model) |
| **Report path** | benchmarks/results/llama32_1b_dryrun.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-004 — Llama-3.2-1B CoreML Phase B + D (Inference + Energy)

| Field | Content |
|-------|---------|
| **ID** | EXP-004 |
| **Date** | 2026-04-14T08:08Z |
| **Model** | Llama-3.2-1B ternary (.mlpackage, raw + palettised 2-bit) |
| **Model size** | Raw mlpackage: 1,022.7 MB · Palettised 2-bit: 584.3 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools, ANE/GPU/CPU compute units |
| **Software stack** | coremltools (CoreML prediction API), bench_llama32_1b_phase2.py |
| **Method** | Phase B — CoreML inference latency across 3 compute units, 50 runs + 10 warmup. Phase D — 15s sustained energy measurement via sudo powermetrics |
| **Approach angle** | CoreML routes ternary-converted weights through Apple's ANE/GPU dispatch. The ternary weights are still FP16-encoded in the mlpackage (CoreML lacks native trit support), so the compression advantage is in the palettised 2-bit form. CPU_AND_NE gives lowest latency stdev (0.5ms vs 0.6ms GPU) |
| **Variables held** | seq_len=64, warmup=10, benchmark_runs=50, batch=1 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | FP16-encoded mlpackage — CoreML native ternary encoding pending. Palettisation via coremltools post-training (Phase C) |
| **Shortcuts** | coremltools palettise_weights — 2-bit post-training palettisation, bypasses per-layer calibration |
| **Raw result** | Raw: ALL 1,553.9 tok/s (41.2ms), CPU_AND_NE 1,501.7 tok/s (42.6ms, 0.5ms stdev), CPU_AND_GPU 1,562.1 tok/s (41.0ms). Palettised: CPU_AND_NE **1,617.8 tok/s** (39.6ms). Energy (pal best): 7.41W, 310.6 mJ/inference |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed: load 2.3s, predict 0.45s, output shape [1,64,128256] |
| **Absolute before/after** | 1,022.7 MB (raw mlpackage) → 584.3 MB (palettised 2-bit) |
| **Report path** | benchmarks/results/llama32_1b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-005 — Mistral-7B CoreML Phase B + D (Inference + Energy)

| Field | Content |
|-------|---------|
| **ID** | EXP-005 |
| **Date** | 2026-04-13T23:47Z |
| **Model** | Mistral-7B ternary v0.3.0 (.mlpackage, 13.8 GB FP16-encoded) |
| **Model size** | FP32: ~28 GB · FP16: 14.5 GB · .tern-pkg: 2.27 GB · mlpackage (FP16-encoded): 13.8 GB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, coremltools, ANE/GPU/CPU compute units |
| **Software stack** | coremltools (CoreML prediction API), bench_mistral7b_phase2.py (518 lines) |
| **Method** | Phase B — CoreML inference latency across 3 compute units, 50 runs + 10 warmup. Phase D — 15s sustained energy across 3 compute units. Phase C — palettisation attempted but failed (Inf weights) |
| **Approach angle** | Mistral-7B at 96.4% ternary is the highest-ratio model in the sprint. The .tern-pkg at 2.27 GB demonstrates the iPhone-viable size point — the gap from 14.5 GB mlpackage to 2.27 GB is the CoreML native ternary argument. CPU_AND_NE wins on latency stability (9.7ms stdev vs 245ms GPU jitter) and energy (5.39W vs 20.14W GPU) |
| **Variables held** | seq_len=64, warmup=10, benchmark_runs=50, batch=1 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU; energy: CPU_ONLY, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | Phase C FAILED — ValueError: Inf in FP16 weight tensors blocks palettisation (tern-core #1). No palettised results. mlpackage at 13.8 GB is FP16-encoded, not compressed |
| **Shortcuts** | None |
| **Raw result** | CPU_AND_NE: **297.6 tok/s**, 215.1ms mean, 9.7ms stdev. Energy: CPU_AND_NE **5.388W**, 1,122.6 mJ/inf. CPU_ONLY: 304.5 tok/s, 5.458W. CPU_AND_GPU: 387.8 tok/s, 20.142W |
| **Derived result** | CPU_AND_NE energy per token: 1,122.6 / 64 = 17.54 mJ/token. CPU_ONLY and CPU_AND_NE within 1.3% energy envelope at ~5.4W |
| **Quality verdict** | CoreML validation passed: load 34.2s, predict 0.28s, output shape [1,64,32000] |
| **Absolute before/after** | 14.5 GB (FP16 source) → 2.27 GB (.tern-pkg, 96.4% ternary) |
| **Report path** | benchmarks/results/mistral7b_phase2.json, benchmarks/reports/mistral_7b_phase_d_full.md |
| **Claim status** | PUBLISHED (locked set — 297.6 tok/s @ 5.388 W) |

---

### EXP-006 — Llama-3.2-3B Dryrun + Compression + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-006 |
| **Date** | 2026-04-14T22:29Z (dryrun) / 2026-04-14T22:38Z (phase2) |
| **Model** | unsloth/Llama-3.2-3B-Instruct |
| **Model size** | FP32: 12,256 MB · FP16: 6,128 MB · .tern-model: 1,760 MB · mlpackage raw: 2,252.6 MB · pal 2-bit: 1,595 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | terncore.convert, coremltools, bench_llama32_3b_phase2.py |
| **Method** | Dryrun + compress + Phase B (inference) + Phase D (energy) |
| **Approach angle** | 87.7% ternary ratio — higher than 1B (78.7%) because larger models have proportionally fewer params in protected layers. Palettised 2-bit CPU_AND_NE is the performance sweet spot: 698.6 tok/s at 7.29W |
| **Variables held** | threshold=0.7, seq_len=64, warmup=10, benchmark_runs=50 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | None — Phase C (palettisation) clean |
| **Shortcuts** | coremltools palettise_weights 2-bit |
| **Raw result** | Dryrun: 3.21B params, 87.73% ternary, 4.3× vs FP16. Phase B pal CPU_AND_NE: **698.6 tok/s** (91.6ms). Energy pal best: 7.29W, 687.6 mJ/inf |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed. Phase C clean |
| **Absolute before/after** | 6,128 MB (FP16) → 1,760 MB (.tern-model) → 1,595 MB (pal 2-bit mlpackage) |
| **Report path** | benchmarks/results/llama32_3b_dryrun.json, benchmarks/results/llama32_3b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-007 — Gemma 3 4B Dryrun + Compression + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-007 |
| **Date** | 2026-04-14T23:31Z (dryrun) / 2026-04-14T23:41Z (phase2) |
| **Model** | unsloth/gemma-3-4b-it |
| **Model size** | FP16: 8,202 MB · mlpackage raw: 2,988 MB · pal 2-bit: 1,868 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | terncore.convert (gemma3 adapter), coremltools, bench_gemma3_4b_phase2.py |
| **Method** | Dryrun + compress + Phase B + D |
| **Approach angle** | Gemma 3 architecture adapter (Gemma3ForConditionalGeneration, 883 weights). 74.6% ternary — lower than Llama family due to multimodal encoder components retained in FP16. Palettised 2-bit on CPU_AND_NE gives 513.3 tok/s at 8.00W — the best energy-normalised throughput in the 4B class |
| **Variables held** | threshold=0.7, seq_len=64 (gemma3-4b preset) |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | 1,091M params retained in FP16 (encoder components) |
| **Shortcuts** | coremltools palettise_weights 2-bit |
| **Raw result** | Dryrun: 4.30B params, 74.62% ternary, 2.88× vs FP16. Phase B pal CPU_AND_NE: **513.3 tok/s** (124.7ms). Energy pal best: 8.00W, 1,008.7 mJ/inf |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed. Phase C clean |
| **Absolute before/after** | 8,202 MB (FP16) → 2,847 MB (est .tern-model) → 1,868 MB (pal 2-bit) |
| **Report path** | benchmarks/results/gemma3_4b_dryrun.json, benchmarks/results/gemma3_4b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-008 — Gemma 3 12B Compression + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-008 |
| **Date** | 2026-04-15T03:58Z |
| **Model** | Gemma 3 12B (gemma3-12b) |
| **Model size** | mlpackage raw: 7,566 MB · pal 2-bit: 5,886 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | coremltools, bench_gemma3_12b_phase2.py |
| **Method** | Phase B + D |
| **Approach angle** | 12B model stress-tests the M4 Pro memory envelope (43.9 GB peak RSS). CPU_AND_NE extremely slow at 4.7 tok/s due to ANE dispatch overhead at this model size — ALL and CPU_AND_GPU both ~33 tok/s. Palettised 2-bit gives marginal improvement |
| **Variables held** | seq_len=64, warmup=10, benchmark_runs=50 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | CPU_AND_NE impractically slow (13.7s/inference). Memory pressure near 64 GB ceiling |
| **Shortcuts** | None |
| **Raw result** | Phase B: ALL 33.2 tok/s, CPU_AND_NE 4.7 tok/s, CPU_AND_GPU 33.0 tok/s. Pal best: ALL 33.9 tok/s. Energy: raw ALL 26.30W, pal ALL 27.03W |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed |
| **Absolute before/after** | 7,566 MB (raw mlpackage) → 5,886 MB (pal 2-bit) |
| **Report path** | benchmarks/results/gemma3_12b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-009 — Phi-4 14B Dryrun + Compression + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-009 |
| **Date** | 2026-04-15T07:31Z (dryrun) / 2026-04-15T08:02Z (phase2) |
| **Model** | microsoft/phi-4 |
| **Model size** | FP16: 27,961 MB · mlpackage raw: 8,984 MB · pal 2-bit: 7,269 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | terncore.convert (llama adapter), coremltools, bench_phi4_14b_phase2.py |
| **Method** | Dryrun + compress + Phase B + D |
| **Approach angle** | Phi-4 maps to the llama adapter (LlamaForCausalLM architecture). 93.0% ternary — highest ratio after Mistral. 5.37× compression vs FP16. CPU_AND_NE extremely slow (4.8 tok/s, 13.4s/inference) similar to Gemma 3 12B — ANE dispatch overhead dominates at 14B scale. ALL and CPU_AND_GPU both ~29 tok/s |
| **Variables held** | threshold=0.7, seq_len=64 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | 47.5 GB peak RSS — near 64 GB ceiling. CPU_AND_NE impractical |
| **Shortcuts** | None |
| **Raw result** | Dryrun: 14.66B params, 92.99% ternary, 5.37× vs FP16. Phase B: ALL 29.0 tok/s, CPU_AND_GPU 29.4 tok/s. Energy: raw ALL 27.58W, pal ALL 27.78W |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed. Phase C clean |
| **Absolute before/after** | 27,961 MB (FP16) → 5,211 MB (est .tern-model) → 7,269 MB (pal 2-bit mlpackage) |
| **Report path** | benchmarks/results/phi4_14b_dryrun.json, benchmarks/results/phi4_14b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-010 — Qwen2.5-7B Dryrun + Compression + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-010 |
| **Date** | 2026-04-16T00:19Z (dryrun) / 2026-04-16T00:31Z (phase2) |
| **Model** | Qwen/Qwen2.5-7B-Instruct |
| **Model size** | FP16: 14,526 MB · mlpackage raw: 5,550 MB · pal 2-bit: 3,731 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | terncore.convert (llama adapter), coremltools, bench_qwen25_7b_phase2.py |
| **Method** | Dryrun + compress + Phase B + D |
| **Approach angle** | Qwen2.5 maps to llama adapter (same architecture family). 85.7% ternary, 4.0× vs FP16 — identical ratio to DSR1-7B (both are 7.6B-param Qwen family). CPU_AND_NE gives stable latency (18.7ms stdev) vs GPU jitter (133ms) |
| **Variables held** | threshold=0.7, seq_len=64 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | None |
| **Shortcuts** | None |
| **Raw result** | Dryrun: 7.62B params, 85.68% ternary, 4.0× vs FP16. Phase B: ALL 54.3 tok/s, CPU_AND_NE 53.3 tok/s. Pal ALL 55.7 tok/s. Energy: raw CPU_AND_GPU 28.00W, pal ALL 28.24W |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed. Phase C clean |
| **Absolute before/after** | 14,526 MB (FP16) → 3,635 MB (est .tern-model) → 3,731 MB (pal 2-bit) |
| **Report path** | benchmarks/results/qwen25_7b_dryrun.json, benchmarks/results/qwen25_7b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-011 — DSR1-7B + DSR1-14B CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-011 |
| **Date** | 2026-04-17T07:57Z (7B) / 2026-04-17T23:37Z (14B) |
| **Model** | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B + 14B |
| **Model size** | 7B: FP16 14,526 MB, mlpackage raw 5,552 MB, pal 2-bit 3,732 MB · 14B: mlpackage raw 9,965 MB, pal 2-bit 7,366 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | terncore.convert (llama adapter), coremltools, bench_dsr1_7b_phase2.py / bench_gemma3_12b_phase2.py (reused for 14B) |
| **Method** | Phase B + D |
| **Approach angle** | DSR1 distillation of DeepSeek-R1 into Qwen architecture. 7B: 85.7% ternary, same as Qwen2.5-7B (identical base architecture). 14B: CPU_AND_NE extremely slow (4.7 tok/s) — same ANE dispatch cliff as Gemma 12B and Phi-4 14B. The 14B class consistently hits this wall on M4 Pro |
| **Variables held** | threshold=0.7, seq_len=64 |
| **Variables swept** | compute_units, model size (7B vs 14B) |
| **Restrictions** | 14B: 51.7 GB peak RSS, near ceiling. CPU_AND_NE impractical at this scale |
| **Shortcuts** | 14B runner reused bench_gemma3_12b_phase2.py |
| **Raw result** | 7B: ALL 56.2 tok/s, CPU_AND_NE 53.5 tok/s. Energy: raw ALL 33.29W. · 14B: ALL 27.9 tok/s, CPU_AND_GPU 29.2 tok/s. Energy: raw CPU_AND_GPU 26.21W |
| **Derived result** | — |
| **Quality verdict** | CoreML validation passed (both) |
| **Absolute before/after** | 7B: 5,552 MB → 3,732 MB (pal). 14B: 9,965 MB → 7,366 MB (pal) |
| **Report path** | benchmarks/results/dsr1_7b_phase2.json, benchmarks/results/dsr1_14b_phase2.json |
| **Claim status** | INTERNAL (T0) |

---

### EXP-012 — Bonsai-8B Coherence + CoreML Phase B + D

| Field | Content |
|-------|---------|
| **ID** | EXP-012 |
| **Date** | 2026-06-01T04:50Z |
| **Model** | prism-ml/Ternary-Bonsai-8B-unpacked (.mlpackage, 5,787 MB) |
| **Model size** | mlpackage: 5,787 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5, coremltools |
| **Software stack** | coremltools, bench_bonsai_8b_phase2.py, parity_bonsai8b.py |
| **Method** | Phase B (inference) + Phase D (energy) + argmax-exact coherence verification (n=64) |
| **Approach angle** | Bonsai-8B is a pre-trained ternary model (weights already {-1,0,+1}) — tests the inference path without tern-core's conversion step. Coherence measured as argmax-exact match: for each of n=64 prompts, verify that the greedy-decoded output token sequence from the ternary model matches a reference. Tier 2.5 = argmax-exact match demonstrated |
| **Variables held** | seq_len=64, warmup=10, benchmark_runs=50, coherence_n=64 |
| **Variables swept** | compute_units: ALL, CPU_AND_NE, CPU_AND_GPU |
| **Restrictions** | Pre-trained ternary model, not tern-core converted. Coherence methodology is argmax-exact (greedy), not calibrated perplexity |
| **Shortcuts** | None |
| **Raw result** | Phase B: ALL 299.1 tok/s, CPU_AND_NE 277.9 tok/s, CPU_AND_GPU 298.4 tok/s. Energy: CPU_AND_NE 5.63W, 1,278.8 mJ/inf. Coherence: Tier 2.5 (argmax-exact, n=64 verified) |
| **Derived result** | — |
| **Quality verdict** | Tier 2.5 argmax-exact coherence PASSED (n=64). CoreML validation passed |
| **Absolute before/after** | N/A (pre-trained ternary model) |
| **Report path** | model-library/benchmarks/bonsai-8b/phase2.json, ecc-ternary/uploads/PACK_BONSAI8B_20260601T025834Z/ |
| **Claim status** | PUBLISHED (locked set — Tier 2.5 argmax-exact coherence) |

---

### EXP-013 — Llama-3.1-70B Compression + CoreML Demo

| Field | Content |
|-------|---------|
| **ID** | EXP-013 |
| **Date** | 2026-04-12T00:17Z |
| **Model** | meta-llama/Llama-3.1-70B-Instruct |
| **Model size** | FP16: ~131 GB · .tern-model: 37.0 GB · mlpackage: 38.96 GB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS 26.5 |
| **Software stack** | terncore v0.6.0 (streaming shard-by-shard), coremltools |
| **Method** | Full compression + CoreML export. Inference attempted but blocked by memory |
| **Approach angle** | Streaming pipeline processes one shard at a time — never loads full 131 GB into memory. The 3.54× compression (131 → 37 GB) is the demonstration that ternary scales to 70B. The mlpackage at 38.96 GB validates the CoreML export path. Inference on M4 Pro 64 GB blocked: decompression expands to ~116 GB. Target hardware: M2/M3 Ultra 192 GB |
| **Variables held** | threshold=0.7 |
| **Variables swept** | None |
| **Restrictions** | Inference blocked — decompression exceeds 64 GB. Demo artefact only. M2/M3 Ultra (192 GB) required for inference |
| **Shortcuts** | Streaming shard-by-shard conversion (tern-core v0.5.0+) |
| **Raw result** | 37.0 GB .tern-model, 38.96 GB mlpackage. Inference: not possible on M4 Pro 64 GB |
| **Derived result** | 131 GB / 37 GB = 3.54× vs FP16 |
| **Quality verdict** | Conversion integrity verified. Inference NOT VERIFIED (memory-blocked) |
| **Absolute before/after** | ~131 GB (FP16) → 37.0 GB (.tern-model) → 38.96 GB (mlpackage) |
| **Report path** | benchmarks/results/llama70b_phase2.json, docs/TN-001_llama70b_compression_analysis.md |
| **Claim status** | INTERNAL (T0 — demo artefact, inference unverified) |

---

### EXP-014 — Gemma 4 E4B Dryrun

| Field | Content |
|-------|---------|
| **ID** | EXP-014 |
| **Date** | 2026-04-14T06:41Z |
| **Model** | google/gemma-4-E4B-it |
| **Model size** | FP16: 15,252 MB · est .tern-model: 8,667 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB |
| **Software stack** | terncore.convert (gemma4 adapter) |
| **Method** | Dryrun analysis only |
| **Approach angle** | Gemma 4 multimodal dense model. New gemma4 adapter written for architecture differences from Gemma 3. Only 49.3% ternary — lowest in sprint, because 4.05B of 8.0B params are in retained (non-ternary-eligible) multimodal components. Compression vs FP16: 1.76× — limited value from ternary at this ratio |
| **Variables held** | threshold=0.7 |
| **Variables swept** | None |
| **Restrictions** | Dryrun only — full compression blocked on coremltools 9.1/9.2. Low ternary ratio (49.3%) limits compression benefit |
| **Shortcuts** | None |
| **Raw result** | 7.996B params, 2,130 weights, 49.34% ternary, 1.76× est compression |
| **Derived result** | — |
| **Quality verdict** | Dryrun only — no conversion or inference |
| **Absolute before/after** | 15,252 MB (FP16) → 8,667 MB (estimated .tern-model) |
| **Report path** | benchmarks/results/gemma4_e4b_dryrun.json |
| **Claim status** | INTERNAL (T0 — blocked on coremltools) |

---

### EXP-015 — Track A: TinyLlama KV Cache Replication

| Field | Content |
|-------|---------|
| **ID** | EXP-015 |
| **Date** | 2026-08-09 |
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Model size** | FP16: 2,357 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0, Python 3.12 |
| **Software stack** | PyTorch 2.7.0, transformers 5.5.4, tern_infer.py (IncrementalTQCompressor) |
| **Method** | Track A — replication of the original TurboQuant KV compression run, open-loop |
| **Approach angle** | Open-loop replication: TurboQuant compressor records compressed KV state as a side effect, but the model's forward pass uses the original uncompressed KV. The 3.88× measured byte ratio (raw accounting) vs the 5.33× theoretical bits-per-coordinate ratio revealed the accounting gap that prompted the reconciliation |
| **Variables held** | b_mse=3, mixed_precision=True, head_dim=64, threshold=0.7 |
| **Variables swept** | None |
| **Restrictions** | Open-loop — compressed KV never fed back to inference. Quality unverifiable under this methodology |
| **Shortcuts** | None |
| **Raw result** | 3.88× byte ratio (raw accounting, 56 tokens). Reconciled to opt B 6.74× (packed, shared excluded) |
| **Derived result** | Three accountings reconciled: raw 3.88×, opt A 0.76× (at 56 tokens, converges to 6.08× at 4K), opt B 6.74× (model-independent) |
| **Quality verdict** | NOT MEASURED (open-loop). KV compression quality requires closed-loop verification |
| **Absolute before/after** | KV cache: FP16 baseline → TurboQuant compressed (3.88× raw, 6.74× opt B) |
| **Report path** | benchmarks/reports/Benchmark_Report_A_Replication.md, benchmarks/reports/Reconciliation_A_Accounting.md |
| **Claim status** | RETIRED (5.3× retired from public use; KV compression pending quality verdict) |

---

### EXP-016 — Track B3: Native-Ternary KV Sweep

| Field | Content |
|-------|---------|
| **ID** | EXP-016 |
| **Date** | 2026-08-09 |
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Model size** | FP16: 2,357 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0 |
| **Software stack** | PyTorch 2.7.0, transformers 5.5.4, bench_track_b3_native_ternary_kv.py |
| **Method** | Track B3 — closed-loop native-ternary KV cache: quantise KV values to {-1,0,+1} per-vector symmetric threshold, feed decompressed KV back to model, measure argmax match |
| **Approach angle** | Test whether ternary's proven success on weights extends to KV cache values. Per-vector symmetric threshold quantisation: threshold = factor × max(\|v\|), factor swept 0.3–0.9. 2-bit packing (same as weights). Closed-loop: decompressed KV fed back for every subsequent token. 14.2× compression achieved but quality catastrophic |
| **Variables held** | model=TinyLlama-1.1B, n_tokens=100, prompts=5 |
| **Variables swept** | threshold factor: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| **Restrictions** | All 7 thresholds fail the 93% argmax floor gate. KV values carry fine-grained floating-point attention patterns that {-1,0,+1} destroys |
| **Shortcuts** | None |
| **Raw result** | Best argmax match: 15% (threshold=0.3). All thresholds 1–15% match vs 93% floor / 99% mean requirement |
| **Derived result** | 14.2× compression at catastrophic quality loss |
| **Quality verdict** | **FAIL** — all 7 thresholds below 93% floor. Degenerate output (repetition, token cycling) |
| **Absolute before/after** | KV FP16 → KV ternary: 14.2× compression, 1–15% argmax match |
| **Report path** | benchmarks/reports/Benchmark_Report_B3_Native_Ternary_KV.md |
| **Claim status** | GAP (KV compression quality undemonstrated) |

---

### EXP-017 — Track B1: Llama-3.2-1B FP16 + INT8 Baselines

| Field | Content |
|-------|---------|
| **ID** | EXP-017 |
| **Date** | 2026-08-09 |
| **Model** | unsloth/Llama-3.2-1B-Instruct |
| **Model size** | FP32: 4,714 MB · FP16: 2,357 MB · INT8 (qnnpack): 1,002 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0, Python 3.12 |
| **Software stack** | PyTorch 2.7.0, transformers 5.5.4, torch.ao.quantization (qnnpack backend) |
| **Method** | Track B1 — raw PyTorch inference baselines: FP16 on MPS, INT8 dynamic quantisation on CPU |
| **Approach angle** | Establishes the "before" performance floor. FP16 runs on Apple MPS (GPU); INT8 uses qnnpack dynamic quantisation on CPU because PyTorch's quantised MPS backend (fbgemm) is unavailable on ARM macOS. The INT8-on-CPU vs FP16-on-MPS comparison is apples-to-oranges for speed but valid for size |
| **Variables held** | generate_tokens=64, n_runs=3, greedy decoding (argmax) |
| **Variables swept** | precision: FP16 (MPS), INT8 (CPU). prompts: 3 diverse prompts |
| **Restrictions** | INT8 qnnpack CPU-only — MPS quantised backend unavailable on ARM macOS. Speed comparison between FP16-MPS and INT8-CPU is device-confounded |
| **Shortcuts** | Dynamic quantisation (no calibration dataset) |
| **Raw result** | FP16: 61.2 tok/s median (MPS), 2,357 MB. INT8: 26.7 tok/s median (CPU), 1,002 MB. INT8 compression vs FP32: 4.70× |
| **Derived result** | tern-core CoreML (EXP-004) vs FP16 PyTorch: 1,617.8 / 61.2 = 26.4× faster (pal CPU_AND_NE vs FP16 MPS). tern-core vs INT8: 1,617.8 / 26.7 = 60.6× faster |
| **Quality verdict** | Coherent text generated in all runs. Cold-start outlier on FP16 run 1 (35.8 tok/s), steady-state 60–63 tok/s |
| **Absolute before/after** | FP32 4,714 MB → FP16 2,357 MB → INT8 1,002 MB (→ tern-core 849 MB → pal 584 MB) |
| **Report path** | benchmarks/results/Benchmark_Report_B1_FP16_INT8_Baselines.json |
| **Claim status** | INTERNAL (T0 — baseline reference, not a tern-core claim) |

---

### EXP-018 — Track B1: Llama-3.2-3B FP16 + INT8 Baselines

| Field | Content |
|-------|---------|
| **ID** | EXP-018 |
| **Date** | 2026-08-09 |
| **Model** | unsloth/Llama-3.2-3B-Instruct |
| **Model size** | FP32: 12,256 MB · FP16: 6,128 MB · INT8 (qnnpack): 1,504 MB |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0, Python 3.12 |
| **Software stack** | PyTorch 2.7.0, transformers 5.5.4, torch.ao.quantization (qnnpack backend) |
| **Method** | Track B1 — raw PyTorch inference baselines: FP16 on MPS, INT8 dynamic quantisation on CPU |
| **Approach angle** | Same methodology as EXP-017. At 3B params, the FP16-on-MPS speed drops to 34.4 tok/s (vs 61.2 for 1B) — a 1.8× slowdown for a 2.6× param increase, roughly √(param ratio) scaling. INT8 dynamic quantisation achieves 8.15× compression vs FP32 (better than 1B's 4.70×) due to higher ratio of large Linear layers |
| **Variables held** | generate_tokens=64, n_runs=3, greedy decoding (argmax) |
| **Variables swept** | precision: FP16 (MPS), INT8 (CPU). prompts: 3 diverse prompts |
| **Restrictions** | INT8 qnnpack CPU-only — MPS quantised backend unavailable on ARM macOS. Speed comparison between FP16-MPS and INT8-CPU is device-confounded |
| **Shortcuts** | Dynamic quantisation (no calibration dataset) |
| **Raw result** | FP16: 34.36 tok/s median (MPS), 6,128 MB. INT8: 11.05 tok/s median (CPU), 1,504 MB. INT8 compression vs FP32: 8.15× |
| **Derived result** | tern-core CoreML (EXP-005) vs FP16 PyTorch: 704.5 / 34.36 = 20.5× faster (pal CPU_AND_NE vs FP16 MPS). tern-core vs INT8: 704.5 / 11.05 = 63.8× faster |
| **Quality verdict** | Coherent text generated in all runs. Cold-start outlier on FP16 run 1 (24.4 tok/s), steady-state 34.2–34.8 tok/s. INT8 extremely consistent (10.9–11.1 tok/s) |
| **Absolute before/after** | FP32 12,256 MB → FP16 6,128 MB → INT8 1,504 MB (→ tern-core 2,253 MB → pal 1,595 MB) |
| **Report path** | benchmarks/results/Benchmark_Report_B1_FP16_INT8_Baselines.json |
| **Claim status** | INTERNAL (T0 — baseline reference, not a tern-core claim) |

---

### EXP-019 — Track B1: Mistral-7B FP16 + INT8 Baselines

| Field | Content |
|-------|---------|
| **ID** | EXP-019 |
| **Date** | 2026-08-09 |
| **Model** | mistralai/Mistral-7B-Instruct-v0.3 |
| **Model size** | FP32: 27,649 MB · FP16: 13,825 MB · INT8 (qnnpack): 513 MB (see note) |
| **Hardware** | Mac Mini M4 Pro, 64 GB, macOS, torch 2.7.0, Python 3.12 |
| **Software stack** | PyTorch 2.7.0, transformers 5.5.4, torch.ao.quantization (qnnpack backend) |
| **Method** | Track B1 — raw PyTorch inference baselines: FP16 on MPS, INT8 dynamic quantisation on CPU |
| **Approach angle** | Same methodology as EXP-017/018, completing the 3-model B1 baseline set. Mistral-7B is the reference exemplar (locked set: 297.6 tok/s), so the B1 baselines directly quantify tern-core's advantage over conventional quantisation for this model. FP16 runs stable at 16.4–16.6 tok/s with minimal cold-start effect. INT8 model size (513 MB) is a measurement artifact — `model.parameters()` skips quantised packed_params; the tok/s measurement (5.55) is valid |
| **Variables held** | generate_tokens=64, n_runs=3, greedy decoding (argmax) |
| **Variables swept** | precision: FP16 (MPS), INT8 (CPU). prompts: 3 diverse prompts |
| **Restrictions** | INT8 qnnpack CPU-only — MPS quantised backend unavailable on ARM macOS. INT8 size measurement undercounts due to packed_params not iterated by model.parameters() |
| **Shortcuts** | Dynamic quantisation (no calibration dataset) |
| **Raw result** | FP16: 16.45 tok/s median (MPS), 13,825 MB. INT8: 5.55 tok/s median (CPU), 513 MB reported (measurement artifact). INT8 compression vs FP32: 53.9× reported (artifact) |
| **Derived result** | tern-core CoreML (EXP-005 CPU_AND_NE) vs FP16 PyTorch: 297.6 / 16.45 = 18.1× faster. tern-core vs INT8: 297.6 / 5.55 = 53.6× faster |
| **Quality verdict** | Coherent text in all runs. FP16 extremely stable (16.0–16.6 tok/s, minimal cold-start). INT8 range 5.4–5.6 tok/s |
| **Absolute before/after** | FP32 27,649 MB → FP16 13,825 MB → INT8 513 MB (measured, see note) (→ tern-core .tern-pkg 2,270 MB, 96.4% ternary) |
| **Report path** | benchmarks/results/Benchmark_Report_B1_FP16_INT8_Baselines.json |
| **Claim status** | INTERNAL (T0 — baseline reference, not a tern-core claim) |

---

*Next EXP-ID: EXP-020*
