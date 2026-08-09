# BUILD 6/6.1 — Measurement Campaign Status · 9 Aug 2026

> Two-track benchmark campaign feeding deck v4.0's three-factor icons and the NPU partner claim sheet.
> Truth discipline: every number tagged Measured / Derived / Projected.
> BUILD 6.1 extends: B1 download authorized; reconciliation task; Syn Archive recovery; NPU comparison deliverable.

---

## Track A — Replication + Reconciliation: COMPLETE

**Script:** `bench_track_a_replication.py`
**Report:** `Benchmark_Report_A_Replication.md` + `.json`
**Reconciliation:** `Reconciliation_A_Accounting.md` + `.json`

### Finding

The published 5.3× KV compression ratio is the **theoretical bits-per-coordinate** ratio (16 bits FP16 / 3 bits b_mse = 5.33×). The **measured byte ratio** with actual TurboQuant encoding (including norms, codebook indices, rotation matrices, QJL state) is **3.88×**. Neither is the right metric.

### Reconciliation (BUILD 6.1 §2)

The recovered Llama-3.2-1B bench (13 May 2026) provides the bits_per_slot formula: `d × (b_mse + 1) / 8 + 6 = 38 bytes/slot`. Three accountings now reconciled:

| Accounting | TinyLlama (56 tok) | Llama-3.2-1B (106 tok) | At serving scale (4K+ tok) |
|---|---:|---:|---:|
| **Opt B** (packed, shared excluded) | 6.74× | 6.74× | 6.74× |
| **Opt A** (packed, all bytes) | 0.76× (expansion) | 2.20× | 6.08× (converges on B) |
| Track A raw (`_count_object_bytes`) | 3.88× | 0.39× (expansion) | implementation-specific |
| Published 5.33× | — | — | FP16 ref / no overhead |

**Track A's 3.88× is an implementation artifact** — neither opt A nor opt B. It measures how the Python compressor stores tensors in RAM, which varies by model and measurement method.

**Recommendation:** Claim opt B (6.7× vs float32 KV) as the serving-scale figure, opt A at stated seq_len as the single-session figure. Retire 5.33× (omits real overhead, wrong reference frame) and 3.88× (implementation artifact). Both carry **no quality verdict** — see B3.

### Additional findings
- TinyLlama uses GQA: 4 KV heads (not 32 attention heads)
- Original PPL (7.82) used teacher-forcing on 5 sentences; different from WikiText-2 evaluation (5.54)
- Compression is deterministic across runs (3.88× on all 3 runs)
- Opt B is model-independent when d and b_mse match — depends only on encoding formula

---

## Track B2 — FP16-KV Baseline: COMPLETE

**Script:** `bench_track_b2_fp16_kv_baseline.py`
**Report:** `Benchmark_Report_B2_FP16_KV_Baseline.md` + `.json`

The "1×" rung is now MEASURED:
- 45,056 bytes per token (float32 KV cache, TinyLlama)
- 2,048 bytes per token per layer (22 layers × 4 KV heads × 64 dims × 2 K+V × 4 bytes)
- Linear scaling confirmed across 64–512 token sequences
- 27.8 tok/s decode throughput on CPU

---

## Track B3 — Native-Ternary KV: COMPLETE — QUALITY FAIL

**Script:** `bench_track_b3_native_ternary_kv.py`
**Report:** `Benchmark_Report_B3_Native_Ternary_KV.md` + `.json`

Native ternary KV ({-1,0,+1} per-vector symmetric threshold, 2-bit packed + FP16 scale) tested closed-loop across 7 thresholds × 5 prompts × 100 tokens.

### Result

| Threshold | Mean argmax match | Min match | Compression | Zero % | Verdict |
|---:|---:|---:|---:|---:|---|
| 0.3 | 4.2% | 1.0% | 14.2× | 22% | FLOOR_FAIL |
| 0.4 | 3.2% | 1.0% | 14.2× | 28% | FLOOR_FAIL |
| 0.5 | 4.4% | 1.0% | 14.2× | 35% | FLOOR_FAIL |
| 0.6 | 6.0% | 1.0% | 14.2× | 41% | FLOOR_FAIL |
| 0.7 | 5.0% | 2.0% | 14.2× | 47% | FLOOR_FAIL |
| 0.8 | 3.2% | 2.0% | 14.2× | 52% | FLOOR_FAIL |
| 0.9 | 3.4% | 1.0% | 14.2× | 57% | FLOOR_FAIL |

**No threshold passes either gate** (93% floor, 99% bell mean). Best mean: 6.0% at threshold=0.6 — off by 87 percentage points. Compression is excellent (14.2× constant), quality is catastrophic.

### Diagnosis

KV cache values encode attention patterns with fine-grained floating-point structure. Unlike model weights (trained to tolerate quantisation), KV values are dynamic per-token computations where small magnitude differences carry semantic information. Mapping to {-1, 0, +1} destroys directional information that cannot be recovered from a per-vector FP16 scale alone.

The compressed outputs degenerate into repetitive tokens ("and, and, and, the, the, the") regardless of threshold — the model loses the ability to distinguish attention keys/values that were originally separated by small floating-point deltas.

### Implications

- **Native ternary KV at the naive level is ruled out** for quality-gated serving
- The 14.2× compression ratio is real but useless without quality
- TurboQuant closed-loop also fails (R12: NaN at all b_mse values)
- **Every KV compression approach tested so far fails closed-loop quality**
- Icon 1 (KV compression) has ratio measurements but zero quality backing

### Potential recovery paths (for Rob)
1. **Mixed-precision KV**: keep critical heads/layers in FP16, compress only tolerant ones (requires per-head sensitivity analysis)
2. **Learned KV projection**: train a small encoder/decoder that maps KV vectors to a lower-dimensional space (closer to vector quantisation than thresholding)
3. **Grouped quantisation**: INT4/INT2 per-group instead of ternary — higher bits, proven quality in weight domain
4. **Accept lower compression**: 2–4× via straightforward FP16→INT8/INT4 KV quantisation (industry standard, quality demonstrated by others)

---

## Track B1 — FP16/INT8 Baselines: AUTHORIZED, PENDING

**Status:** Download AUTHORIZED per BUILD 6.1. Scope expanded to include Llama-3.2-1B and 3B alongside Mistral-7B.

**Per model:** FP16 + INT8 weight baselines + FP16-KV baseline, same rig, n≥3.

**Models:**
- Mistral-7B-Instruct-v0.3 (~14 GB download)
- Llama-3.2-1B-Instruct (already in HF cache)
- Llama-3.2-3B-Instruct (already in HF cache)

---

## Syn Archive Recovery (BUILD 6.1 §3)

### Inventory: models/compressed/

| Model | Size | Version | Date |
|---|---:|---|---|
| llama32-1b | 849 MB | v0.1.0 | 14 Apr |
| llama32-3b | 1.7 GB | v0.1.0 | 15 Apr |
| llama-3-1-70b | 35 GB | v0.6.0-mixed | 10 Apr |
| gemma3-4b | 3.2 GB | v0.1.0 | 15 Apr |
| gemma3-12b | 6.4 GB | v0.1.0 | 15 Apr |
| gemma4-e4b | 8.9 GB | v0.1.0 | 1 May |
| gemma4-31b | 14 GB | v0.1.0 | 7 May |
| gemma4-26b-a4b | 11 GB | v0.1.0 | 6 May |
| gemopus-4-e4b | 8.9 GB | v0.1.0 | 1 May |
| gemopus-4-26b-a4b | 11 GB | v0.1.0 | 6 May |
| phi4-14b | 6.7 GB | v0.1.0 | 15 Apr |
| phi-4 | 6.7 GB | v0.1.1 | 7 May |
| qwen25-7b | 4.3 GB | v0.1.0 | 16 Apr |
| qwen3-30b-a3b | 12 GB | v0.1.0 + sweep (t050/055/060) | 7 May / 28 May |
| dsr1-7b | 4.3 GB | v0.1.0 | 17 Apr |
| dsr1-14b | 7.5 GB | v0.1.0 | 18 Apr |

**Notable absences:** No Mistral-7B .tern-model (only CoreML mlpackage exists). No TinyLlama .tern-model.

### Recovered TQ bench results

Two bench results found on archive (both in `analysis_results/`):
1. `tq_bench_results_llama_3_2_1b_instruct_ternpacked_20260513T041543Z.json` — the recovered record with three accountings (0.39×/2.20×/6.74×). **Filed to originals/.**
2. `tq_bench_results_gemma_4_e4b_it_20260512T222533Z.json` — pathological (490K baseline PPL, massive inflation). **Filed to originals/.**

No WikiText-2 sliding-window results file found (referenced in ppl_note but absent from archive).

### Llama-3.2-1B packed-load key mapping issue

From the recovered JSON: `load_packed_keys: {missing: 115, unexpected: 224}`. The .tern-model at `compressed/llama32-1b/llama32_1b_ternary_v0.1.0.tern-model` (849 MB) has key mismatches when loaded against the model architecture. Diagnosis deferred — not blocking current campaign but flagged for B1's Llama-3.2 baselines.

### Additional analysis data on archive
- Per-expert tolerance analysis: phi4, google-base, jackrong, gemma4-31b, qwen3-30b-a3b
- Sensitivity scans: gemma4-26b-a4b, qwen3-30b-a3b (per-channel and aggregate)
- Threshold coherence: qwen3-30b-a3b sweep (t050/055/060)

---

## Originals Archive

All original experiment artifacts assembled in `benchmarks/originals/`:
- `tq_bench_results.json` — original 5.3× result (2026-03-30)
- `tq_bench_results_llama_3_2_1b_instruct_ternpacked_20260513T041543Z.json` — **recovered** Llama-3.2-1B three-accounting record (2026-05-13)
- `tq_bench_results_gemma_4_e4b_it_20260512T222533Z.json` — **recovered** Gemma-4-E4B bench (2026-05-12)
- `energy_results.json` + `energy_benchmark_cleanroom.md` — TinyLlama ANE energy (2026-03-28)
- `mistral7b_phase2.json` + `mistral_7b_phase_d_full.md` — 297.6/5.388 energy data (2026-04-15)
- `tinyllama_benchmark.json` + `tinyllama_benchmark.md` — TinyLlama end-to-end (2026-03-25)
- `tn003_turboquant_baseline.py` — TN-003 orchestration (from git 7d17b1b)
- `tern_infer.py` — TurboQuant integration code
- `TERNARY_BENCHMARK_SUMMARY.md` — March 2026 summary
- `r12_kv_sweep_20260518/` — R12 sweep results (all NaN)

---

## Programme Cube cells (v1.5)

| Cell | State | Date |
|------|-------|------|
| `bench.replication` | SURE — DONE | 8 Aug 2026 |
| `bench.fp16-kv-baseline` | SURE — DONE | 8 Aug 2026 |
| `bench.reconciliation` | SURE — DONE | 9 Aug 2026 |
| `bench.fp16-int8-baseline` | UNSURE — authorized, pending download | — |
| `bench.ternary-kv` | SURE — DONE (QUALITY FAIL) | 9 Aug 2026 |

---

## Claim Sheet — current state (pre-v1)

Locked language unchanged per BUILD 5 exclusions. Reconciliation changes the accounting picture but does not change the public set until Rob + Fable rule.

| Icon | Claim | Reconciled value | Baseline | Model | Tag | Report |
|------|-------|:-------------:|----------|-------|:---:|--------|
| 1 — KV compression | KV ratio | **Opt B: 6.74×** (serving) / **Opt A: varies by seq_len** | FP16-KV 1× (45,056 B/token float32) | d=64, b_mse=3 | Derived | Reconciliation |
| 1 — KV compression | Quality | **0% quality demonstrated** (TQ: NaN; native: 1–15% match) | — | TinyLlama | Measured | B3, R12 |
| 2 — Energy | 5.388 W @ 297.6 tok/s | 5.388 W (from original Phase D) | FP16: pending (B1) / INT8: pending (B1) | Mistral-7B | Measured (2026-04-15) | originals |
| 2 — Energy | 1.64× tok/W | 1257 vs 768 tok/W | FP16 ANE | TinyLlama (ANE) | Measured (2026-03-28) | originals |
| 3 — Throughput | 297.6 tok/s | from original Phase D | FP16: pending (B1) | Mistral-7B | Measured (2026-04-15) | originals |

**Key finding:** Icon 1 has compression ratios (opt B: 6.74× at serving scale) but **zero quality backing**. No closed-loop KV compression approach has passed the 93% floor at any compression level. This is the campaign's primary open question.

ClaimSheet_v1.md deferred until B1 complete and Rob rules on KV compression quality path.
