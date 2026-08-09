# Build Sheet — tern-core Reproducibility Document

**Copyright (c) 2025-2026 Synapticode. All rights reserved.**

> Process, not results. One section per model. How to reproduce every artefact from source weights to published number.

---

## TinyLlama-1.1B — Build Sheet

### Source
- HuggingFace ID: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Params: 1,034M (Block 3 count) / 1,236M (Llama-3.2 adapter count — includes tied weights)
- HF cache: `/Volumes/Syn Archive/cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/` (525 MB)

### Conversion pipeline
- Step 1: `python -m terncore.convert --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --threshold 0.7 --output models/compressed/tinyllama/`
- Output: `tinyllama_ternary.tern-model` — 471.6 MB, 154/155 layers converted, 8.4× vs FP32
- Roadblock: None — TinyLlama is the simplest model in the sprint
- Gate mechanism: Integrity check (pack → unpack → compare, bit-identical)
- Note: Block 3 ran on x86_64 (iMac i9-9900K). ARM/Apple Silicon runs are separate (EXP-003+)

### Artefact locations
- .tern-model: Block 3 artefact (original host, iMac). Demo video shows 471.57 MB
- CoreML mlpackage: not produced for TinyLlama (Block 3 predates CoreML pipeline)
- Benchmark JSON: `benchmarks/results/tinyllama_benchmark.json`
- Benchmark report: `benchmarks/reports/EVIDENCE_PACKAGE.md` §1.1

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (compression) | — | Compression vs FP32 | 8.4× | EXP-001 |
| A (compression) | — | Sparsity | 43.4% | EXP-001 |
| STE PoC | — | PPL improvement | 45.8× (77K→1.7K) | EXP-002 |

### Known issues
- No CoreML mlpackage — conversion pipeline established after Block 3
- Compression measured vs FP32 (not FP16) — consistent with the locked public claim

---

## Mistral-7B-Instruct-v0.3 — Build Sheet

### Source
- HuggingFace ID: `mistralai/Mistral-7B-Instruct-v0.3`
- Params: 7.24B
- FP16 size: ~14.5 GB
- HF cache: `/Volumes/Syn Archive/cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/` (13.5 GB, complete)
- Source safetensors previously on internal disk; migrated safetensors not in Syn Archive `models/source/`

### Conversion pipeline
- Step 1: `python -m terncore.convert --model mistralai/Mistral-7B-Instruct-v0.3 --threshold 0.7 --output models/compressed/mistral-7b/`
- Step 2: `python -m terncore.package --input models/compressed/mistral-7b/*.tern-model --output packages/kaist_delivery/mistral_7b_ternary_v0.3.0.tern-pkg`
- Step 3: CoreML export — `python -m terncore.coreml_export --model ... --output ... --arch-preset mistral-7b`
- Output: .tern-pkg 2.27 GB (96.4% ternary), mlpackage 14.5 GB (FP16-encoded)
- Roadblock: Phase C palettisation fails — Inf values in FP16 weight tensors (tern-core #1)
- Approach angle: Mistral-7B has 96.4% ternary ratio — highest in sprint. The remaining 3.6% are protected layers (embeddings, LM head, norms). At 2.27 GB, the model fits on iPhone storage. The CoreML mlpackage at 14.5 GB remains FP16-encoded because CoreML lacks native trit storage — this size gap (14.5 → 2.27 GB) is the CoreML native ternary partnership argument
- Gate mechanism: CoreML validation (load + predict + output shape check)

### Artefact locations
- .tern-pkg: `/Volumes/Syn Archive/packages/kaist_delivery/mistral_7b_ternary_v0.3.0.tern-pkg` (2.27 GB, on Syn Archive)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/mistral-7b/` (14.5 GB, on Syn Archive). Original path was `tern-core/output/coreml_models/mistral_7b_ternary.mlpackage`
- Benchmark JSON: `benchmarks/results/mistral7b_phase2.json`
- Energy report: `benchmarks/reports/mistral_7b_phase_d_full.md`
- Energy baselines log: `benchmarks/originals/mistral7b_energy_baselines.log`
- Inf weight investigation: `benchmarks/reports/mistral_7b_issue1_validation_2026-04-29.md`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| B (inference) | CPU_AND_NE | tok/s | **297.6** | EXP-005 |
| B (inference) | CPU_AND_NE | latency stdev | 9.7 ms | EXP-005 |
| B (inference) | CPU_AND_GPU | tok/s | 387.8 | EXP-005 |
| C (palettise) | — | — | FAILED (Inf weights) | EXP-005 |
| D (energy) | CPU_AND_NE | power | **5.388 W** | EXP-005 |
| D (energy) | CPU_AND_NE | mJ/inference | 1,122.6 | EXP-005 |
| D (energy) | CPU_ONLY | power | 5.458 W | EXP-005 |
| D (energy) | CPU_AND_GPU | power | 20.142 W | EXP-005 |
| B1 (FP16 baseline) | MPS | tok/s | 16.45 | EXP-019 |
| B1 (INT8 baseline) | CPU (qnnpack) | tok/s | 5.55 | EXP-019 |
| B1 (INT8 baseline) | CPU | compression vs FP32 | 53.9× (artifact) | EXP-019 |

### Known issues
- **tern-core #1**: Inf values in FP16-encoded weights block palettisation (Phase C). Fix belongs in tern-core compiler path
- mlpackage is FP16-encoded (14.5 GB) — no compression benefit until CoreML native trit support
- Original mlpackage path (`output/coreml_models/`) migrated to Syn Archive; local path empty
- **B1 INT8 size anomaly**: INT8 reported as 513 MB / 53.9× compression — `model.parameters()` skips quantised packed_params. The tok/s measurement (5.55) is valid

---

## Llama-3.2-1B-Instruct — Build Sheet

### Source
- HuggingFace ID: `unsloth/Llama-3.2-1B-Instruct`
- Params: 1,235,814,400
- FP16 size: 2,357 MB · FP32 size: 4,714 MB
- HF cache: `/Volumes/Syn Archive/cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/` (2.3 GB)

### Conversion pipeline
- Step 1: Dryrun — `python -m terncore.convert --dry-run --model unsloth/Llama-3.2-1B-Instruct --threshold 0.7`
  - Result: 78.74% ternary, 3.22× est compression vs FP16, 262.7M params retained
- Step 2: Compress — same command without `--dry-run`
  - Output: `llama32_1b_ternary_v0.1.0.tern-model` (849 MB)
- Step 3: CoreML export — `python -m terncore.coreml_export --model ... --arch-preset llama32-1b`
  - Output: mlpackage raw 1,023 MB
- Step 4: Palettise — coremltools `palettize_weights` (2-bit)
  - Output: palettised mlpackage 584 MB
- Roadblock: None — all phases clean
- Gate mechanism: CoreML validation + palettisation integrity

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/llama32-1b/llama32_1b_ternary_v0.1.0.tern-model` (849 MB)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/llama32-1b/` (raw + palettised)
- Dryrun JSON: `benchmarks/results/llama32_1b_dryrun.json`
- Phase 2 JSON: `benchmarks/results/llama32_1b_phase2.json`
- Conversion report: `/Volumes/Syn Archive/models/compressed/llama32-1b/llama32_1b_ternary_v0.1.0_conversion_report.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (dryrun) | — | Ternary ratio | 78.74% | EXP-003 |
| A (compress) | — | .tern-model size | 849 MB | EXP-003 |
| B (inference, raw) | ALL | tok/s | 1,553.9 | EXP-004 |
| B (inference, raw) | CPU_AND_NE | tok/s | 1,501.7 | EXP-004 |
| B (inference, pal) | CPU_AND_NE | tok/s | **1,617.8** | EXP-004 |
| C (palettise) | — | pal 2-bit size | 584 MB | EXP-004 |
| D (energy, pal) | CPU_AND_NE | power | 7.41 W | EXP-004 |
| D (energy, pal) | CPU_AND_NE | mJ/inference | 310.6 | EXP-004 |
| B1 baseline | MPS (FP16) | tok/s | 61.2 | EXP-017 |
| B1 baseline | CPU (INT8) | tok/s | 26.7 | EXP-017 |

### Known issues
- None — cleanest model in the sprint

---

## Llama-3.2-3B-Instruct — Build Sheet

### Source
- HuggingFace ID: `unsloth/Llama-3.2-3B-Instruct`
- Params: 3,212,749,824
- FP16 size: 6,128 MB · FP32 size: 12,256 MB
- HF cache: `/Volumes/Syn Archive/cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct/` (6.0 GB, complete)

### Conversion pipeline
- Step 1: Dryrun — 87.73% ternary, 4.3× est compression vs FP16, 394M params retained
- Step 2: Compress — `llama32_3b_ternary_v0.1.0.tern-model` (1,760 MB)
- Step 3: CoreML export (arch-preset llama32-3b) → raw 2,253 MB
- Step 4: Palettise 2-bit → 1,595 MB
- Roadblock: None
- Gate mechanism: CoreML validation + Phase C clean

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/llama32-3b/llama32_3b_ternary_v0.1.0.tern-model` (1.7 GB)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/llama32-3b/`
- Dryrun JSON: `benchmarks/results/llama32_3b_dryrun.json`
- Phase 2 JSON: `benchmarks/results/llama32_3b_phase2.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (dryrun) | — | Ternary ratio | 87.73% | EXP-006 |
| A (compress) | — | .tern-model size | 1,760 MB | EXP-006 |
| B (inference, pal) | CPU_AND_NE | tok/s | **698.6** | EXP-006 |
| C (palettise) | — | pal 2-bit size | 1,595 MB | EXP-006 |
| D (energy, pal) | CPU_AND_NE | power | 7.29 W | EXP-006 |
| D (energy, pal) | CPU_AND_NE | mJ/inference | 687.6 | EXP-006 |
| B1 (FP16 baseline) | MPS | tok/s | 34.36 | EXP-018 |
| B1 (INT8 baseline) | CPU (qnnpack) | tok/s | 11.05 | EXP-018 |
| B1 (INT8 baseline) | CPU | compression vs FP32 | 8.15× | EXP-018 |

### Known issues
- None

---

## Gemma 3 4B — Build Sheet

### Source
- HuggingFace ID: `unsloth/gemma-3-4b-it`
- Params: 4,300,079,472
- FP16 size: 8,202 MB
- Architecture: Gemma3ForConditionalGeneration (883 weights — multimodal)

### Conversion pipeline
- Step 1: Dryrun — gemma3 adapter, 74.62% ternary, 2.88× vs FP16. 1,091M params retained (encoder components)
- Step 2: Compress → est .tern-model 2,847 MB
- Step 3: CoreML export (arch-preset gemma3-4b) → raw 2,988 MB
- Step 4: Palettise 2-bit → 1,868 MB
- Roadblock: Lower ternary ratio (74.6%) due to multimodal encoder components
- Gate mechanism: CoreML validation + Phase C clean

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/gemma3-4b/` (3.15 GB, sha256 `81b09137…a5d9652c`)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/gemma3-4b/`
- Dryrun JSON: `benchmarks/results/gemma3_4b_dryrun.json`
- Phase 2 JSON: `benchmarks/results/gemma3_4b_phase2.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (dryrun) | — | Ternary ratio | 74.62% | EXP-007 |
| B (inference, pal) | CPU_AND_NE | tok/s | **513.3** | EXP-007 |
| D (energy, pal) | CPU_AND_NE | power | 8.00 W | EXP-007 |
| D (energy, pal) | CPU_AND_NE | mJ/inference | 1,008.7 | EXP-007 |

### Known issues
- None

---

## Gemma 3 12B — Build Sheet

### Source
- HuggingFace ID: Gemma 3 12B (exact ID in compress log)
- Source safetensors: `/Volumes/Syn Archive/models/source/gemma3-12b-it/` (5 shards)

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/gemma3-12b/`
- .mlpackage: `/Volumes/Syn Archive/models/coreml/gemma3-12b/`
- Phase 2 JSON: `benchmarks/results/gemma3_12b_phase2.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| B (inference, pal) | ALL | tok/s | **33.9** | EXP-008 |
| D (energy) | ALL | power | 27.03 W | EXP-008 |

### Known issues
- CPU_AND_NE extremely slow (4.7 tok/s, 13.7s/inference) — ANE dispatch overhead at 12B scale
- 43.9 GB peak RSS — near 64 GB ceiling

---

## Phi-4 14B — Build Sheet

### Source
- HuggingFace ID: `microsoft/phi-4`
- Params: 14,659,507,200
- FP16 size: 27,961 MB

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/phi4-14b/` (sha256 `6bf59562…0c6dd7` on weight.bin)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/phi4-14b/`
- Dryrun JSON: `benchmarks/results/phi4_14b_dryrun.json`
- Phase 2 JSON: `benchmarks/results/phi4_14b_phase2.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (dryrun) | — | Ternary ratio | 92.99% | EXP-009 |
| A (dryrun) | — | Compression vs FP16 | 5.37× | EXP-009 |
| B (inference) | CPU_AND_GPU | tok/s | 29.4 | EXP-009 |
| D (energy) | ALL | power | 27.58 W | EXP-009 |

### Known issues
- CPU_AND_NE impractical (4.8 tok/s) — same ANE cliff as Gemma 12B
- 47.5 GB peak RSS — near ceiling

---

## Qwen2.5-7B-Instruct — Build Sheet

### Source
- HuggingFace ID: `Qwen/Qwen2.5-7B-Instruct`
- Params: 7,615,616,512
- FP16 size: 14,526 MB

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/qwen25-7b/`
- .mlpackage: `/Volumes/Syn Archive/models/coreml/qwen25-7b/`
- Dryrun JSON: `benchmarks/results/qwen25_7b_dryrun.json`
- Phase 2 JSON: `benchmarks/results/qwen25_7b_phase2.json`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| A (dryrun) | — | Ternary ratio | 85.68% | EXP-010 |
| B (inference) | ALL | tok/s | 54.3 | EXP-010 |
| B (inference, pal) | ALL | tok/s | 55.7 | EXP-010 |
| D (energy) | CPU_AND_GPU | power | 28.00 W | EXP-010 |

### Known issues
- None

---

## DSR1-7B / DSR1-14B — Build Sheet

### Source
- HuggingFace ID: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` / `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- 7B: dryrun confirmed 85.68% ternary (identical to Qwen2.5-7B — same base architecture)

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/dsr1-7b/`, `dsr1-14b/`
- .mlpackage: `/Volumes/Syn Archive/models/coreml/dsr1-7b/`, `dsr1-14b/`
- Phase 2 JSONs: `benchmarks/results/dsr1_7b_phase2.json`, `benchmarks/results/dsr1_14b_phase2.json`

### Phase results summary
| Phase | Model | Compute unit | Key metric | Value | EXP-ID |
|-------|-------|-------------|-----------|-------|--------|
| B | 7B | ALL | tok/s | 56.2 | EXP-011 |
| B | 7B | CPU_AND_NE | tok/s | 53.5 | EXP-011 |
| D | 7B | ALL | power | 33.29 W | EXP-011 |
| B | 14B | CPU_AND_GPU | tok/s | 29.2 | EXP-011 |
| D | 14B | CPU_AND_GPU | power | 26.21 W | EXP-011 |

### Known issues
- 14B: CPU_AND_NE impractical (same ANE cliff)
- 14B: 51.7 GB peak RSS

---

## Bonsai-8B — Build Sheet

### Source
- HuggingFace ID: `prism-ml/Ternary-Bonsai-8B-unpacked`
- Pre-trained ternary model (weights already {-1,0,+1})
- HF cache: `/Volumes/Syn Archive/cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-8B-unpacked/` (5.0 GB)

### Conversion pipeline
- No tern-core conversion — model ships ternary-native
- CoreML export + pack for inference only
- Coherence verification via `parity_bonsai8b.py` (n=64 prompts, argmax-exact match)

### Artefact locations
- .mlpackage: `~/synapticode/model-library/coreml/bonsai-8b/bonsai-8b-s64.mlpackage`
- Compressed source: `~/synapticode/model-library/compressed/bonsai-8b/`
- Phase 2 JSON: `~/synapticode/model-library/benchmarks/bonsai-8b/phase2.json`
- Parity scripts: `ecc-ternary/uploads/PACK_BONSAI8B_20260601T025834Z/`

### Phase results summary
| Phase | Compute unit | Key metric | Value | EXP-ID |
|-------|-------------|-----------|-------|--------|
| B | ALL | tok/s | 299.1 | EXP-012 |
| B | CPU_AND_NE | tok/s | 277.9 | EXP-012 |
| D | CPU_AND_NE | power | 5.63 W | EXP-012 |
| Coherence | — | Tier | **2.5** (argmax-exact, n=64) | EXP-012 |

### Known issues
- None

---

## Llama-3.1-70B — Build Sheet

### Source
- HuggingFace ID: `meta-llama/Llama-3.1-70B-Instruct`
- FP16: ~131 GB
- Source tokenizer stub: `/Volumes/Syn Archive/models/source/llama-3-1-70b/` (8.9 MB)

### Conversion pipeline
- Step 1: Streaming shard-by-shard ternary conversion (tern-core v0.6.0) → 37.0 GB .tern-model
- Step 2: CoreML export → 38.96 GB mlpackage (iOS 18 spec)
- Roadblock: Inference blocked — decompression expands to ~116 GB, exceeds 64 GB M4 Pro
- Approach angle: Streaming pipeline never loads full model. 3.54× compression proves scaling. M2/M3 Ultra (192 GB) required for inference

### Artefact locations
- .tern-model: `/Volumes/Syn Archive/models/compressed/llama-3-1-70b/` (37.0 GB)
- .mlpackage: `/Volumes/Syn Archive/models/coreml/llama-3-1-70b/` (38.96 GB, sha256 `076f4388…935571d`)
- Phase 2 JSON: `benchmarks/results/llama70b_phase2.json` (conversion data only, no inference)
- Analysis: `docs/TN-001_llama70b_compression_analysis.md`

### Known issues
- Inference blocked on M4 Pro 64 GB — demo artefact only
- Source safetensors removed from workspace, backed up to MacBook Pro

---

## Gemma 4 E4B — Build Sheet

### Source
- HuggingFace ID: `google/gemma-4-E4B-it`
- Params: 7,996,157,418 (multimodal dense)
- FP16 size: 15,252 MB

### Conversion pipeline
- Step 1: Dryrun with gemma4 adapter — 49.34% ternary, 1.76× est compression
- Blocked: Full compression requires coremltools 9.1/9.2
- Approach angle: Only 49.3% ternary (4.05B of 8.0B params in retained multimodal components). Limited compression benefit from ternary at this ratio

### Artefact locations
- Dryrun JSON: `benchmarks/results/gemma4_e4b_dryrun.json`
- Compressed (partial): `/Volumes/Syn Archive/models/compressed/gemma4-e4b/`

### Known issues
- Blocked on coremltools 9.1/9.2
- Low ternary ratio (49.3%) limits value proposition
- Gemma 4 adapter required and written — validated on E4B

---

*Additional models on Syn Archive (compressed but not yet in phase2 suite): gemma4-26b-a4b, gemma4-31b, gemopus-4-26b-a4b, gemopus-4-e4b, qwen3-30b-a3b. Build sheets for these will be added when phase2 runs are conducted.*
