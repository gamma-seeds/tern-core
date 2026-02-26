# Patent-to-Code Mapping

**Synapticode Co., Ltd. — CNS Synaptic Ternary Computing**
**Date:** 2026-02-26
**tern-core version:** `bdb4f84` (Day 15)
**Hardware:** iMac 2019, Intel i9-9900K, 16 GB DDR4, macOS Darwin 24.6.0

---

## How to Read This Document

This document maps each demonstrated patent to its implementing source code. It serves three audiences:

1. **IP Australia** — Establishes reduction to practice. Each claim maps to a specific file, function, line number, test, and measured result.
2. **Apple evaluation** — Proves the patent portfolio has working code behind every claim. An engineer can follow patent claim -> source file -> test -> benchmark result in a single reading.
3. **KSGC reviewers** — Demonstrates that 14 patents from the 56-patent portfolio have real, tested, measured implementations.

**Conventions:**
- All source paths are relative to `src/terncore/` unless otherwise noted.
- Line numbers reference commit `bdb4f84` (Day 15).
- All results are reproducible via commands in `benchmarks/EVIDENCE_PACKAGE.md`.
- "Day N" references correspond to the 12-day engineering sprint documented in `benchmarks/RESULTS.md`.

---

## Patent 1 — Ternary Weight Encoding

### Claim Summary

A method for encoding neural network weights as ternary values {-1, 0, +1} using adaptive threshold quantisation. The threshold delta is computed as `threshold * mean(|W|)`, mapping each weight to +1 (above delta), -1 (below -delta), or 0 (within delta). A per-layer scaling factor alpha (mean of absolute non-zero original weights) preserves magnitude information.

### Implementation

| Component | Location |
|-----------|----------|
| Primary source | `arithmetic/quantizer.py` |
| Key class | `TernaryQuantizer` (line 36) |
| Quantise method | `TernaryQuantizer.quantize()` (line 60) |
| Dequantise method | `TernaryQuantizer.dequantize()` (line 97) |
| Statistics | `TernaryQuantizer.stats()` (line 112) |
| Drop-in Linear | `arithmetic/linear.py`, `TernaryLinear` (line 36) |
| Eval forward | `TernaryLinear._forward_eval()` (line 105) |
| STE gradient | `ste.py`, `STEQuantize.forward()` (line 45) |

### How It Works

Weights are quantised via threshold-based comparison: `ternary = sign(W) * (|W| > delta)` where `delta = threshold * mean(|W|)`. The scaling factor `alpha = mean(|W[W != 0]|)` is stored per-layer. `TernaryLinear` is a drop-in replacement for `nn.Linear` that uses STE gradients during training and cached ternary weights during eval. The `STEQuantize` autograd function passes gradients straight through the discrete quantisation step.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestTernaryQuantizer::test_output_values` | `tests/test_stage1a.py` | Output contains only {-1, 0, +1} |
| `TestTernaryQuantizer::test_alpha_positive` | `tests/test_stage1a.py` | Scaling factor is positive |
| `TestTernaryQuantizer::test_shape_preserved` | `tests/test_stage1a.py` | Output shape matches input |
| `TestTernaryQuantizer::test_reconstruction_error_bounded` | `tests/test_stage1a.py` | Dequantised weights approximate originals |
| `TestTernaryQuantizer::test_sparsity_increases_with_threshold` | `tests/test_stage1a.py` | Higher threshold produces more zeros |
| `TestTernaryQuantizer::test_dequantize_roundtrip` | `tests/test_stage1a.py` | Dequantise(quantise(W)) is consistent |
| `TestTernaryQuantizer::test_known_values` | `tests/test_stage1a.py` | Hand-computed reference values match |
| `TestTernaryQuantizer::test_stats_fractions_sum_to_one` | `tests/test_stage1a.py` | +1, -1, 0 fractions sum to 1.0 |
| `TestTernaryLinear::test_forward_shape` | `tests/test_stage1a.py` | Forward pass output shape correct |
| `TestTernaryLinear::test_forward_eval_shape` | `tests/test_stage1a.py` | Eval mode forward shape correct |
| `TestTernaryLinear::test_gradient_flows` | `tests/test_stage1a.py` | STE gradients propagate through quantisation |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| FP32 baseline perplexity | 7.19 | Day 1, `benchmarks/eval_perplexity.py` |
| All-ternary perplexity | 130,127 | Day 1, `benchmarks/eval_perplexity.py` |
| Sparsity at threshold 0.7 | 43-46% | Day 2, `benchmarks/eval_sensitivity.py` |
| Weight fractions (+1/-1/0) | ~28%/~28%/~44% | Day 5, `benchmarks/analyse_weights.py` |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestTernaryQuantizer -v
pytest tests/test_stage1a.py::TestTernaryLinear -v
python benchmarks/eval_perplexity.py --skip-accel
```

---

## Patent 4 — Progressive Compression via STE

### Claim Summary

A method for progressive neural network compression using Straight-Through Estimator (STE) gradient-based training. Discrete ternary quantisation is made differentiable by passing gradients through the quantisation step unchanged. Combined with per-layer sensitivity analysis, this enables selective ternarisation of layers ordered by tolerance, achieving quality-compression trade-offs.

### Implementation

| Component | Location |
|-----------|----------|
| STE autograd function | `ste.py`, `STEQuantize` (line 31) |
| STE forward (quantise) | `STEQuantize.forward()` (line 45) |
| STE backward (straight-through) | `STEQuantize.backward()` (line 86) |
| STE linear layer | `ste.py`, `TernaryLinearSTE` (line 100) |
| STE training loop | `ste_trainer.py`, `STETrainer` (line 69) |
| Sensitivity analyser | `arithmetic/quantizer.py`, `SensitivityAnalyzer` (line 145) |
| Per-layer analysis | `SensitivityAnalyzer.analyze_layer()` (line 171) |
| Full-model analysis | `SensitivityAnalyzer.analyze_model()` (line 211) |
| Mixed-precision configs | `mixed_precision.py` |

### How It Works

`STEQuantize.forward()` quantises weights to {-1, 0, +1} during the forward pass. `STEQuantize.backward()` passes the upstream gradient through unchanged (the "straight-through" trick), enabling gradient descent on discrete weights. `SensitivityAnalyzer` evaluates each layer independently by ternarising it while keeping all others in FP32, measuring perplexity impact. Layers are ranked by sensitivity ratio (ternary PPL / baseline PPL) to guide progressive ternarisation order.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestSensitivityAnalyzer::test_analyze_layer` | `tests/test_stage1a.py` | Single-layer sensitivity produces valid ratio |
| `TestSensitivityAnalyzer::test_analyze_model` | `tests/test_stage1a.py` | Full-model analysis returns per-layer results |
| `TestTernaryLinear::test_gradient_flows` | `tests/test_stage1a.py` | STE gradients propagate correctly |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Layers analysed | 155 | Day 2, `benchmarks/eval_sensitivity.py` |
| Layers below 1.1x baseline | 135 (87.1%) | Day 2, `benchmarks/eval_sensitivity.py` |
| Catastrophic outlier | `model.layers.2.mlp.down_proj` (9,609x) | Day 2, `benchmarks/eval_sensitivity.py` |
| STE pre-train PPL | 77,370 | Day 4, `benchmarks/eval_ste_training.py` |
| STE post-train PPL (500 steps) | 1,688 | Day 4, `benchmarks/eval_ste_training.py` |
| STE improvement | 45.8x (97.8%) | Day 4, `benchmarks/eval_ste_training.py` |
| v_proj_late3 (3 layers) PPL | 5.98 (+2.8%) | Day 3, `benchmarks/eval_mixed_precision.py` |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestSensitivityAnalyzer -v
python benchmarks/eval_sensitivity.py
python benchmarks/eval_ste_training.py --steps 500 --eval-tokens 2048
python benchmarks/eval_mixed_precision.py
```

---

## Patent 6 — Ternary Model Storage Format

### Claim Summary

A binary file format (.tern-model v2) for storing mixed-precision neural network models containing both ternary (2-bit packed) and floating-point (FP16) layers. The format uses a fixed 256-byte header, a JSON manifest with per-layer byte offsets for random access, 32-byte aligned weight storage for direct SIMD DMA, and a CRC32 footer for integrity verification.

### Implementation

| Component | Location |
|-----------|----------|
| Primary source | `tern_model.py` |
| Writer class | `TernModelWriter` (line 56) |
| Write method | `TernModelWriter.write()` (line 211) |
| Ternary packing | `TernModelWriter.pack_ternary()` (line 359) |
| Bitmap generation | `TernModelWriter.generate_sparsity_bitmap()` (line 399) |
| Reader class | `TernModelReader` (line 433) |
| Integrity check | `TernModelReader.verify()` (line 487) |
| Layer reconstruction | `TernModelReader.reconstruct_layer()` (line 549) |
| Full reconstruction | `TernModelReader.reconstruct_all()` (line 638) |
| Lazy loading API | `TernModelReader.layer()` (line 674), `.layer_names()` (line 682), `.layer_info()` (line 686) |
| Model loading | `TernModelReader.load_as_model()` |
| Format specification | `docs/tern-model-spec.md` |

### How It Works

`TernModelWriter` accepts layers as either ternary (2-bit packed, 4 weights/byte, encoding: 00=0, 01=+1, 10=-1) or FP16. Each layer's weight data is written at 32-byte aligned offsets into the binary file. A JSON manifest records layer name, dtype, shape, offset, size, and metadata. The 256-byte header contains magic bytes ("TERN"), version, section offsets, and layer counts. A 16-byte footer stores CRC32, file size, and reverse magic ("NRET"). `TernModelReader` parses the header and manifest at init, then provides random-access layer reading via byte offsets.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestWriteSingleLayer::test_write_single_layer` | `tests/test_tern_model.py` | Single ternary layer write/read |
| `TestWriteMixedPrecision::test_write_mixed_precision` | `tests/test_tern_model.py` | Mixed ternary + FP16 layers |
| `TestAlignment::test_alignment` | `tests/test_tern_model.py` | 32-byte weight alignment |
| `TestHeaderMagic::test_header_magic` | `tests/test_tern_model.py` | "TERN" magic bytes |
| `TestManifestReadable::test_manifest_readable` | `tests/test_tern_model.py` | JSON manifest parses correctly |
| `TestFileIntegrity::test_file_integrity` | `tests/test_tern_model.py` | CRC32 passes on valid file |
| `TestFileIntegrity::test_file_integrity_corrupted` | `tests/test_tern_model.py` | CRC32 fails on corrupted file |
| `TestFooterMagic::test_footer_magic` | `tests/test_tern_model.py` | "NRET" reverse magic at EOF |
| `TestHeaderSize::test_header_size` | `tests/test_tern_model.py` | Header is exactly 256 bytes |
| `TestFileSizeConsistency::test_file_size_matches` | `tests/test_tern_model.py` | Footer size matches actual file |
| `TestRandomAccess::test_random_access_read` | `tests/test_tern_model.py` | Read specific layer by manifest offset |
| `TestBiasHandling::test_layer_with_bias` | `tests/test_tern_model.py` | Bias vector stored and retrieved |
| `TestReconstructTernaryLayer::test_reconstruct_ternary_layer` | `tests/test_tern_model.py` | Ternary layer round-trip |
| `TestReconstructFP16Layer::test_reconstruct_fp16_layer` | `tests/test_tern_model.py` | FP16 layer round-trip |
| `TestReconstructAllMixed::test_reconstruct_all_mixed` | `tests/test_tern_model.py` | Mixed-model full reconstruction |
| `TestRoundTripLogitsSynthetic::test_roundtrip_logits_synthetic` | `tests/test_tern_model.py` | End-to-end logit-identical round-trip |
| `TestLazyAPI::test_layer_names` | `tests/test_tern_model.py` | Lazy API returns layer names |
| `TestLazyAPI::test_layer_info` | `tests/test_tern_model.py` | Lazy API returns layer metadata |
| `TestLazyAPI::test_lazy_single_layer` | `tests/test_tern_model.py` | Single-layer lazy read |
| `TestLazyAPI::test_load_all` | `tests/test_tern_model.py` | Full state_dict reconstruction |
| `TestLoadAsModel::test_load_as_model` | `tests/test_tern_model.py` | Load into nn.Module |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| TinyLlama .tern-model size | 471.6 MB | Day 10, `benchmarks/bench_day10_pipeline.py` |
| Compression vs FP32 | 8.4x | Day 10, `benchmarks/bench_day10_pipeline.py` |
| Round-trip tensor max diff | 0.0 (bit-identical) | Day 7, `benchmarks/bench_day7_roundtrip.py` |
| Round-trip logit max diff | 0.0 (bit-identical) | Day 7, `benchmarks/bench_day7_roundtrip.py` |
| Header parse time | 13.9 ms | Day 7, `benchmarks/bench_day7_roundtrip.py` |
| Full reconstruct time | 10.5s | Day 7, `benchmarks/bench_day7_roundtrip.py` |

### Reproducibility
```bash
pytest tests/test_tern_model.py -v
python benchmarks/bench_day6.py
python benchmarks/bench_day7_roundtrip.py
```

---

## Patent 7 — Sparsity Bitmap and Zero-Skip

### Claim Summary

A method for generating and caching a sparsity bitmap from ternary weights, where each bit indicates whether the corresponding weight is non-zero. The bitmap enables zero-skip execution: the compute kernel iterates only over non-zero weights, skipping multiply-accumulate operations for zero entries. The bitmap is generated once at construction time and cached as a persistent buffer.

### Implementation

| Component | Location |
|-----------|----------|
| Bitmap generation | `sparse/__init__.py`, `generate_sparsity_bitmap()` (line 46) |
| Block-level analysis | `sparse/__init__.py`, `analyze_block_sparsity()` (line 156) |
| Cached bitmap storage | `packed_linear.py`, `PackedTernaryLinear` (line 49) |
| Bitmap caching in init | `PackedTernaryLinear.from_float()` (line 100) |
| Bitmap from .tern-model | `PackedTernaryLinear.from_packed_data()` (line 185) |
| C sparse kernel | `csrc/sparse_skip.c`, `tern_sparse64_matvec_f32()` (line 98) |

### How It Works

`generate_sparsity_bitmap()` converts a ternary weight tensor to a 1-bit-per-weight bitmap packed into uint64 words (32 trits per word). `PackedTernaryLinear` stores the bitmap as a registered buffer at construction time, avoiding per-forward recomputation. The C sparse kernel (`tern_sparse64_matvec_f32`) uses CTZ (count-trailing-zeros) bit-scan to iterate only over set bits in the bitmap, skipping all-zero regions without branch overhead.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestSparse::test_bitmap_shape` | `tests/test_stage1a.py` | Bitmap has correct dimensions |
| `TestSparse::test_bitmap_correctness` | `tests/test_stage1a.py` | Bitmap bits match non-zero weights |
| `TestBitmapCaching::test_bitmap_stored_at_construction` | `tests/test_sparsity.py` | Bitmap is a persistent buffer |
| `TestBitmapCaching::test_bitmap_matches_weights` | `tests/test_sparsity.py` | Cached bitmap matches weight pattern |
| `TestBitmapCaching::test_forward_with_cached_bitmap_matches_reference` | `tests/test_sparsity.py` | Cached path matches F.linear |
| `TestBitmapCaching::test_bitmap_from_tern_model` | `tests/test_sparsity.py` | Bitmap survives serialisation |
| `TestBlockSparsity::test_block_analysis_all_zero` | `tests/test_sparsity.py` | All-zero blocks detected |
| `TestBlockSparsity::test_block_analysis_partial` | `tests/test_sparsity.py` | Partial sparsity reported correctly |
| `TestZeroSkipCorrectness::test_zero_skip_same_output` | `tests/test_sparsity.py` | Zero-skip kernel matches dense output |
| `TestZeroSkipCorrectness::test_high_sparsity_correctness` | `tests/test_sparsity.py` | Correctness at 90% sparsity |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Bitmap caching speedup | 2.07x | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Cached vs F.linear ratio | 0.92x (competitive) | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Zero-skip at 40% sparsity | 1.22x speedup | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Zero-skip at 60% sparsity | 1.80x speedup | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Zero-skip at 90% sparsity | 5.28x speedup | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Break-even sparsity | ~35% | Day 9, `benchmarks/bench_day9_sparsity.py` |

### Reproducibility
```bash
pytest tests/test_sparsity.py -v
python benchmarks/bench_day9_sparsity.py
```

---

## Patent 8 — Packed 2-Bit Weight Storage

### Claim Summary

A method for storing ternary neural network weights in a packed 2-bit format, achieving 16x compression versus FP32. Each byte stores 4 ternary weights using the encoding: 00=0, 01=+1, 10=-1. The packed representation is used for both on-disk storage (.tern-model format) and in-memory inference via `PackedTernaryLinear`, a drop-in replacement for `nn.Linear`.

### Implementation

| Component | Location |
|-----------|----------|
| PackedTernaryLinear class | `packed_linear.py` (line 49) |
| From float weights | `PackedTernaryLinear.from_float()` (line 100) |
| From TernaryLinear | `PackedTernaryLinear.from_ternary_linear()` (line 143) |
| From packed data | `PackedTernaryLinear.from_packed_data()` (line 185) |
| Forward pass | `PackedTernaryLinear.forward()` (line 236) |
| Model-wide conversion | `convert_model_to_packed()` (line 297) |
| 2-bit packing | `sparse/__init__.py`, `pack_ternary_weights()` (line 62) |
| 2-bit unpacking | `sparse/__init__.py`, `unpack_ternary_weights()` (line 116) |
| .tern-model packed load | `tern_model.py`, `TernModelReader.load_packed_model()` (line 716) |

### How It Works

`pack_ternary_weights()` encodes a ternary tensor into uint8 bytes, 4 weights per byte, LSB-first. `PackedTernaryLinear` stores these packed bytes as a registered buffer (not a parameter), along with a scalar alpha and optional bias. On forward, weights are unpacked to float, scaled by alpha, and passed to `F.linear`. `convert_model_to_packed()` traverses a model and replaces all `nn.Linear` and `TernaryLinear` layers in-place. `TernModelReader.load_packed_model()` loads a .tern-model directly into `PackedTernaryLinear` layers without re-quantisation.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestPackedTernaryLinear::test_from_float_basic` | `tests/test_packed_linear.py` | Construction from float weights |
| `TestPackedTernaryLinear::test_from_float_matches_ternary` | `tests/test_packed_linear.py` | Output matches TernaryLinear |
| `TestPackedTernaryLinear::test_from_ternary_linear` | `tests/test_packed_linear.py` | Conversion from TernaryLinear |
| `TestPackedTernaryLinear::test_from_packed_data` | `tests/test_packed_linear.py` | Construction from raw packed bytes |
| `TestPackedTernaryLinear::test_memory_footprint` | `tests/test_packed_linear.py` | Storage uses uint8 buffers |
| `TestPackedTernaryLinear::test_3d_input` | `tests/test_packed_linear.py` | Handles (batch, seq, features) input |
| `TestPackedOps::test_packed_matmul_correctness` | `tests/test_packed_linear.py` | Packed matmul matches reference |
| `TestModelConversion::test_convert_simple_model` | `tests/test_packed_linear.py` | Full model conversion to packed |
| `TestModelConversion::test_memory_reduction_after_conversion` | `tests/test_packed_linear.py` | Memory reduction measured |
| `TestTernModelReaderPacked::test_load_packed_from_tern_model` | `tests/test_packed_linear.py` | Load .tern-model as packed layers |
| `TestSparse::test_pack_unpack_roundtrip` | `tests/test_stage1a.py` | Pack/unpack round-trip preserves values |
| `TestSparse::test_compression_ratio` | `tests/test_stage1a.py` | 4:1 byte compression verified |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Weight storage compression | 10.7x (3 bits/weight effective) | Day 8, `benchmarks/bench_day8_packing.py` |
| In-memory compression (GPT-2) | 2.5x | Day 12, `benchmarks/bench_day12_performance.py` |
| In-memory compression (BERT) | 3.4x | Day 12, `benchmarks/bench_day12_performance.py` |
| TinyLlama .tern-model size | 471.6 MB (vs 4,137 MB FP32) | Day 10, `benchmarks/bench_day10_pipeline.py` |
| BERT .tern-model size | 30.9 MB (vs 418 MB FP32) | Day 11, `benchmarks/bench_day11_multi_model.py` |

### Reproducibility
```bash
pytest tests/test_packed_linear.py -v
pytest tests/test_stage1a.py::TestSparse -v
python benchmarks/bench_day8_packing.py
```

---

## Patent 9 — Block-Level Sparsity Analysis

### Claim Summary

A method for analysing sparsity at the block level within ternary weight matrices, determining what fraction of fixed-size blocks (e.g., 64-element) are entirely zero. Block-level all-zero detection enables coarser-grained skip logic in hardware, bypassing entire vector loads rather than individual elements.

### Implementation

| Component | Location |
|-----------|----------|
| Block analysis | `sparse/__init__.py`, `analyze_block_sparsity()` (line 156) |
| Model-wide report | `sparse/__init__.py`, `model_sparsity_report()` (line 229) |
| Sparsity info | `sparse/__init__.py`, `sparsity_info()` (line 256) |
| C sparse kernel | `csrc/sparse_skip.c`, `tern_sparse64_matvec_f32()` (line 98) |

### How It Works

`analyze_block_sparsity()` divides a ternary weight tensor into fixed-size blocks and counts the fraction of blocks that are entirely zero. It returns element-level sparsity, block-level sparsity, and block-skip ratio (fraction of blocks skippable). The C kernel `tern_sparse64_matvec_f32` uses CTZ bit-scan to iterate non-zero elements within each uint64 bitmap word, achieving element-level zero-skip. The analysis revealed that uniform random sparsity produces near-zero all-zero blocks — structured pruning would be needed for block-level gains.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestBlockSparsity::test_block_analysis_all_zero` | `tests/test_sparsity.py` | 100% block-skip for all-zero tensor |
| `TestBlockSparsity::test_block_analysis_no_zero` | `tests/test_sparsity.py` | 0% block-skip for dense tensor |
| `TestBlockSparsity::test_block_analysis_partial` | `tests/test_sparsity.py` | Correct partial sparsity stats |
| `TestBlockSparsity::test_block_analysis_returns_expected_keys` | `tests/test_sparsity.py` | Returns all expected fields |
| `TestBlockSparsity::test_model_sparsity_report` | `tests/test_sparsity.py` | Model-wide report covers all layers |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Element sparsity (typical) | 43-46% at threshold 0.7 | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Block-skip ratio (uniform 80%) | Near zero | Day 9, `benchmarks/bench_day9_sparsity.py` |
| CTZ element-skip at 90% | 5.28x speedup | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Break-even sparsity for speedup | ~35% | Day 9, `benchmarks/bench_day9_sparsity.py` |

### Reproducibility
```bash
pytest tests/test_sparsity.py::TestBlockSparsity -v
python benchmarks/bench_day9_sparsity.py
```

---

## Patent 10 — Inference Engine Architecture

### Claim Summary

An automated inference engine that converts a standard PyTorch neural network model to ternary execution. The engine traverses the module tree, identifies eligible weight layers (Linear and Conv1D), protects precision-critical layers (embeddings, LayerNorm, RMSNorm, LM head) from quantisation, and replaces eligible layers with ternary equivalents. The conversion is in-place and produces a deployment-ready model.

### Implementation

| Component | Location |
|-----------|----------|
| Primary source | `engine/inference.py` |
| Engine class | `TernaryInferenceEngine` (line 61) |
| Convert method | `TernaryInferenceEngine.convert()` (line 97) |
| Inference method | `TernaryInferenceEngine.infer()` (line 191) |
| Protection logic | `TernaryInferenceEngine._should_protect()` (line 233) |
| Module replacement | `TernaryInferenceEngine._replace_module()` (line 306) |
| CLI converter | `convert.py`, `TernaryConverter` (line 99) |
| CLI convert method | `TernaryConverter.convert()` (line 144) |
| CLI entry point | `convert.py`, `main()` (line 466) |
| CLI command | `tern-convert` (registered in `pyproject.toml`) |

### How It Works

`TernaryInferenceEngine.convert()` walks the model's module tree. For each `nn.Linear` (or Conv1D), it checks `_should_protect()` against name patterns ("embed", "layernorm", "layer_norm", "rmsnorm", "lm_head", "output", "classifier"). Protected layers stay in FP16. Eligible layers are replaced by `TernaryLinear` via `_replace_module()`, which handles parent module attribute setting. `TernaryConverter` wraps this with .tern-model output, verification, and statistics reporting.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestInferenceEngine::test_convert_simple_model` | `tests/test_stage1a.py` | Basic model conversion works |
| `TestInferenceEngine::test_inference_produces_output` | `tests/test_stage1a.py` | Converted model produces output |
| `TestInferenceEngine::test_deterministic_inference` | `tests/test_stage1a.py` | Two infer() calls produce identical output |
| `TestInferenceEngine::test_conversion_report` | `tests/test_stage1a.py` | Report contains expected fields |
| `TestTernaryConverter::test_convert_synthetic_model` | `tests/test_convert.py` | Full converter pipeline |
| `TestTernaryConverter::test_protection_patterns` | `tests/test_convert.py` | Protection patterns match correctly |
| `TestTernaryConverter::test_protection_always_protects_critical` | `tests/test_convert.py` | Critical layers never quantised |
| `TestTernaryConverter::test_output_file_loadable` | `tests/test_convert.py` | .tern-model output is valid |
| `TestTernaryConverter::test_round_trip_synthetic` | `tests/test_convert.py` | Write+read round-trip works |
| `TestTernaryConverter::test_verify_output` | `tests/test_convert.py` | CRC32 verification passes |
| `TestCLI::test_cli_help` | `tests/test_convert.py` | CLI --help exits cleanly |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| TinyLlama: 154/155 layers converted | 1 protected (lm_head) | Day 10, `benchmarks/bench_day10_pipeline.py` |
| Conversion time (TinyLlama) | 212.7s | Day 11, `benchmarks/bench_day11_multi_model.py` |
| Conversion time (GPT-2) | 14.9s | Day 11, `benchmarks/bench_day11_multi_model.py` |
| Conversion time (BERT) | 14.7s | Day 11, `benchmarks/bench_day11_multi_model.py` |
| Architectures validated | 5 (TinyLlama, GPT-2, GPT-2-medium, BERT, DistilGPT-2) | Day 11 |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestInferenceEngine -v
pytest tests/test_convert.py::TestTernaryConverter -v
python -m terncore.convert distilgpt2 --output /tmp/test.tern --verify
```

---

## Patent 11 — Conv1D Layer Support

### Claim Summary

A method for extending ternary conversion to support HuggingFace Conv1D layers, which store weights transposed relative to nn.Linear. Conv1D uses shape `(in_features, out_features)` versus nn.Linear's `(out_features, in_features)`. The conversion detects Conv1D layers, transposes weights for quantisation, and replaces them with standard TernaryLinear modules.

### Implementation

| Component | Location |
|-----------|----------|
| Conv1D detection | `convert.py`, `_is_weight_layer()` (line 56) |
| Weight extraction | `convert.py`, `_get_weight_and_shape()` (line 66) |
| Engine Conv1D handling | `engine/inference.py`, within `convert()` (line 97) |

### How It Works

`_is_weight_layer()` checks if a module is either `nn.Linear` or `transformers.pytorch_utils.Conv1D`. `_get_weight_and_shape()` extracts the weight tensor and transposes it if the source is Conv1D (converting from `(in, out)` to `(out, in)` layout). The converted TernaryLinear always uses standard `(out, in)` layout regardless of the source layer type. This enables the same conversion pipeline to handle GPT-2 family models (Conv1D) and LLaMA family models (nn.Linear) without model-specific code.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestConv1DSupport::test_conv1d_detected` | `tests/test_convert.py` | Conv1D layers identified as weight layers |
| `TestConv1DSupport::test_conv1d_weight_transposed` | `tests/test_convert.py` | Weight shape transposed correctly |
| `TestConv1DSupport::test_conv1d_model_conversion` | `tests/test_convert.py` | Full model with Conv1D layers converts |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| GPT-2: 48/49 Conv1D layers converted | 1 protected (lm_head) | Day 11, `benchmarks/bench_day11_multi_model.py` |
| GPT-2 .tern-model size | 104.3 MB (4.55x compression) | Day 11, `benchmarks/bench_day11_multi_model.py` |
| DistilGPT-2: 24/25 Conv1D layers converted | 1 protected (lm_head) | Day 11, `benchmarks/bench_day11_multi_model.py` |
| GPT-2-medium: 96/97 Conv1D layers converted | 1 protected (lm_head) | Day 11, `benchmarks/bench_day11_multi_model.py` |

### Reproducibility
```bash
pytest tests/test_convert.py::TestConv1DSupport -v
python -m terncore.convert gpt2 --output /tmp/gpt2.tern --verify
```

---

## Patent 12 — Automated Conversion Pipeline

### Claim Summary

An end-to-end CLI pipeline that downloads a HuggingFace model, converts it to ternary, writes a .tern-model file, and optionally verifies integrity — all in a single command. The pipeline handles model loading, layer detection (nn.Linear and Conv1D), protection of critical layers, quantisation, packing, serialisation, and CRC32 verification.

### Implementation

| Component | Location |
|-----------|----------|
| CLI entry point | `convert.py`, `main()` (line 466) |
| Converter class | `convert.py`, `TernaryConverter` (line 99) |
| Convert method | `TernaryConverter.convert()` (line 144) |
| HuggingFace loader | `hf_loader/__init__.py`, `HFTernaryLoader` (line 113) |
| Load and convert | `HFTernaryLoader.load_and_convert()` (line 152) |
| CLI registration | `pyproject.toml`, `[project.scripts]` |

### How It Works

The `tern-convert` CLI accepts a HuggingFace model ID or local path, downloads the model, runs `TernaryConverter.convert()` to quantise eligible layers, writes the result as a .tern-model file, and optionally runs `TernModelReader.verify()` to confirm CRC32 integrity. The pipeline reports per-layer statistics (original size, packed size, sparsity) and summary compression ratio.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestTernaryConverter::test_convert_synthetic_model` | `tests/test_convert.py` | Full pipeline with synthetic model |
| `TestTernaryConverter::test_convert_stats_returned` | `tests/test_convert.py` | Statistics include expected fields |
| `TestTernaryConverter::test_compression_ratio` | `tests/test_convert.py` | Compression ratio reported correctly |
| `TestTernaryConverter::test_per_layer_stats` | `tests/test_convert.py` | Per-layer statistics available |
| `TestCLI::test_cli_help` | `tests/test_convert.py` | `tern-convert --help` exits cleanly |
| `TestCLI::test_cli_missing_output` | `tests/test_convert.py` | Missing output path produces error |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| TinyLlama .tern-model | 471.6 MB (8.4x compression) | Day 10, `benchmarks/bench_day10_pipeline.py` |
| BERT .tern-model | 30.9 MB (13.5x compression) | Day 11, `benchmarks/bench_day11_multi_model.py` |
| Models validated | 5 architectures, zero model-specific code | Day 11, `benchmarks/bench_day11_multi_model.py` |
| Pipeline end-to-end time | 7.5s (DistilGPT-2) to 212.7s (TinyLlama) | Day 11, `benchmarks/bench_day11_multi_model.py` |

### Reproducibility
```bash
pytest tests/test_convert.py -v
python -m terncore.convert distilgpt2 --output /tmp/distilgpt2.tern --verify
python -m terncore.convert bert-base-uncased --output /tmp/bert.tern --verify
```

---

## Patent 36 — Deterministic Execution

### Claim Summary

A method for ensuring bit-identical output from ternary neural network inference across multiple runs. Determinism is achieved through fixed random seeds, disabled CUDA benchmarking, disabled sampling during generation (`do_sample=False`), OpenMP static scheduling, and explicit `torch.manual_seed()` before each inference pass.

### Implementation

| Component | Location |
|-----------|----------|
| Deterministic inference | `engine/inference.py`, `TernaryInferenceEngine.infer()` (line 191) |
| Seed control | `torch.manual_seed(42)` in `infer()` |
| Deterministic eval | `arithmetic/linear.py`, `TernaryLinear._forward_eval()` (line 105) |
| Cached ternary weights | `TernaryLinear._cache_ternary_weights()` (line 143) |
| OpenMP static schedule | `csrc/ternary_avx2.c`, `#pragma omp parallel for schedule(static)` |
| Greedy generation | `do_sample=False` in all benchmark scripts |

### How It Works

`TernaryInferenceEngine.infer()` sets `torch.manual_seed(42)` before each forward pass and disables CUDA benchmarking. `TernaryLinear` caches ternary weights at the first eval forward call, ensuring identical quantisation across runs (no re-quantisation). The C+SIMD kernel uses OpenMP `schedule(static)` to assign identical loop iterations to identical threads. All benchmark scripts use `do_sample=False` to ensure greedy (deterministic) token selection.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestInferenceEngine::test_deterministic_inference` | `tests/test_stage1a.py` | Two infer() calls produce identical output |
| `TestTernaryLinear::test_deterministic_eval` | `tests/test_stage1a.py` | Eval forward is deterministic |
| `TestTernaryLinearAccel::test_deterministic_100_runs` | `tests/test_stage1b.py` | 100 consecutive runs produce identical output |
| `TestAccelIntegration::test_deterministic_model_inference` | `tests/test_stage1b.py` | Multi-layer accel model is deterministic |
| `TestMockLlamaConversion::test_deterministic_output` | `tests/test_phase3.py` | Mock LLaMA inference is deterministic |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Bit-identical across 100 runs | Verified | Stage 1B, `tests/test_stage1b.py` |
| Deterministic generation (DistilGPT-2) | `do_sample=False`, `manual_seed(42)` | Day 12, `benchmarks/bench_day12_performance.py` |
| OpenMP static scheduling | Consistent thread-to-iteration mapping | Phase 4, `csrc/ternary_avx2.c` |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestInferenceEngine::test_deterministic_inference -v
pytest tests/test_stage1b.py::TestTernaryLinearAccel::test_deterministic_100_runs -v
pytest tests/test_phase3.py::TestMockLlamaConversion::test_deterministic_output -v
```

---

## Patent 37 — Zero-Skip Acceleration

### Claim Summary

A method for accelerating ternary matrix operations by skipping multiply-accumulate operations for zero-valued weights. A bitmap-driven sparse kernel uses CTZ (count-trailing-zeros) bit-scan to iterate only over non-zero weight positions within each uint64 bitmap word, achieving linear speedup proportional to sparsity.

### Implementation

| Component | Location |
|-----------|----------|
| Sparse kernel | `csrc/sparse_skip.c`, `tern_sparse64_matvec_f32()` (line 98) |
| Sparse kernel header | `csrc/sparse_skip.h` |
| Bitmap generation | `sparse/__init__.py`, `generate_sparsity_bitmap()` (line 46) |
| Packed ternary kernel | `csrc/ternary_packed.c` |
| Python dispatch | `packed_ops.py` |

### How It Works

The C kernel `tern_sparse64_matvec_f32` processes weight rows in 64-trit chunks (two uint64 words: one for each bit of the 2-bit encoding). For each chunk, it loads the corresponding bitmap uint64 word and uses `__builtin_ctzll()` to find the position of the next set bit, processes that element (add or subtract based on sign bit), then clears the bit and repeats until the word is zero. This skips all zero elements without branch misprediction overhead.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestZeroSkipCorrectness::test_zero_skip_same_output` | `tests/test_sparsity.py` | Sparse kernel matches dense reference |
| `TestZeroSkipCorrectness::test_high_sparsity_correctness` | `tests/test_sparsity.py` | Correct at 90% sparsity |
| C test suite: `test_sparse_skip` | `csrc/test_sparse_skip.c` | C-level correctness + edge cases |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Speedup at 40% sparsity | 1.22x | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Speedup at 60% sparsity | 1.80x | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Speedup at 80% sparsity | 2.93x | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Speedup at 90% sparsity | 5.28x | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Break-even sparsity | ~35% | Day 9, `benchmarks/bench_day9_sparsity.py` |
| Typical ternary sparsity | 43-46% (1.2-1.4x) | Days 2, 11 |

### Reproducibility
```bash
pytest tests/test_sparsity.py::TestZeroSkipCorrectness -v
cd src/terncore/csrc && make test && cd ../../..
python benchmarks/bench_day9_sparsity.py
```

---

## Patent 38 — Configurable Multi-Precision

### Claim Summary

A method for configurable multi-precision execution in ternary neural networks, where different compute backends are selected at runtime based on hardware capabilities. CPUID detection determines available SIMD instruction sets (AVX2, AVX-512, NEON), and the appropriate kernel (AVX2 branchless mask-and-blend, NEON, or scalar fallback) is dispatched automatically. Additionally, the system supports dual backend paths: a PyTorch C++ extension (zero-copy, OpenMP-parallel) and a ctypes shared library (cross-platform fallback).

### Implementation

| Component | Location |
|-----------|----------|
| CPUID detection | `csrc/ternary_simd.h` |
| AVX2 kernel | `csrc/ternary_avx2.c`, `tern_packed_matvec_f32_avx2()` (line 83) |
| AVX2 matmul | `csrc/ternary_avx2.c`, `tern_packed_matmul_f32_avx2()` (line 190) |
| NEON kernel | `csrc/ternary_neon.c` |
| Scalar kernel | `csrc/ternary_matmul.c`, `tern_matvec_f32()` (line 52) |
| Backend selection | `accel/__init__.py`, `TernaryLinearAccel` (line 292) |
| Accel info | `accel/__init__.py`, `get_acceleration_info()` (line 243) |
| Torch C++ extension | `csrc/torch_bindings.cpp` |
| ctypes bindings | `csrc/bindings.c` |

### How It Works

`ternary_simd.h` uses inline assembly CPUID checks to detect AVX2, AVX-512, SSE4.1, and NEON at runtime. The dispatch function selects the fastest available kernel: AVX2 branchless mask-and-blend on x86, NEON on ARM, or scalar fallback. `TernaryLinearAccel` provides a Python-level abstraction with automatic backend selection: the PyTorch C++ extension path (zero-copy tensor access, OpenMP parallelism) is preferred, falling back to ctypes if the extension isn't compiled.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestSIMDAcceleration::test_simd_detection_reports_capability` | `tests/test_stage1b.py` | CPUID detection returns valid flags |
| `TestSIMDAcceleration::test_simd_matches_scalar_small` | `tests/test_stage1b.py` | SIMD matches scalar at small sizes |
| `TestSIMDAcceleration::test_simd_matches_scalar_large` | `tests/test_stage1b.py` | SIMD matches scalar at large sizes |
| `TestSIMDAcceleration::test_simd_deterministic` | `tests/test_stage1b.py` | SIMD output is deterministic |
| `TestAccelInfo::test_is_accelerated_returns_bool` | `tests/test_stage1b.py` | Acceleration detection works |
| `TestAccelInfo::test_info_structure` | `tests/test_stage1b.py` | Info dict has expected fields |
| `TestAccelInfo::test_simd_keys` | `tests/test_stage1b.py` | SIMD capability keys present |
| C test suite: `test_simd` | `csrc/test_simd.c` | AVX2/NEON kernel correctness |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| AVX2 detected on i9-9900K | Yes (CPUID) | Phase 4 |
| SIMD speedup at 2048x2048 | 2.45x vs BLAS | Phase 4, `benchmarks/bench_stage1b.py` |
| Torch ext vs ctypes speedup | 6.0-6.9x | Phase 4, `benchmarks/bench_stage1b.py` |
| FP32 tok/s (DistilGPT-2) | 62.7 | Day 12, `benchmarks/bench_day12_performance.py` |
| FP32 tok/s (GPT-2) | 37.0 | Day 12, `benchmarks/bench_day12_performance.py` |
| FP32 tok/s (TinyLlama) | 5.8 | Day 12, `benchmarks/bench_day12_performance.py` |

### Reproducibility
```bash
pytest tests/test_stage1b.py::TestSIMDAcceleration -v
pytest tests/test_stage1b.py::TestAccelInfo -v
cd src/terncore/csrc && make test && cd ../../..
python benchmarks/bench_stage1b.py
```

---

## Patent 39 — Packed Ternary Memory Format

### Claim Summary

A ternary-native memory format that stores neural network weights as 2-bit packed values in uint8 arrays, achieving 16x compression versus FP32. The encoding (00=0, 01=+1, 10=-1) stores 4 weights per byte with LSB-first packing. Combined with a 1-bit sparsity bitmap, the effective encoding is 3 bits/weight (10.7x vs FP32). The format supports both storage (.tern-model files) and runtime inference (`PackedTernaryLinear`).

### Implementation

| Component | Location |
|-----------|----------|
| Pack function | `sparse/__init__.py`, `pack_ternary_weights()` (line 62) |
| Unpack function | `sparse/__init__.py`, `unpack_ternary_weights()` (line 116) |
| PackedTernaryLinear | `packed_linear.py`, `PackedTernaryLinear` (line 49) |
| Packed forward pass | `PackedTernaryLinear.forward()` (line 236) |
| .tern-model packing | `tern_model.py`, `TernModelWriter.pack_ternary()` (line 359) |
| C packed kernel | `csrc/ternary_packed.c` |
| C packed header | `csrc/ternary_packed.h` |

### How It Works

`pack_ternary_weights()` maps ternary values to 2-bit codes (0->00, +1->01, -1->10) and packs 4 values per uint8, LSB-first. `unpack_ternary_weights()` reverses the process, extracting 2-bit codes and mapping back to float {-1, 0, +1}. `PackedTernaryLinear` stores the packed uint8 array as a registered buffer and unpacks to float on each forward call. The C kernel `ternary_packed.c` provides hardware-accelerated unpacking and matmul directly from the packed format.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestSparse::test_pack_unpack_roundtrip` | `tests/test_stage1a.py` | Pack then unpack recovers original values |
| `TestSparse::test_pack_unpack_large` | `tests/test_stage1a.py` | Round-trip at 1024x1024 scale |
| `TestSparse::test_compression_ratio` | `tests/test_stage1a.py` | 4:1 byte compression |
| `TestPackTernary::test_pack_ternary_basic` | `tests/test_tern_model.py` | Basic packing correctness |
| `TestPackTernary::test_pack_ternary_roundtrip` | `tests/test_tern_model.py` | Writer packing round-trip |
| `TestPackTernary::test_pack_ternary_all_zeros` | `tests/test_tern_model.py` | All-zero tensor packs correctly |
| `TestPackedTernaryLinear::test_memory_footprint` | `tests/test_packed_linear.py` | Uses uint8 not float32 storage |
| C test suite: `test_packed` | `csrc/test_packed.c` | C-level pack/unpack correctness |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Storage compression | 10.7x (3 bits/weight effective) | Day 8, `benchmarks/bench_day8_packing.py` |
| In-memory compression (BERT) | 3.4x | Day 12, `benchmarks/bench_day12_performance.py` |
| In-memory compression (GPT-2) | 2.5x | Day 12, `benchmarks/bench_day12_performance.py` |
| 2-bit encoding per weight | 4 weights/byte | Day 8, `benchmarks/bench_day8_packing.py` |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestSparse -v
pytest tests/test_packed_linear.py::TestPackedTernaryLinear::test_memory_footprint -v
cd src/terncore/csrc && make test && cd ../../..
python benchmarks/bench_day8_packing.py
```

---

## Patent 40 — Sensitivity-Guided Layer Protection

### Claim Summary

A method for protecting precision-critical layers from ternary quantisation based on sensitivity analysis. Each layer is individually ternarised while all others remain in FP32, and perplexity impact is measured. Layers are classified into a taxonomy (catastrophic, high, moderate, tolerant) and protection rules are generated. Pattern-based protection ("embed", "layernorm", "lm_head") supplements data-driven sensitivity results.

### Implementation

| Component | Location |
|-----------|----------|
| Sensitivity analyser | `arithmetic/quantizer.py`, `SensitivityAnalyzer` (line 145) |
| Per-layer analysis | `SensitivityAnalyzer.analyze_layer()` (line 171) |
| Full-model analysis | `SensitivityAnalyzer.analyze_model()` (line 211) |
| Protection logic | `engine/inference.py`, `_should_protect()` (line 233) |
| Converter protection | `convert.py`, within `TernaryConverter.convert()` (line 144) |
| Custom protection | `TernaryConverter` accepts custom protection patterns |

### How It Works

`SensitivityAnalyzer.analyze_layer()` replaces a single layer with its ternary equivalent, runs a perplexity evaluation, and records the ratio vs baseline. `analyze_model()` iterates over all eligible layers, producing a sorted sensitivity ranking. `_should_protect()` matches layer names against known critical patterns (embedding, normalisation, output head). The converter combines pattern-based protection (always applied) with sensitivity-derived recommendations (user-configurable) to determine which layers to protect.

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `TestSensitivityAnalyzer::test_analyze_layer` | `tests/test_stage1a.py` | Single-layer analysis returns valid ratio |
| `TestSensitivityAnalyzer::test_analyze_model` | `tests/test_stage1a.py` | Full-model analysis returns per-layer results |
| `TestTernaryConverter::test_protection_patterns` | `tests/test_convert.py` | Name patterns match correctly |
| `TestTernaryConverter::test_protection_always_protects_critical` | `tests/test_convert.py` | Embeddings/LN/head always protected |
| `TestTernaryConverter::test_custom_protection_patterns` | `tests/test_convert.py` | Custom patterns work |
| `TestTernaryConverter::test_transformer_protection` | `tests/test_convert.py` | Transformer-style model protection |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| Layers analysed (TinyLlama) | 155 | Day 2, `benchmarks/eval_sensitivity.py` |
| Tolerant layers (<1.1x) | 135 (87.1%) | Day 2, `benchmarks/eval_sensitivity.py` |
| Catastrophic outlier | `layers.2.mlp.down_proj` (9,609x) | Day 2, `benchmarks/eval_sensitivity.py` |
| Cross-architecture (GPT-2) | 34/49 safe (69.4%) | Day 11, `benchmarks/bench_day11_multi_model.py` |
| v_proj most tolerant type | Avg ratio 1.002x | Day 5, `benchmarks/analyse_weights.py` |
| Analysis time (TinyLlama) | 10,955s | Day 2, `benchmarks/eval_sensitivity.py` |

### Reproducibility
```bash
pytest tests/test_stage1a.py::TestSensitivityAnalyzer -v
pytest tests/test_convert.py::TestTernaryConverter::test_protection_patterns -v
python benchmarks/eval_sensitivity.py
```

---

## Appendix A: File Manifest with Patent Tags

All source files in `src/terncore/` with the patents they implement.

### Python Source

| File | Purpose | Patents |
|------|---------|---------|
| `__init__.py` | Package init | -- |
| `arithmetic/__init__.py` | Arithmetic subpackage | -- |
| `arithmetic/quantizer.py` | TernaryQuantizer, SensitivityAnalyzer | 1, 4, 40 |
| `arithmetic/linear.py` | TernaryLinear, TernaryConv2d | 1, 36 |
| `ste.py` | STEQuantize, TernaryLinearSTE | 1, 4 |
| `ste_trainer.py` | STE training loop | 4 |
| `mixed_precision.py` | Mixed-precision configs | 4 |
| `engine/__init__.py` | Engine subpackage | -- |
| `engine/inference.py` | TernaryInferenceEngine | 10, 36, 40 |
| `convert.py` | TernaryConverter CLI | 10, 11, 12, 40 |
| `tern_model.py` | TernModelWriter, TernModelReader | 6, 8, 39 |
| `packed_linear.py` | PackedTernaryLinear | 7, 8, 39 |
| `packed_ops.py` | Packed matmul dispatch | 37, 39 |
| `sparse/__init__.py` | Packing, bitmap, block analysis | 7, 8, 9, 39 |
| `accel/__init__.py` | TernaryLinearAccel | 38 |
| `hf_loader/__init__.py` | HuggingFace model loading | 12 |
| `memory/__init__.py` | Memory profiling | -- |
| `model_loader/__init__.py` | Legacy .tern-model v1 | 6 |

### C Source

| File | Purpose | Patents |
|------|---------|---------|
| `csrc/ternary_matmul.c` | Scalar ternary matmul | 1, 38 |
| `csrc/ternary_matmul.h` | Scalar kernel header | 1, 38 |
| `csrc/ternary_avx2.c` | AVX2 SIMD kernel | 36, 38 |
| `csrc/ternary_neon.c` | ARM NEON kernel | 38 |
| `csrc/ternary_packed.c` | Packed 2-bit kernel | 39 |
| `csrc/ternary_packed.h` | Packed kernel header | 39 |
| `csrc/ternary_simd.h` | CPUID detection, dispatch | 38 |
| `csrc/sparse_skip.c` | Bitmap sparse kernel (CTZ) | 7, 9, 37 |
| `csrc/sparse_skip.h` | Sparse kernel header | 7, 9, 37 |
| `csrc/bindings.c` | ctypes bindings | 38 |
| `csrc/torch_bindings.cpp` | PyTorch C++ extension | 38 |

### C Test Files

| File | Purpose | Patents Tested |
|------|---------|----------------|
| `csrc/test_matmul.c` | Scalar kernel tests | 1, 38 |
| `csrc/test_packed.c` | Pack/unpack tests | 39 |
| `csrc/test_simd.c` | AVX2/NEON tests | 38 |
| `csrc/test_sparse_skip.c` | Sparse skip tests | 7, 9, 37 |

---

## Appendix B: Test-to-Patent Coverage Matrix

Each row is a test class; columns indicate which patents it provides evidence for.

| Test Class | File | P1 | P4 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P36 | P37 | P38 | P39 | P40 |
|------------|------|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
| TestTernaryQuantizer | test_stage1a.py | x | | | | | | | | | | | | | |
| TestTernaryLinear | test_stage1a.py | x | | | | | | | | | x | | | | |
| TestTernaryConv2d | test_stage1a.py | x | | | | | | | | | | | | | |
| TestSparse | test_stage1a.py | | | | x | x | | | | | | | | x | |
| TestInferenceEngine | test_stage1a.py | | | | | | | x | | | x | | | | |
| TestSensitivityAnalyzer | test_stage1a.py | | x | | | | | | | | | | | | x |
| TestMemoryProfile | test_stage1a.py | | | | | | | | | | | | | | |
| TestTernModel | test_stage1a.py | | | x | | | | | | | | | | | |
| TestIntegration | test_stage1a.py | x | | | | | | x | | | | | | | |
| TestTernaryLinearAccel | test_stage1b.py | | | | | | | | | | x | | x | | |
| TestAccelFallback | test_stage1b.py | | | | | | | | | | | | x | | |
| TestAccelInfo | test_stage1b.py | | | | | | | | | | | | x | | |
| TestAccelIntegration | test_stage1b.py | | | | | | | | | | x | | x | | |
| TestSIMDAcceleration | test_stage1b.py | | | | | | | | | | | | x | | |
| TestMockLlamaConversion | test_phase3.py | | | | | | | x | | | x | | | | |
| TestMockLlamaAccel | test_phase3.py | | | | | | | | | | x | | x | | |
| TestFreeOriginalWeights | test_phase3.py | | | | | | | | | | | | | | |
| TestHFLoaderImport | test_phase3.py | | | | | | | | | x | | | | | |
| TestTinyLlamaIntegration | test_phase3.py | | | | | | | | | x | x | | | | |
| TestTernaryConverter | test_convert.py | | | | | | | x | | x | | | | | x |
| TestCLI | test_convert.py | | | | | | | | | x | | | | | |
| TestConv1DSupport | test_convert.py | | | | | | | | x | | | | | | |
| TestPackedTernaryLinear | test_packed_linear.py | | | | | x | | | | | | | | x | |
| TestPackedOps | test_packed_linear.py | | | | | x | | | | | | | | | |
| TestModelConversion | test_packed_linear.py | | | | | x | | | | | | | | | |
| TestTernModelReaderPacked | test_packed_linear.py | | | | | x | | | | | | | | x | |
| TestBitmapCaching | test_sparsity.py | | | | x | | | | | | | | | | |
| TestBlockSparsity | test_sparsity.py | | | | | | x | | | | | | | | |
| TestZeroSkipCorrectness | test_sparsity.py | | | | | | | | | | | x | | | |
| TestPackTernary | test_tern_model.py | | | x | | | | | | | | | | x | |
| TestSparsityBitmap | test_tern_model.py | | | x | x | | | | | | | | | | |
| TestWriteSingleLayer | test_tern_model.py | | | x | | | | | | | | | | | |
| TestWriteMixedPrecision | test_tern_model.py | | | x | | | | | | | | | | | |
| TestAlignment | test_tern_model.py | | | x | | | | | | | | | | | |
| TestHeaderMagic | test_tern_model.py | | | x | | | | | | | | | | | |
| TestManifestReadable | test_tern_model.py | | | x | | | | | | | | | | | |
| TestFileIntegrity | test_tern_model.py | | | x | | | | | | | | | | | |
| TestFooterMagic | test_tern_model.py | | | x | | | | | | | | | | | |
| TestHeaderSize | test_tern_model.py | | | x | | | | | | | | | | | |
| TestFileSizeConsistency | test_tern_model.py | | | x | | | | | | | | | | | |
| TestRandomAccess | test_tern_model.py | | | x | | | | | | | | | | | |
| TestBiasHandling | test_tern_model.py | | | x | | | | | | | | | | | |
| TestReconstructTernaryLayer | test_tern_model.py | | | x | | | | | | | | | | | |
| TestReconstructFP16Layer | test_tern_model.py | | | x | | | | | | | | | | | |
| TestReconstructAllMixed | test_tern_model.py | | | x | | | | | | | | | | | |
| TestRoundTripLogitsSynthetic | test_tern_model.py | | | x | | | | | | | | | | | |
| TestLazyAPI | test_tern_model.py | | | x | | | | | | | | | | | |
| TestLoadAsModel | test_tern_model.py | | | x | | | | | | | | | | | |
| C: test_matmul | csrc/test_matmul.c | x | | | | | | | | | | | x | | |
| C: test_packed | csrc/test_packed.c | | | | | | | | | | | | | x | |
| C: test_simd | csrc/test_simd.c | | | | | | | | | | | | x | | |
| C: test_sparse_skip | csrc/test_sparse_skip.c | | | | x | | x | | | | | x | | | |

**Coverage summary:**

| Patent | Test Classes | Total Tests |
|--------|-------------|-------------|
| Patent 1 | 6 | 22 |
| Patent 4 | 2 | 4 |
| Patent 6 | 18 | 28 |
| Patent 7 | 5 | 14 |
| Patent 8 | 5 | 14 |
| Patent 9 | 2 | 6 |
| Patent 10 | 4 | 14 |
| Patent 11 | 1 | 3 |
| Patent 12 | 5 | 10 |
| Patent 36 | 7 | 10 |
| Patent 37 | 2 | 3 |
| Patent 38 | 7 | 14 |
| Patent 39 | 5 | 11 |
| Patent 40 | 3 | 8 |

All 14 demonstrated patents have test coverage. 166 Python tests + 53 C tests = 219 total tests passing.

---

*Generated 2026-02-26. All line numbers reference commit `bdb4f84`. Synapticode Co., Ltd.*
