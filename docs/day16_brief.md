# Day 16 Brief — Patent-Code Mapping

## Context
Days 1-15 complete. Block 4 continues. README clean, TEA template built, evidence compiled. Now create the explicit link between patent claims and source code — the document that proves reduction to practice.

**Hardware:** 2019 iMac i9-9900K, 16GB DDR4.
**M4 Pro Status:** Mac Mini M4 Pro 12/16 64GB 1TB ordered from Apple AU. Pickup Brisbane City store ~March 30, 2026. M2 Ultra Mac Studio (64GB, 800 GB/s) offer pending — may arrive this week.
**Location:** Brisbane, Queensland, Australia.
**Doctrine:** No new computation. Read source, map to patents, write.
**Sprint goal:** All documentation prepared to Apple Core Quality engineering standards.

## The Standard

This document serves three audiences simultaneously:

1. **IP Australia** — For provisional patent filings. Maps each claim to specific code implementing it. Establishes reduction to practice with file paths, function names, line numbers, and dated benchmark results.

2. **Apple evaluation** — Shows the patent portfolio isn't vapourware. Every claim has working code behind it. An engineer can follow the mapping from patent claim → source file → test → benchmark result in a single reading.

3. **KSGC reviewers** — Demonstrates that a 56-patent portfolio (68 total including pending) has real implementation backing it. Not theoretical — built, tested, measured.

## Today's Deliverables

### `docs/PATENT_CODE_MAP.md` — The Master Mapping

Links each demonstrated patent to specific source files, functions, tests, and sprint results.

## Structure

```markdown
# Patent-to-Code Mapping

## How to Read This Document
- Each patent section lists: claim summary, implementing source, key functions,
  test coverage, and experimental evidence from the sprint.
- Line numbers reference commit {hash} (Day 15).
- All results reproducible via commands in benchmarks/EVIDENCE_PACKAGE.md.

## Patent 1 — Ternary Weight Encoding
### Claim Summary
### Implementation
### Tests
### Evidence

## Patent 4 — Progressive Compression via STE
...

## Patent 6 — Ternary Model Storage Format
## Patent 7 — Sparsity Bitmap and Zero-Skip
## Patent 8 — Packed 2-Bit Weight Storage
## Patent 9 — Block-Level Sparsity Analysis

## Patent 10 — Inference Engine Architecture
## Patent 11 — Conv1D Layer Support
## Patent 12 — Automated Conversion Pipeline

## Patent 36 — Deterministic Execution
## Patent 37 — Zero-Skip Acceleration
## Patent 38 — Configurable Multi-Precision
## Patent 39 — Packed Ternary Memory Format
## Patent 40 — Sensitivity-Guided Layer Protection

## Appendix: File Manifest with Patent Tags
## Appendix: Test-to-Patent Coverage Matrix
```

## Patent-to-Code Data Sources

Every mapping pulls from existing source. No new computation.

| Patent | Primary Source File | Key Function/Class | Test File | Sprint Day |
|--------|-------------------|-------------------|-----------|------------|
| Patent 1 | `arithmetic/quantizer.py` | `TernaryQuantizer.quantize()` | `test_stage1a.py` | Day 1 |
| Patent 4 | `ste.py` | `TernaryLinearSTE`, `STEQuantize` | `test_ste.py` | Day 4 |
| Patent 6 | `tern_model.py` | `TernModelWriter`, `TernModelReader` | `test_tern_model.py` | Days 6-7 |
| Patent 7 | `sparse/__init__.py` | `generate_sparsity_bitmap()` | `test_sparse.py` | Day 9 |
| Patent 8 | `packed_linear.py` | `PackedTernaryLinear` | `test_packed_linear.py` | Day 8 |
| Patent 9 | `sparse/__init__.py` | `analyze_block_sparsity()` | `test_sparse.py` | Day 9 |
| Patent 10 | `engine/inference.py` | `TernaryInferenceEngine.convert()` | `test_convert.py` | Day 10 |
| Patent 11 | `engine/inference.py` | `_convert_conv1d()` | `test_convert.py` | Day 11 |
| Patent 12 | `convert.py` | `main()` CLI pipeline | `test_convert.py` | Day 10 |
| Patent 36 | benchmarks/bench_day12 | `do_sample=False`, `manual_seed(42)` | Day 12 results | Day 12 |
| Patent 37 | `csrc/sparse_skip.c` | `ternary_sparse_matmul()` | C test suite | Day 9 |
| Patent 38 | benchmarks/bench_day12 | FP32/Ternary/Packed modes | Day 12 results | Day 12 |
| Patent 39 | `packed_linear.py` | `pack_weights()`, `unpack_weights()` | `test_packed_linear.py` | Day 8 |
| Patent 40 | `arithmetic/quantizer.py` | `SensitivityAnalyzer` | `test_stage1a.py` | Days 2, 5 |

## Per-Patent Section Format

Each patent entry follows this exact structure:

```markdown
## Patent {N} — {Title}

### Claim Summary
One paragraph describing what the patent claims, in plain technical language.

### Implementation

| Component | Location |
|-----------|----------|
| Primary source | `src/terncore/{file}` |
| Key class/function | `{ClassName}.{method}()` (line {N}) |
| Supporting modules | `src/terncore/{other files}` |
| CLI integration | `tern-convert` flag or mode |

### How It Works
2-3 sentences describing the implementation approach. Reference specific
code patterns (e.g., "threshold-based quantisation using adaptive delta
computed as threshold × mean(|W|)").

### Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| `test_{name}` | `tests/{file}` | {description} |

### Experimental Evidence

| Metric | Value | Source |
|--------|-------|--------|
| {metric} | {value} | Day {N}, `benchmarks/{file}` |

### Reproducibility
```bash
{exact command to reproduce the evidence}
```
```

## Implementation Order

1. **Get current commit hash and line numbers** (10 min) — grep key functions for exact line numbers at HEAD
2. **Write Patent 1 section** (10 min) — the reference example, most detailed
3. **Write Patents 4, 6-9** (20 min) — core algorithm patents
4. **Write Patents 10-12** (15 min) — pipeline and conversion patents
5. **Write Patents 36-40** (15 min) — Silicon Web series patents
6. **Write appendices** (15 min) — file manifest with patent tags, test coverage matrix
7. **Cross-reference** (10 min) — verify every line number, every metric, every file path
8. **Commit** (5 min)

## Exit Criteria
- [ ] `docs/PATENT_CODE_MAP.md` complete
- [ ] All 14 demonstrated patents mapped (1, 4, 6-12, 36-40)
- [ ] Every mapping has: claim summary, source file + line, tests, evidence, reproducibility command
- [ ] Line numbers verified against current HEAD
- [ ] File manifest appendix covers all src/terncore/ files with patent tags
- [ ] Test coverage matrix shows which tests cover which patents
- [ ] No broken file paths or function references
- [ ] 166+ tests still pass
- [ ] Commit pushed

## Time Budget
| Phase | Estimate |
|-------|----------|
| Line number extraction | 10 min |
| Patent 1 (reference section) | 10 min |
| Patents 4, 6-9 | 20 min |
| Patents 10-12 | 15 min |
| Patents 36-40 | 15 min |
| Appendices | 15 min |
| Cross-reference + verify | 10 min |
| Commit | 5 min |
| **Total** | **~1.5 hours** |

## What NOT To Do
- Do NOT modify any source code. Documentation only.
- Do NOT add patent numbers as comments in source files. That's Day 17 consideration.
- Do NOT include unpublished patent claim text. Use plain-language summaries only.
- Do NOT map patents that have no code implementation yet (e.g., Patent 68 in-band signaling — that's post-M4-Pro work).
- Do NOT include business projections or market analysis. Technical mapping only.
- Do NOT spend more than 2 hours. Good mapping today, refined on M4 Pro.

## What This Enables
- IP Australia filings: Direct evidence of reduction to practice for each claim
- KSGC application: "56-patent portfolio with 14 patents demonstrated in working code"
- Apple evaluation: Patent portfolio credibility — claims backed by implementation
- Rebellions outreach: Technical brief can reference specific patents for each capability
- Day 17: CI setup can tag test failures by affected patent
- Day 18: KSGC draft pulls patent-code mapping directly
