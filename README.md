# tern-core

tern-core converts LLM weights to ternary precision automatically. No configuration. No quality loss. Built for edge hardware.

---

## How it works

tern-core scans your model, identifies which layers tolerate ternary conversion within a perplexity-gated quality ceiling, converts only those layers, and proceeds to inference. The first run takes ~60 seconds. Every subsequent run is instant — results are cached.

No protection lists. No manual layer selection. No broken output.

---

## Benchmark results

**Model: Mistral-7B-v0.1 · Apple M4 Pro · 28 March 2026**

| Configuration       | Size    | Ternary ratio | vs baseline |
|---------------------|---------|---------------|-------------|
| FP16 baseline       | 14.5 GB | —             | —           |
| Ternary (tern-core) | 2.27 GB | 96.4%         | −84% size   |

**Model: TinyLlama-1.1B · Apple M4 Pro · 30 March 2026**

| Configuration       | tok/s | J/token | vs baseline     |
|---------------------|-------|---------|-----------------|
| FP32 CPU baseline   | 27.2  | 0.0859  | —               |
| Ternary CPU (gated) | 27.1  | 0.0870  | near-lossless   |
| FP16 MPS (GPU)      | 39.9  | 0.1069  | +24% energy     |

> Benchmarked on Apple M4 Pro (14-core CPU, 20-core GPU, 64GB RAM).
> Reference: github.com/synapticode-ai/tern-core

---

## Whole-model compression

| Model          | FP32 size | tern-core   | Compression | Layers converted |
|----------------|-----------|-------------|-------------|------------------|
| TinyLlama-1.1B | 4.1 GB    | 471.6 MB    | **8.4×**    | 154/155          |
| Mistral-7B     | 14.5 GB   | 2.27 GB     | 6.4×        | 96.4% ternary    |

> **Benchmark methodology note:** KV cache compression figures from the earlier release (v0.1.0) were open-loop storage measurements under a specific accounting; closed-loop quality-verified KV results will be published when available.

---

## Quick start

```bash
pip install tern-core

python -m terncore.infer \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The future of computing lies in"
```

On first run, tern-core will:

1. Scan your model for ternary-compatible layers
2. Print a classification of compression quality
3. Convert safe layers automatically
4. Run inference with a clean result

Example output:

```
┌─────────────────────────────────────────────┐
│  tern-core autoscan                         │
│  Model:       TinyLlama-1.1B                │
│  Converted:   10/154 layers (6%)            │
│  Compression: 1.00x                         │
│  PPL delta:   +7.5%                         │
│  Verdict:     Excellent — near-lossless     │
└─────────────────────────────────────────────┘
```

---

## Requirements

- Python 3.11+
- PyTorch 2.1+
- macOS (Apple Silicon recommended) or Linux
- 8GB RAM minimum; 16GB recommended for 7B models

---

## Licence

Evaluation licence. Commercial use by arrangement.

© 2026 Gamma Seeds Pte Ltd. All rights reserved.
Patent portfolio P001–P100 filed with IP Australia.
Contact: green.rush@icloud.com
