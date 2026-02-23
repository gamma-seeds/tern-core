# tern-core

**Ternary execution engine for NPU inference.**

CNS Synaptic™ by Synapticode Co., Ltd.

## What This Is

The foundational execution engine for ternary AI computing. Every neural network weight is reduced to three values: **{-1, 0, +1}**. All arithmetic is compare-and-add. No multiply-accumulate. Deterministic by design.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Convert a model to ternary
python -c "
import torch
from terncore import TernaryInferenceEngine

model = torch.nn.Sequential(
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),
)

engine = TernaryInferenceEngine()
report = engine.convert(model)
print(f'Converted {report.converted_layers}/{report.total_layers} layers')
print(f'Compression: {report.compression_ratio:.1f}x')

result = engine.infer(model, torch.randn(1, 64))
print(f'Inference: {result.latency_ms:.2f}ms')
"
```

## Architecture

```
terncore/
├── arithmetic/     ← Quantiser, TernaryLinear, TernaryConv2d (Patents 1-3)
├── engine/         ← Inference engine, auto-conversion (Patents 10-12)
├── sparse/         ← Sparsity bitmap, zero-skip, packing (Patents 7-9)
├── memory/         ← Memory profiling (Patent 3)
└── model_loader/   ← .tern-model format read/write (Patents 17, 22)
```

## Patents

This code implements technology covered by Synapticode Patents 1-9 (Foundation layer).

## Licence

Proprietary. All rights reserved. Synapticode Co., Ltd.
