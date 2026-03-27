#!/usr/bin/env python3
"""
bench_7b_ane.py — 7B-scale ANE benchmark: ternary vs FP16
==========================================================
Builds a representative linear stack matching Llama 2 7B dimensions
and benchmarks dense vs sparse ternary inference on ANE.

Llama 2 7B architecture:
  - hidden_size:       4096
  - intermediate_size: 11008
  - num_hidden_layers: 32
  - num_attention_heads: 32
  - head_dim:          128

Linear params per block:
  q_proj: 4096 × 4096  = 16.8M
  k_proj: 4096 × 4096  = 16.8M
  v_proj: 4096 × 4096  = 16.8M
  o_proj: 4096 × 4096  = 16.8M
  gate:   4096 × 11008  = 45.1M
  up:     4096 × 11008  = 45.1M
  down:   11008 × 4096  = 45.1M
  Total per block:       202.4M
  Total 32 blocks:       6,476M ≈ 6.5B params (linear layers only)

vs TinyLlama 1.1B (previous benchmark):
  hidden=2048, intermediate=5632, 22 blocks → 969M linear params

The gap should widen at 7B because:
  1. Larger matmuls amplify the ternary 2-bit advantage (memory-bound → compute-bound)
  2. Channel pruning removes proportionally more compute at higher dims
  3. ANE's fixed-width matrix engine benefits from power-of-2 aligned dims

Terncore · Cubey/Synapticode · 2026
"""

import gc
import json
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from terncore.sparse.channel_pruning import (
    prune_mlp_channels,
    prune_attention_channels,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 10
RUNS = 30  # fewer runs — 7B models are slower
NUM_BLOCKS = 32
HIDDEN = 4096
INTERMEDIATE = 11008
NUM_HEADS = 32
HEAD_DIM = HIDDEN // NUM_HEADS  # 128
INPUT_SHAPE = (1, 64, HIDDEN)   # batch=1, seq=64, hidden=4096

RESULTS_DIR = Path(__file__).parent
MODELS_DIR = Path(__file__).parent.parent / "output" / "coreml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 7B-scale model
# ---------------------------------------------------------------------------
class Block7B(nn.Module):
    """Single transformer block matching Llama 2 7B linear dimensions."""

    def __init__(self, hidden=HIDDEN, intermediate=INTERMEDIATE):
        super().__init__()
        # Attention (no GQA — Llama 2 7B uses full MHA)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)
        h = x + attn_out

        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)
        return h + mlp_out


class Stack7B(nn.Module):
    """Full 7B-scale linear stack."""

    def __init__(self, num_blocks=NUM_BLOCKS, hidden=HIDDEN, intermediate=INTERMEDIATE):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block7B(hidden, intermediate) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SparseBlock7B(nn.Module):
    """Pruned block with variable attention and MLP dimensions."""

    def __init__(self, hidden, attn_dim, intermediate):
        super().__init__()
        self.q_proj = nn.Linear(hidden, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(attn_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)
        h = x + attn_out
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)
        return h + mlp_out


class SparseStack7B(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_ternary(model, threshold=0.7):
    n_layers = 0
    total_params = 0
    total_zeros = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        abs_w = w.abs()
        mean_abs = abs_w.mean(dim=1, keepdim=True)
        delta = threshold * mean_abs

        codes = torch.zeros_like(w, dtype=torch.int8)
        codes[w > delta] = 1
        codes[w < -delta] = -1

        mask = codes != 0
        scales = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
        for i in range(w.shape[0]):
            sel = abs_w[i][mask[i]]
            scales[i] = sel.mean() if sel.numel() > 0 else mean_abs[i, 0]

        module.weight.data = (codes.float() * scales.unsqueeze(1)).to(w.dtype)
        zeros = (codes == 0).sum().item()
        total_zeros += zeros
        total_params += w.numel()
        n_layers += 1

    return {
        "n_layers": n_layers,
        "total_params": total_params,
        "total_zeros": total_zeros,
        "sparsity": total_zeros / total_params if total_params > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Structural pruning
# ---------------------------------------------------------------------------
def build_sparse_7b(dense: Stack7B, mlp_ratio=0.30, attn_ratio=0.20):
    """Build structurally pruned 7B model."""
    # Probe first block for pruned dimensions
    b0 = dense.blocks[0]
    _, _, _, mlp_s = prune_mlp_channels(b0.gate_proj, b0.up_proj, b0.down_proj, mlp_ratio)
    _, _, attn_s = prune_attention_channels(b0.q_proj, b0.o_proj, attn_ratio)

    pruned_mlp = mlp_s.pruned_out
    pruned_attn = attn_s.pruned_out

    blocks = []
    for db in dense.blocks:
        sb = SparseBlock7B(HIDDEN, pruned_attn, pruned_mlp)

        pg, pu, pd, _ = prune_mlp_channels(db.gate_proj, db.up_proj, db.down_proj, mlp_ratio)
        sb.gate_proj, sb.up_proj, sb.down_proj = pg, pu, pd

        pq, po, _ = prune_attention_channels(db.q_proj, db.o_proj, attn_ratio)
        sb.q_proj, sb.o_proj = pq, po

        sb.k_proj.weight = nn.Parameter(db.k_proj.weight.data.clone())
        sb.v_proj.weight = nn.Parameter(db.v_proj.weight.data.clone())
        blocks.append(sb)

    return SparseStack7B(blocks), {
        "mlp_intermediate": f"{INTERMEDIATE} → {pruned_mlp}",
        "attn_dim": f"{HIDDEN} → {pruned_attn}",
        "mlp_prune": mlp_ratio,
        "attn_prune": attn_ratio,
    }


# ---------------------------------------------------------------------------
# CoreML helpers
# ---------------------------------------------------------------------------
def convert_coreml(model, name, path):
    print(f"  Converting {name}...")
    model.eval()
    dummy = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model.float(), dummy, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=INPUT_SHAPE, dtype=np.float16)],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(str(path))
    mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)
    print(f"    {path.name} ({mb:.1f} MB)")
    return mlmodel


def palettize(mlmodel, path, nbits=2):
    print(f"  Palettizing {nbits}-bit...")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
    )
    result = palettize_weights(mlmodel, config)
    result.save(str(path))
    mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)
    print(f"    {path.name} ({mb:.1f} MB)")
    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench_coreml(path, label, cu):
    print(f"  [{label}]", end=" ", flush=True)
    mlmodel = ct.models.MLModel(str(path), compute_units=cu)
    inp = {"input": np.random.randn(*INPUT_SHAPE).astype(np.float16)}

    for _ in range(WARMUP):
        mlmodel.predict(inp)

    lats = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        mlmodel.predict(inp)
        lats.append(time.perf_counter() - t0)

    mean = statistics.mean(lats) * 1000
    mn = min(lats) * 1000
    std = statistics.stdev(lats) * 1000 if len(lats) > 1 else 0
    print(f"{mean:.2f} ms (min {mn:.2f}, std {std:.2f})")

    del mlmodel; gc.collect()
    return {"label": label, "mean_ms": mean, "min_ms": mn, "stdev_ms": std}


def bench_mps(model, label):
    print(f"  [{label}]", end=" ", flush=True)
    model = model.to(torch.float16).to("mps").eval()
    x = torch.randn(*INPUT_SHAPE, dtype=torch.float16, device="mps")

    for _ in range(WARMUP):
        with torch.no_grad():
            model(x)
        torch.mps.synchronize()

    lats = []
    for _ in range(RUNS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.mps.synchronize()
        lats.append(time.perf_counter() - t0)

    mean = statistics.mean(lats) * 1000
    mn = min(lats) * 1000
    std = statistics.stdev(lats) * 1000 if len(lats) > 1 else 0
    print(f"{mean:.2f} ms (min {mn:.2f}, std {std:.2f})")

    return {"label": label, "mean_ms": mean, "min_ms": mn, "stdev_ms": std}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_mb(path):
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  7B-Scale ANE Benchmark — Ternary vs FP16")
    print("  Llama 2 7B dimensions: 4096 hidden, 11008 intermediate, 32 blocks")
    print("=" * 72)

    hw = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"]
    ).decode().strip()
    print(f"\n  Hardware:  {hw}")
    print(f"  Blocks:    {NUM_BLOCKS}")
    print(f"  Hidden:    {HIDDEN}")
    print(f"  MLP dim:   {INTERMEDIATE}")
    print(f"  Input:     {INPUT_SHAPE} (seq=64)")
    print(f"  Runs:      {WARMUP} warmup, {RUNS} measured")

    R = {}  # all results

    # ==================================================================
    # Phase 1: FP16 baselines
    # ==================================================================
    print(f"\n{'─' * 72}")
    print("Phase 1: FP16 baselines (MPS + CoreML)")
    print(f"{'─' * 72}")

    model_fp16 = Stack7B().to(torch.float16)
    n_params = count_params(model_fp16)
    print(f"  Parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    R["mps_fp16"] = bench_mps(model_fp16, "MPS FP16")
    model_fp16 = model_fp16.cpu()
    torch.mps.empty_cache(); gc.collect()

    # CoreML FP16
    fp16_path = MODELS_DIR / "7b_fp16.mlpackage"
    model_fp16 = Stack7B().to(torch.float16)
    mlm = convert_coreml(model_fp16, "7B FP16", fp16_path)
    del model_fp16, mlm; gc.collect()

    R["coreml_fp16_all"] = bench_coreml(fp16_path, "CoreML FP16 (ALL)", ct.ComputeUnit.ALL)
    R["coreml_fp16_ane"] = bench_coreml(fp16_path, "CoreML FP16 (ANE)", ct.ComputeUnit.CPU_AND_NE)

    # ==================================================================
    # Phase 2: Dense ternary 2-bit
    # ==================================================================
    print(f"\n{'─' * 72}")
    print("Phase 2: Dense ternary 2-bit")
    print(f"{'─' * 72}")

    model_tern = Stack7B().to(torch.float16)
    qs = quantize_ternary(model_tern)
    print(f"  Sparsity: {qs['sparsity']:.1%} "
          f"({qs['total_zeros']:,} / {qs['total_params']:,} zeros)")

    # MPS ternary (same as FP16 path — weights are dequantized floats)
    R["mps_ternary"] = bench_mps(model_tern, "MPS Ternary")
    model_tern = model_tern.cpu()
    torch.mps.empty_cache(); gc.collect()

    # CoreML ternary → 2-bit
    model_tern = Stack7B().to(torch.float16)
    quantize_ternary(model_tern)

    tern_path = MODELS_DIR / "7b_ternary_2bit.mlpackage"
    mlm = convert_coreml(model_tern, "7B Ternary", tern_path)
    mlm = palettize(mlm, tern_path, nbits=2)
    del mlm; gc.collect()

    R["tern_2bit_all"] = bench_coreml(tern_path, "Ternary 2-bit (ALL)", ct.ComputeUnit.ALL)
    R["tern_2bit_ane"] = bench_coreml(tern_path, "Ternary 2-bit (ANE)", ct.ComputeUnit.CPU_AND_NE)
    R["tern_2bit_gpu"] = bench_coreml(tern_path, "Ternary 2-bit (GPU)", ct.ComputeUnit.CPU_AND_GPU)

    # ==================================================================
    # Phase 3: Sparse ternary — channel pruning
    # ==================================================================
    prune_configs = [
        (0.30, 0.20, "30% MLP / 20% attn"),
        (0.40, 0.30, "40% MLP / 30% attn"),
    ]

    for mlp_r, attn_r, label in prune_configs:
        tag = f"sparse_m{int(mlp_r*100)}_a{int(attn_r*100)}"
        print(f"\n{'─' * 72}")
        print(f"Phase 3: Sparse ternary — {label}")
        print(f"{'─' * 72}")

        sparse_model, prune_info = build_sparse_7b(model_tern, mlp_r, attn_r)
        sparse_model = sparse_model.to(torch.float16)
        sp = count_params(sparse_model)
        reduction = 1 - sp / n_params
        print(f"  Params: {n_params:,} → {sp:,} ({reduction:.1%} reduction)")
        print(f"  {prune_info['mlp_intermediate']}, {prune_info['attn_dim']}")

        sp_path = MODELS_DIR / f"7b_{tag}_2bit.mlpackage"
        mlm = convert_coreml(sparse_model, f"Sparse {label}", sp_path)
        mlm = palettize(mlm, sp_path, nbits=2)
        del sparse_model, mlm; gc.collect()

        R[f"{tag}_all"] = bench_coreml(sp_path, f"Sparse {label} (ALL)", ct.ComputeUnit.ALL)
        r = bench_coreml(sp_path, f"Sparse {label} (ANE)", ct.ComputeUnit.CPU_AND_NE)
        r["params"] = sp
        r["param_reduction"] = reduction
        r["model_mb"] = model_mb(sp_path)
        R[f"{tag}_ane"] = r

    del model_tern; gc.collect()

    # ==================================================================
    # Results
    # ==================================================================
    print(f"\n{'=' * 72}")
    print("  7B-SCALE ANE BENCHMARK RESULTS")
    print(f"  Llama 2 7B dimensions · {n_params/1e9:.2f}B linear params")
    print(f"{'=' * 72}\n")

    mps_ms = R["mps_fp16"]["mean_ms"]
    dense_ane_ms = R["tern_2bit_ane"]["mean_ms"]
    fp16_ane_ms = R["coreml_fp16_ane"]["mean_ms"]

    # Size table
    print("  Model sizes:")
    for p, label in [
        (fp16_path, "FP16 CoreML"),
        (tern_path, "Ternary 2-bit"),
    ]:
        print(f"    {label:<24} {model_mb(p):>8.1f} MB")
    for mlp_r, attn_r, label in prune_configs:
        tag = f"sparse_m{int(mlp_r*100)}_a{int(attn_r*100)}"
        p = MODELS_DIR / f"7b_{tag}_2bit.mlpackage"
        print(f"    Sparse {label:<18} {model_mb(p):>8.1f} MB")
    print()

    # Latency table
    hdr = f"  {'Backend':<40} {'Mean':>8} {'Min':>8} {'vs FP16':>8} {'vs Dense':>9}"
    print(hdr)
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")

    for key, r in R.items():
        mean = r["mean_ms"]
        mn = r["min_ms"]
        vs_fp16 = mps_ms / mean if mean > 0 else 0
        vs_dense = dense_ane_ms / mean if mean > 0 else 0
        print(f"  {r['label']:<40} {mean:>7.2f} {mn:>7.2f} "
              f"{vs_fp16:>7.2f}x {vs_dense:>8.2f}x")

    # Key comparisons
    best_sparse_key = min(
        (k for k in R if "sparse" in k and "ane" in k),
        key=lambda k: R[k]["mean_ms"], default=None
    )

    print(f"\n  {'─' * 72}")
    print(f"  KEY METRICS — 7B Scale")
    print(f"  {'─' * 72}")
    print(f"  MPS FP16 baseline:         {mps_ms:.2f} ms")
    print(f"  CoreML FP16 (ANE):         {fp16_ane_ms:.2f} ms  ({mps_ms/fp16_ane_ms:.2f}× vs MPS)")
    print(f"  Dense Ternary 2-bit (ANE): {dense_ane_ms:.2f} ms  ({mps_ms/dense_ane_ms:.2f}× vs MPS)")
    if best_sparse_key:
        best = R[best_sparse_key]
        bm = best["mean_ms"]
        print(f"  Best Sparse (ANE):         {bm:.2f} ms  "
              f"({mps_ms/bm:.2f}× vs MPS, {dense_ane_ms/bm:.2f}× vs dense)")

    print(f"\n  Compression:")
    print(f"    FP16 CoreML:     {model_mb(fp16_path):>8.1f} MB")
    print(f"    Ternary 2-bit:   {model_mb(tern_path):>8.1f} MB  "
          f"({model_mb(fp16_path)/model_mb(tern_path):.1f}× smaller)")
    if best_sparse_key:
        sp_tag = best_sparse_key.replace("_ane", "")
        sp_path = MODELS_DIR / f"7b_{sp_tag}_2bit.mlpackage"
        if sp_path.exists():
            print(f"    Best sparse:     {model_mb(sp_path):>8.1f} MB  "
                  f"({model_mb(fp16_path)/model_mb(sp_path):.1f}× smaller)")

    # Comparison with TinyLlama results
    print(f"\n  {'─' * 72}")
    print(f"  SCALE COMPARISON — 7B vs 270M (TinyLlama)")
    print(f"  {'─' * 72}")
    # TinyLlama numbers from previous benchmarks
    tl_dense_ms = 7.23
    tl_mps_ms = 26.22
    tl_sparse_ms = 4.68  # 40/30 prune
    print(f"  {'Metric':<30} {'TinyLlama':>12} {'Llama 7B':>12} {'Scale':>8}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8}")
    print(f"  {'Linear params':<30} {'969M':>12} {f'{n_params/1e9:.1f}B':>12} {'6.7×':>8}")
    print(f"  {'MPS FP16':<30} {f'{tl_mps_ms:.2f} ms':>12} {f'{mps_ms:.2f} ms':>12}")
    print(f"  {'Dense 2-bit ANE':<30} {f'{tl_dense_ms:.2f} ms':>12} {f'{dense_ane_ms:.2f} ms':>12}")
    print(f"  {'Dense 2-bit vs MPS':<30} {f'{tl_mps_ms/tl_dense_ms:.2f}×':>12} {f'{mps_ms/dense_ane_ms:.2f}×':>12}")
    if best_sparse_key:
        bm = R[best_sparse_key]["mean_ms"]
        print(f"  {'Best sparse ANE':<30} {f'{tl_sparse_ms:.2f} ms':>12} {f'{bm:.2f} ms':>12}")
        print(f"  {'Best sparse vs MPS':<30} {f'{tl_mps_ms/tl_sparse_ms:.2f}×':>12} {f'{mps_ms/bm:.2f}×':>12}")

    # ==================================================================
    # Save
    # ==================================================================
    output = {
        "benchmark": "7B-scale ANE — ternary vs FP16",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "architecture": {
            "model": "Llama 2 7B dimensions",
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "blocks": NUM_BLOCKS,
            "heads": NUM_HEADS,
            "linear_params": n_params,
        },
        "config": {
            "input_shape": list(INPUT_SHAPE),
            "warmup": WARMUP,
            "runs": RUNS,
        },
        "quantization": qs,
        "results": R,
    }

    json_path = RESULTS_DIR / "7b_ane_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    md = f"""# 7B-Scale ANE Benchmark — Ternary vs FP16

> Llama 2 7B dimensions: {HIDDEN} hidden, {INTERMEDIATE} intermediate, {NUM_BLOCKS} blocks
> {n_params/1e9:.2f}B linear params · {hw} · {datetime.now().strftime('%Y-%m-%d')}

## Results

| Backend | Mean (ms) | Min (ms) | vs MPS FP16 | vs Dense ANE |
|---------|:---------:|:--------:|:-----------:|:------------:|
"""
    for key, r in R.items():
        mean = r["mean_ms"]
        mn = r["min_ms"]
        vs = mps_ms / mean if mean > 0 else 0
        vd = dense_ane_ms / mean if mean > 0 else 0
        md += f"| {r['label']} | {mean:.2f} | {mn:.2f} | {vs:.2f}x | {vd:.2f}x |\n"

    md += f"""
## Model Sizes

| Format | Size | vs FP16 |
|--------|-----:|--------:|
| FP16 CoreML | {model_mb(fp16_path):.1f} MB | 1.0x |
| Ternary 2-bit | {model_mb(tern_path):.1f} MB | {model_mb(fp16_path)/max(model_mb(tern_path),1):.1f}x |
"""
    for mlp_r, attn_r, label in prune_configs:
        tag = f"sparse_m{int(mlp_r*100)}_a{int(attn_r*100)}"
        p = MODELS_DIR / f"7b_{tag}_2bit.mlpackage"
        if p.exists():
            md += f"| Sparse {label} | {model_mb(p):.1f} MB | {model_mb(fp16_path)/max(model_mb(p),1):.1f}x |\n"

    md += f"""
## Scale Comparison — 7B vs 270M

| Metric | TinyLlama (270M) | Llama 7B ({n_params/1e9:.1f}B) |
|--------|:----------------:|:----------------------------:|
| MPS FP16 | {tl_mps_ms:.2f} ms | {mps_ms:.2f} ms |
| Dense 2-bit ANE | {tl_dense_ms:.2f} ms | {dense_ane_ms:.2f} ms |
| Dense vs MPS | {tl_mps_ms/tl_dense_ms:.2f}x | {mps_ms/dense_ane_ms:.2f}x |
"""
    if best_sparse_key:
        bm = R[best_sparse_key]["mean_ms"]
        md += f"| Best sparse ANE | {tl_sparse_ms:.2f} ms | {bm:.2f} ms |\n"
        md += f"| Best sparse vs MPS | {tl_mps_ms/tl_sparse_ms:.2f}x | {mps_ms/bm:.2f}x |\n"

    md += f"""
## Method

Linear stack matching Llama 2 7B: 32 blocks × 7 linear layers = 224 matmuls.
Same methodology as TinyLlama benchmark — coremltools cannot convert full
transformer ops (RoPE/attention), so we benchmark the linear layers which
are the bottleneck and where ternary provides the speedup.

Channel pruning targets MLP intermediate (11008 → pruned) and attention
internal dimension (4096 → pruned). k/v projections kept at full size.

---
*7B ANE benchmark · Terncore · Cubey/Synapticode · {datetime.now().strftime('%Y-%m-%d')}*
"""
    md_path = RESULTS_DIR / "7b_ane_benchmark.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n  Results: {json_path}")
    print(f"  Report:  {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
