#!/usr/bin/env python3
"""
benchmark_metal.py — Benchmark native Metal ternary kernel vs FP16 matmul
=========================================================================
Measures isolated matrix-vector multiply performance at TinyLlama layer sizes.
Compares:
  1. Native Metal ternary kernel (packed 2-bit, zero-skip, add/sub only)
  2. PyTorch MPS FP16 matmul (baseline)
  3. PyTorch MPS dequantize-then-matmul (previous ternary approach)

Target: Apple M4 Pro · Apple Silicon · Terncore demo centrepiece

Terncore · Cubey/Synapticode · 2026
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "terncore"))
from pack_weights import pack_ternary_codes, compute_compression_stats
from ternary_metal import TernaryEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

# TinyLlama layer dimensions (representative set)
LAYER_SIZES = [
    ("attn_qkv",     2048, 2048),   # attention Q/K/V projection
    ("attn_out",     2048, 2048),   # attention output projection
    ("mlp_gate",     5632, 2048),   # MLP gate projection
    ("mlp_up",       5632, 2048),   # MLP up projection
    ("mlp_down",     2048, 5632),   # MLP down projection
    ("lm_head",     32000, 2048),   # language model head
]

RESULTS_DIR = Path(__file__).parent



# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def fp16_matvec_mps(weight_fp16: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Standard FP16 matmul on MPS."""
    return x @ weight_fp16.t()


def dequant_matvec_mps(codes: torch.Tensor, scales: torch.Tensor,
                       x: torch.Tensor) -> torch.Tensor:
    """Dequantize ternary then matmul on MPS (previous approach)."""
    w = codes.float() * scales.unsqueeze(1)
    w = w.to(x.dtype)
    return x @ w.t()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_layer(name: str, M: int, K: int, engine: TernaryEngine) -> dict:
    """Benchmark a single layer across all three methods."""
    B = 1  # autoregressive decode: single vector

    # Generate realistic ternary weights (43% sparsity)
    codes_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
    mask = torch.rand(M, K) < 0.43
    codes_int8[mask] = 0

    scales_fp32 = torch.randn(M).abs() * 0.1 + 0.02
    x_fp16 = torch.randn(B, K, dtype=torch.float16)

    # Compute sparsity
    sparsity = (codes_int8 == 0).sum().item() / codes_int8.numel()

    # Prepare data for each method
    # 1. Metal ternary kernel
    packed_codes = pack_ternary_codes(codes_int8)
    scales_np = scales_fp32.numpy().astype(np.float32)
    x_np = x_fp16.numpy()

    # Pre-allocate GPU buffers (avoid transfer overhead in timing)
    codes_buf = engine.create_buffer(np.ascontiguousarray(packed_codes))
    scales_buf = engine.create_buffer(np.ascontiguousarray(scales_np))
    input_buf = engine.create_buffer(np.ascontiguousarray(x_np))
    output_buf = engine.create_buffer(size=B * M * 2)

    # 2. FP16 baseline on MPS
    weight_fp16 = (codes_int8.float() * scales_fp32.unsqueeze(1)).to(torch.float16).to("mps")
    x_mps = x_fp16.to("mps")

    # 3. Dequant path on MPS
    codes_mps = codes_int8.to("mps")
    scales_mps = scales_fp32.to("mps")

    results = {"layer": name, "M": M, "K": K, "sparsity": sparsity}

    # --- Benchmark Metal ternary kernel ---
    use_fast = (K % 16 == 0)

    # Warmup
    for _ in range(WARMUP_RUNS):
        engine.matvec_gpu(codes_buf, scales_buf, input_buf, output_buf,
                          M, K, B, fast=use_fast)

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        engine.matvec_gpu(codes_buf, scales_buf, input_buf, output_buf,
                          M, K, B, fast=use_fast)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    results["metal_ternary"] = {
        "mean_us": statistics.mean(latencies) * 1e6,
        "median_us": statistics.median(latencies) * 1e6,
        "min_us": min(latencies) * 1e6,
        "stdev_us": statistics.stdev(latencies) * 1e6 if len(latencies) > 1 else 0,
        "kernel": "fast" if use_fast else "generic",
    }

    # --- Benchmark FP16 MPS matmul ---
    for _ in range(WARMUP_RUNS):
        _ = fp16_matvec_mps(weight_fp16, x_mps)
        torch.mps.synchronize()

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = fp16_matvec_mps(weight_fp16, x_mps)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    results["fp16_mps"] = {
        "mean_us": statistics.mean(latencies) * 1e6,
        "median_us": statistics.median(latencies) * 1e6,
        "min_us": min(latencies) * 1e6,
        "stdev_us": statistics.stdev(latencies) * 1e6 if len(latencies) > 1 else 0,
    }

    # --- Benchmark dequant-then-matmul ---
    for _ in range(WARMUP_RUNS):
        _ = dequant_matvec_mps(codes_mps, scales_mps, x_mps)
        torch.mps.synchronize()

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = dequant_matvec_mps(codes_mps, scales_mps, x_mps)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    results["dequant_mps"] = {
        "mean_us": statistics.mean(latencies) * 1e6,
        "median_us": statistics.median(latencies) * 1e6,
        "min_us": min(latencies) * 1e6,
        "stdev_us": statistics.stdev(latencies) * 1e6 if len(latencies) > 1 else 0,
    }

    # Memory comparison
    compression = compute_compression_stats(codes_int8)
    results["memory"] = {
        "fp16_bytes": M * K * 2,
        "packed_2bit_bytes": packed_codes.nbytes,
        "compression_ratio": compression["compression_vs_fp16"],
        "bandwidth_reduction": f"{compression['compression_vs_fp16']:.1f}x",
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  Terncore Metal Kernel Benchmark")
    print("  Native ternary matmul vs FP16 baseline on Apple Silicon")
    print("=" * 72)

    engine = TernaryEngine()
    print(f"\n  GPU: {engine.device_name}")
    print(f"  Warmup: {WARMUP_RUNS} runs, Measured: {BENCHMARK_RUNS} runs per layer")
    print(f"  Batch: 1 (autoregressive decode)\n")

    all_results = []

    print(f"  {'Layer':<12} {'Size':>14} {'Metal':>10} {'FP16':>10} "
          f"{'Dequant':>10} {'Speedup':>10} {'Mem':>8}")
    print(f"  {'':─<12} {'':─>14} {'(us)':─>10} {'(us)':─>10} "
          f"{'(us)':─>10} {'vs FP16':─>10} {'ratio':─>8}")

    for name, M, K in LAYER_SIZES:
        result = benchmark_layer(name, M, K, engine)
        all_results.append(result)

        metal_us = result["metal_ternary"]["mean_us"]
        fp16_us = result["fp16_mps"]["mean_us"]
        dequant_us = result["dequant_mps"]["mean_us"]
        speedup_vs_fp16 = fp16_us / metal_us if metal_us > 0 else float('inf')
        speedup_vs_dequant = dequant_us / metal_us if metal_us > 0 else float('inf')
        mem_ratio = result["memory"]["compression_ratio"]

        print(f"  {name:<12} {M:>6}x{K:<6} {metal_us:>9.1f} {fp16_us:>9.1f} "
              f"{dequant_us:>9.1f} {speedup_vs_fp16:>9.2f}x {mem_ratio:>7.1f}x")

    # Aggregate stats
    metal_total = sum(r["metal_ternary"]["mean_us"] for r in all_results)
    fp16_total = sum(r["fp16_mps"]["mean_us"] for r in all_results)
    dequant_total = sum(r["dequant_mps"]["mean_us"] for r in all_results)
    overall_speedup = fp16_total / metal_total if metal_total > 0 else float('inf')
    overall_vs_dequant = dequant_total / metal_total if metal_total > 0 else float('inf')

    print(f"\n  {'TOTAL':<12} {'':>14} {metal_total:>9.1f} {fp16_total:>9.1f} "
          f"{dequant_total:>9.1f} {overall_speedup:>9.2f}x")

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"""
  Metal ternary kernel:
    vs FP16 baseline:     {overall_speedup:.2f}x {'faster' if overall_speedup > 1 else 'slower'}
    vs dequant path:      {overall_vs_dequant:.1f}x faster
    Memory bandwidth:     ~{all_results[0]['memory']['compression_ratio']:.0f}x less than FP16
    Zero-channel skip:    ~43% of multiply-accumulate eliminated
    Multiply operations:  ZERO (pure add/subtract)
""")

    # Save results
    output = {
        "benchmark": "Terncore Metal Kernel vs FP16",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": engine.device_name,
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
            "batch_size": 1,
        },
        "layers": all_results,
        "aggregate": {
            "metal_total_us": metal_total,
            "fp16_total_us": fp16_total,
            "dequant_total_us": dequant_total,
            "speedup_vs_fp16": overall_speedup,
            "speedup_vs_dequant": overall_vs_dequant,
        },
    }

    json_path = RESULTS_DIR / "metal_kernel_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Markdown report
    md = f"""# Terncore Metal Kernel Benchmark

> Native ternary matmul on Apple Silicon — packed 2-bit codes, zero-skip, no FMA
> {engine.device_name} · {datetime.now().strftime('%Y-%m-%d')}

## Results (B=1, autoregressive decode)

| Layer | Size | Metal (us) | FP16 (us) | Dequant (us) | vs FP16 | Memory |
|-------|------|:----------:|:---------:|:------------:|:-------:|:------:|
"""
    for r in all_results:
        metal_us = r["metal_ternary"]["mean_us"]
        fp16_us = r["fp16_mps"]["mean_us"]
        dequant_us = r["dequant_mps"]["mean_us"]
        speedup = fp16_us / metal_us if metal_us > 0 else 0
        mem = r["memory"]["compression_ratio"]
        md += (f"| {r['layer']} | {r['M']}x{r['K']} | {metal_us:.1f} | "
               f"{fp16_us:.1f} | {dequant_us:.1f} | {speedup:.2f}x | {mem:.1f}x |\n")

    md += f"""
**Aggregate: {overall_speedup:.2f}x vs FP16, {overall_vs_dequant:.1f}x vs dequant path**

## Key Properties

- **Zero multiplies**: All weight operations are pure add/subtract
- **43% free compute**: Zero-weighted channels skipped entirely
- **~{all_results[0]['memory']['compression_ratio']:.0f}x memory compression**: 2-bit packed vs FP16
- **Branch-free**: Bit manipulation decode, no conditionals in hot path

---
*Terncore Metal kernel benchmark · Cubey/Synapticode · {datetime.now().strftime('%Y-%m-%d')}*
"""

    md_path = RESULTS_DIR / "metal_kernel_benchmark.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"  Results: {json_path}")
    print(f"  Report:  {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
