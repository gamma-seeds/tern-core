"""
Microbenchmark: TernaryLinearAccel (C+SIMD) vs TernaryLinear (pure PyTorch).

Compares eval-mode forward-pass latency across matrix sizes with ~65% sparsity.
Reports mean +/- std latency (us), speedup ratio, and memory footprint (bytes).

Usage:
    python benchmarks/bench_stage1b.py
    python benchmarks/bench_stage1b.py --json-only
    python benchmarks/bench_stage1b.py --warmup 200 --iters 2000

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure tern-core is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.accel import (
    TernaryLinearAccel,
    get_acceleration_info,
    is_accelerated,
)
from terncore.arithmetic.linear import TernaryLinear
from terncore.arithmetic.quantizer import TernaryQuantizer


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MATRIX_SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]
DEFAULT_WARMUP = 100
DEFAULT_ITERS = 1000
BATCH_SIZE = 1
SEED = 42
TARGET_SPARSITY = 0.65
THRESHOLD = 0.7  # default threshold typically gives ~65% sparsity


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class MemoryFootprint:
    """Memory footprint for a single layer configuration."""

    weight_params: int
    fp32_bytes: int        # raw FP32 weight storage
    packed_2bit_bytes: int  # 2-bit packed ternary weights
    bitmap_bytes: int      # sparsity bitmap
    total_ternary_bytes: int  # packed + bitmap + alpha (4 bytes)
    compression_vs_fp32: float


@dataclass
class BenchResult:
    """Result for one (size, backend) measurement."""

    backend: str
    in_features: int
    out_features: int
    sparsity: float
    warmup_iters: int
    measured_iters: int
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    median_us: float
    memory: MemoryFootprint


@dataclass
class SizeComparison:
    """Comparison between PyTorch and C+SIMD for one matrix size."""

    in_features: int
    out_features: int
    pytorch: BenchResult
    accel: Optional[BenchResult]
    speedup: Optional[float]


@dataclass
class BenchReport:
    """Full benchmark report."""

    timestamp: str
    platform: dict
    acceleration: dict
    config: dict
    results: list[SizeComparison] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Weight initialisation — target ~65% sparsity
# ═══════════════════════════════════════════════════════════════


def _init_weights_for_sparsity(
    layer: TernaryLinear, target: float = TARGET_SPARSITY
) -> float:
    """
    Construct a zero-inflated weight distribution that produces
    approximately `target` fraction of zero weights after ternary
    quantisation at the configured threshold.

    Standard normal weights with threshold=0.7 yield only ~42% sparsity.
    Real ternary models have 60-70% zeros because trained weight
    distributions are peaked at zero. We simulate this by zeroing out
    a calibrated fraction of weights, then binary-searching for the
    fraction that hits the target sparsity after quantisation.

    Returns actual sparsity after quantisation.
    """
    lo, hi = 0.0, 0.99

    for _ in range(20):  # converges to <0.1% tolerance
        mid = (lo + hi) / 2.0
        torch.manual_seed(SEED)
        w = torch.randn(layer.out_features, layer.in_features)
        w = w * (1.0 / layer.in_features**0.5)
        # Zero out `mid` fraction — simulates zero-peaked distribution
        mask = (torch.rand_like(w) >= mid).float()
        w = w * mask
        layer.weight.data.copy_(w)
        layer.invalidate_cache()

        actual = layer.sparsity
        if abs(actual - target) < 0.005:
            return actual
        if actual < target:
            lo = mid
        else:
            hi = mid

    return layer.sparsity


# ═══════════════════════════════════════════════════════════════
# Memory measurement
# ═══════════════════════════════════════════════════════════════


def _measure_memory(layer: TernaryLinear) -> MemoryFootprint:
    """Compute memory footprint for a ternary layer."""
    M = layer.out_features
    N = layer.in_features
    weight_params = M * N

    fp32_bytes = weight_params * 4  # FP32 reference

    # Ternary packed: 2 bits per weight = 4 weights per byte
    packed_bytes = (weight_params + 3) // 4
    # Sparsity bitmap: 1 bit per weight
    bm_bytes = (weight_params + 7) // 8
    # Alpha scalar (FP32) + bias (FP32 * M) not counted in weight footprint
    alpha_bytes = 4
    total_ternary = packed_bytes + bm_bytes + alpha_bytes

    return MemoryFootprint(
        weight_params=weight_params,
        fp32_bytes=fp32_bytes,
        packed_2bit_bytes=packed_bytes,
        bitmap_bytes=bm_bytes,
        total_ternary_bytes=total_ternary,
        compression_vs_fp32=fp32_bytes / total_ternary if total_ternary > 0 else 0,
    )


# ═══════════════════════════════════════════════════════════════
# Timing harness
# ═══════════════════════════════════════════════════════════════


def _bench_layer(
    layer: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> np.ndarray:
    """
    Time `layer(x)` for `iters` iterations after `warmup` warmup calls.

    Returns array of per-iteration times in microseconds.
    """
    layer.eval()

    # Warmup — populate caches, stabilise branch predictors
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)

    # Measured iterations
    times = np.empty(iters, dtype=np.float64)
    with torch.no_grad():
        for i in range(iters):
            t0 = time.perf_counter()
            _ = layer(x)
            t1 = time.perf_counter()
            times[i] = (t1 - t0) * 1e6  # seconds → microseconds

    return times


# ═══════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════


def run_benchmarks(
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> BenchReport:
    """Run all benchmarks and return the report."""
    import platform as plat

    accel_info = get_acceleration_info()

    report = BenchReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        platform={
            "machine": plat.machine(),
            "processor": plat.processor(),
            "system": plat.system(),
            "python": plat.python_version(),
            "torch": torch.__version__,
        },
        acceleration=accel_info,
        config={
            "batch_size": BATCH_SIZE,
            "warmup_iters": warmup,
            "measured_iters": iters,
            "seed": SEED,
            "target_sparsity": TARGET_SPARSITY,
            "threshold": THRESHOLD,
        },
    )

    for N_in, N_out in MATRIX_SIZES:
        torch.manual_seed(SEED)

        # --- PyTorch baseline ---
        base = TernaryLinear(N_in, N_out, bias=True, threshold=THRESHOLD)
        sparsity = _init_weights_for_sparsity(base)
        mem = _measure_memory(base)

        x = torch.randn(BATCH_SIZE, N_in)

        times_pt = _bench_layer(base, x, warmup, iters)

        result_pt = BenchResult(
            backend="PyTorch",
            in_features=N_in,
            out_features=N_out,
            sparsity=sparsity,
            warmup_iters=warmup,
            measured_iters=iters,
            mean_us=float(np.mean(times_pt)),
            std_us=float(np.std(times_pt)),
            min_us=float(np.min(times_pt)),
            max_us=float(np.max(times_pt)),
            median_us=float(np.median(times_pt)),
            memory=mem,
        )

        # --- C+SIMD accelerated ---
        result_accel = None
        speedup = None

        if is_accelerated() and N_in % 4 == 0:
            accel = TernaryLinearAccel.from_ternary_linear(base)
            times_ac = _bench_layer(accel, x, warmup, iters)

            result_accel = BenchResult(
                backend="C+SIMD",
                in_features=N_in,
                out_features=N_out,
                sparsity=sparsity,
                warmup_iters=warmup,
                measured_iters=iters,
                mean_us=float(np.mean(times_ac)),
                std_us=float(np.std(times_ac)),
                min_us=float(np.min(times_ac)),
                max_us=float(np.max(times_ac)),
                median_us=float(np.median(times_ac)),
                memory=mem,  # same weight footprint
            )

            speedup = result_pt.mean_us / result_accel.mean_us

        report.results.append(
            SizeComparison(
                in_features=N_in,
                out_features=N_out,
                pytorch=result_pt,
                accel=result_accel,
                speedup=speedup,
            )
        )

    return report


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════

HEADER = (
    f"{'Size':>12s}  {'Sparsity':>8s}  "
    f"{'PyTorch (us)':>16s}  {'C+SIMD (us)':>16s}  "
    f"{'Speedup':>8s}  {'Packed (B)':>10s}  {'Bitmap (B)':>10s}  "
    f"{'Tern Tot (B)':>12s}  {'FP32 (B)':>10s}  {'Compress':>8s}"
)

SEP = "-" * len(HEADER)


def _fmt_us(mean: float, std: float) -> str:
    return f"{mean:8.1f} +/- {std:5.1f}"


def print_table(report: BenchReport) -> None:
    """Print results as a formatted table."""
    print()
    print("=" * len(HEADER))
    print("  Stage 1B Microbenchmark: TernaryLinearAccel (C+SIMD) vs TernaryLinear (PyTorch)")
    print("=" * len(HEADER))
    print()

    # Platform info
    p = report.platform
    print(f"  Platform : {p['system']} {p['machine']} ({p['processor']})")
    print(f"  Python   : {p['python']}   PyTorch: {p['torch']}")

    a = report.acceleration
    if a["accelerated"]:
        simd = a["simd_support"]
        caps = [k.upper() for k, v in simd.items() if v and k != "scalar"]
        print(f"  Accel    : libterncore v{a['version']}  SIMD: {', '.join(caps) or 'scalar only'}")
    else:
        print("  Accel    : NOT AVAILABLE (C library not loaded)")

    c = report.config
    print(f"  Config   : B={c['batch_size']}, warmup={c['warmup_iters']}, "
          f"iters={c['measured_iters']}, seed={c['seed']}, "
          f"threshold={c['threshold']}")
    print()
    print(HEADER)
    print(SEP)

    for comp in report.results:
        size_str = f"{comp.in_features}x{comp.out_features}"
        sparsity_str = f"{comp.pytorch.sparsity:.1%}"

        pt_str = _fmt_us(comp.pytorch.mean_us, comp.pytorch.std_us)

        if comp.accel is not None:
            ac_str = _fmt_us(comp.accel.mean_us, comp.accel.std_us)
            sp_str = f"{comp.speedup:.2f}x"
        else:
            ac_str = "      N/A       "
            sp_str = "   N/A  "

        m = comp.pytorch.memory
        print(
            f"{size_str:>12s}  {sparsity_str:>8s}  "
            f"{pt_str:>16s}  {ac_str:>16s}  "
            f"{sp_str:>8s}  {m.packed_2bit_bytes:>10,d}  {m.bitmap_bytes:>10,d}  "
            f"{m.total_ternary_bytes:>12,d}  {m.fp32_bytes:>10,d}  "
            f"{m.compression_vs_fp32:>7.1f}x"
        )

    print(SEP)
    print()


def report_to_dict(report: BenchReport) -> dict:
    """Convert report to a JSON-serialisable dict."""

    def _bench_dict(b: Optional[BenchResult]) -> Optional[dict]:
        if b is None:
            return None
        d = asdict(b)
        return d

    return {
        "timestamp": report.timestamp,
        "platform": report.platform,
        "acceleration": report.acceleration,
        "config": report.config,
        "results": [
            {
                "size": f"{c.in_features}x{c.out_features}",
                "in_features": c.in_features,
                "out_features": c.out_features,
                "speedup": round(c.speedup, 4) if c.speedup else None,
                "pytorch": _bench_dict(c.pytorch),
                "accel": _bench_dict(c.accel),
            }
            for c in report.results
        ],
    }


def print_json(report: BenchReport) -> None:
    """Print results as formatted JSON."""
    print(json.dumps(report_to_dict(report), indent=2))


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1B microbenchmark: C+SIMD vs PyTorch"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS,
        help=f"Measured iterations (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Output JSON only (no table)",
    )
    args = parser.parse_args()

    report = run_benchmarks(warmup=args.warmup, iters=args.iters)

    if args.json_only:
        print_json(report)
    else:
        print_table(report)
        print("JSON output:")
        print()
        print_json(report)


if __name__ == "__main__":
    main()
