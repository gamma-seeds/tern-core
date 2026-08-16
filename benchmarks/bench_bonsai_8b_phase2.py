#!/usr/bin/env python3
"""
bench_bonsai_8b_phase2.py — Phase B + D benchmark for Bonsai-8B
native-ternary (ternary_g128) on Apple silicon. Sibling of
bench_mistral7b_phase2.py.

Phase B (inference): tokens/sec, latency stats, peak RSS across three
compute units (ALL, CPU_AND_NE, CPU_AND_GPU), 50 measured runs at
seq_len=64. Phase D (energy): 15 s sustained inference under
``sudo powermetrics`` per compute unit. No palettisation phase — the
mlpackage is already native-ternary int4
(constexpr_blockwise_shift_scale), not FP16-encoded.

Watchdog raised to 3.0M compressor pages for the 8B class.

Note (provenance, 2026-06-01): MLComputePlan op-placement shows this
export uses **no ANE ops** — CPU_AND_NE runs on CPU. The CPU_AND_NE
energy/stability win is CPU-vs-GPU, not ANE. See the Phase-4
OPTION4_ANE_PLACEMENT diagnosis; do not read low CPU_AND_NE power as ANE.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

import argparse
import gc
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import coremltools as ct

DEFAULT_MODEL = (
    "/Users/syn/synapticode/model-library/coreml/bonsai-8b"
    "/bonsai-8b-s64.mlpackage"
)
DEFAULT_RESULTS = (
    "/Users/syn/synapticode/model-library/benchmarks/bonsai-8b/phase2.json"
)

WARMUP_RUNS = 10
BENCHMARK_RUNS = 50
SEQ_LEN = 64
VOCAB = 151669

COMPUTE_UNITS = [
    ("ALL", ct.ComputeUnit.ALL),
    ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
]

WATCHDOG_COMPRESSOR_TRIP_PAGES = 3_000_000  # 8B class
WATCHDOG_POLL_S = 3.0

_VM_STAT_LINE = re.compile(r'^([^:]+):\s+(\d+)\.?\s*$')
_POWER_RE = re.compile(
    r'(?:Package|Combined)\s+Power.*?:\s*([\d.]+)\s*(?:m?W)', re.IGNORECASE)


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024 * 1024)) if sys.platform == "darwin" else (r / 1024)


def read_vm_stat() -> dict:
    out = subprocess.check_output(["vm_stat"]).decode()
    stats = {}
    for line in out.splitlines():
        m = _VM_STAT_LINE.match(line)
        if m:
            key = m.group(1).strip().lower().replace(' ', '_').replace('"', '')
            stats[key] = int(m.group(2))
    return stats


def model_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


def start_watchdog(report: dict, dump_fn) -> dict:
    state = {"stop_event": threading.Event(), "last_sample": None}

    def _run():
        while not state["stop_event"].is_set():
            try:
                s = read_vm_stat()
                comp = s.get("pages_occupied_by_compressor", 0)
                free = s.get("pages_free", 0)
                spec = s.get("pages_speculative", 0)
                state["last_sample"] = {"comp": comp, "free": free, "spec": spec}
                if comp >= WATCHDOG_COMPRESSOR_TRIP_PAGES:
                    report["stage"] = "killed_presumed"
                    report["watchdog_trip"] = {
                        "reason": "compressor_trip", "comp": comp,
                        "free": free, "spec": spec, "rss_mb": rss_mb(),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    try:
                        dump_fn()
                    finally:
                        sys.stdout.write(
                            f"[watchdog] TRIPPED: comp={comp} free={free}\n")
                        sys.stdout.flush()
                    os._exit(137)
            except Exception:
                pass
            state["stop_event"].wait(WATCHDOG_POLL_S)

    t = threading.Thread(target=_run, name="watchdog", daemon=True)
    t.start()
    return state


def benchmark_cu(model_path, cu_name, cu, input_dict, seq_len) -> dict:
    print(f"  [{cu_name}] loading...", flush=True)
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(model_path), compute_units=cu)
    load_s = time.perf_counter() - t0
    print(f"  [{cu_name}] loaded in {load_s:.2f}s, rss={rss_mb():.0f} MB",
          flush=True)
    for _ in range(WARMUP_RUNS):
        model.predict(input_dict)
    print(f"  [{cu_name}] measuring {BENCHMARK_RUNS} runs...", flush=True)
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        model.predict(input_dict)
        latencies.append(time.perf_counter() - t0)
    peak = rss_mb()
    mean_s = statistics.mean(latencies)
    result = {
        "compute_units": cu_name,
        "load_seconds": load_s,
        "latency_mean_ms": mean_s * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_min_ms": min(latencies) * 1000,
        "latency_max_ms": max(latencies) * 1000,
        "latency_stdev_ms": (statistics.stdev(latencies) * 1000
                             if len(latencies) > 1 else 0),
        "tokens_per_second": seq_len / mean_s,
        "peak_rss_mb": peak,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
    }
    print(f"  [{cu_name}] {result['latency_mean_ms']:.2f} ms mean, "
          f"{result['tokens_per_second']:.1f} tok/s, "
          f"stdev={result['latency_stdev_ms']:.2f} ms, rss={peak:.0f} MB",
          flush=True)
    del model
    gc.collect()
    return result


def sudo_available() -> bool:
    try:
        subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
             "-n", "1", "-i", "100"], capture_output=True, timeout=5)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def measure_energy(model_path, cu, input_dict, label, duration_s=15.0) -> dict:
    print(f"  [energy:{label}] sustained {duration_s:.0f}s...", flush=True)
    model = ct.models.MLModel(str(model_path), compute_units=cu)
    for _ in range(WARMUP_RUNS):
        model.predict(input_dict)
    n_samples = max(int(duration_s), 10)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pm = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
             "-i", "1000", "-n", str(n_samples)],
            stdout=open(tmp_path, 'w'), stderr=subprocess.DEVNULL)
    except (FileNotFoundError, PermissionError):
        return {}
    t_end = time.time() + duration_s
    n_inferences = 0
    while time.time() < t_end:
        model.predict(input_dict)
        n_inferences += 1
    try:
        pm.wait(timeout=duration_s + 5)
    except subprocess.TimeoutExpired:
        pm.terminate()
        pm.wait(timeout=3)
    with open(tmp_path) as f:
        text = f.read()
    os.unlink(tmp_path)
    watts = []
    for line in text.splitlines():
        m = _POWER_RE.search(line)
        if m:
            val = float(m.group(1))
            if 'mW' in line:
                val /= 1000.0
            watts.append(val)
    del model
    gc.collect()
    if not watts:
        return {"label": label, "note": "no power samples parsed"}
    if len(watts) > 2:
        watts = watts[1:]
    total_j = statistics.mean(watts) * duration_s
    result = {
        "label": label,
        "power_mean_w": statistics.mean(watts),
        "power_median_w": statistics.median(watts),
        "power_stdev_w": statistics.stdev(watts) if len(watts) > 1 else 0,
        "power_samples": len(watts),
        "inferences": n_inferences,
        "energy_per_inference_mj": (total_j / n_inferences) * 1000,
    }
    print(f"  [energy:{label}] {result['power_mean_w']:.2f} W, "
          f"{result['energy_per_inference_mj']:.2f} mJ/inf, "
          f"{n_inferences} inferences", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Bonsai-8B Phase B+D benchmark.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="mlpackage path")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="phase2.json path")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    model_path = Path(args.model)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "runner": "bench_bonsai_8b_phase2.py",
        "model": str(model_path),
        "model_size_mb": round(model_size_mb(model_path), 1),
        "seq_len": args.seq_len,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "watchdog_trip_pages": WATCHDOG_COMPRESSOR_TRIP_PAGES,
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "stage": "started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase_b_inference": [],
        "phase_d_energy": [],
    }

    def dump():
        results_path.write_text(json.dumps(report, indent=2) + "\n")

    start_watchdog(report, dump)

    rng = np.random.default_rng(0)
    input_dict = {
        "input_ids": rng.integers(0, VOCAB,
                                  size=(1, args.seq_len)).astype(np.int32)
    }

    print("=" * 68, flush=True)
    print("  Bonsai-8B Phase 2 — Phase B (inference) + Phase D (energy)",
          flush=True)
    print(f"  model={model_path.name} ({report['model_size_mb']:.0f} MB)",
          flush=True)
    print("=" * 68, flush=True)

    print("\n  Phase B: inference (50 runs × 3 compute units)", flush=True)
    report["stage"] = "phase_b"
    for cu_name, cu in COMPUTE_UNITS:
        report["phase_b_inference"].append(
            benchmark_cu(model_path, cu_name, cu, input_dict, args.seq_len))
        dump()

    print("\n  Phase D: energy (15 s sustained × 3 compute units)", flush=True)
    report["stage"] = "phase_d"
    if sudo_available():
        for cu_name, cu in COMPUTE_UNITS:
            e = measure_energy(model_path, cu, input_dict, cu_name,
                               duration_s=15.0)
            if e:
                report["phase_d_energy"].append(e)
            dump()
    else:
        report["phase_d_note"] = "sudo powermetrics unavailable — skipped"
        print("  sudo powermetrics unavailable — skipping energy", flush=True)

    report["stage"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    dump()

    print("\n" + "─" * 68, flush=True)
    print("  Phase B summary:", flush=True)
    for r in report["phase_b_inference"]:
        print(f"    {r['compute_units']:12s} {r['latency_mean_ms']:.1f} ms  "
              f"{r['tokens_per_second']:.1f} tok/s  "
              f"stdev {r['latency_stdev_ms']:.1f} ms  "
              f"rss {r['peak_rss_mb']:.0f} MB", flush=True)
    if report["phase_d_energy"]:
        print("  Phase D summary:", flush=True)
        for e in report["phase_d_energy"]:
            print(f"    {e['label']:12s} {e['power_mean_w']:.2f} W  "
                  f"{e['energy_per_inference_mj']:.1f} mJ/inf", flush=True)
    print(f"  results: {results_path}", flush=True)
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
