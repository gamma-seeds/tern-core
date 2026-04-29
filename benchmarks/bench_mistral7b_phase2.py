#!/usr/bin/env python3
"""
bench_mistral7b_phase2.py — Apple brief benchmark for Mistral-7B ternary on M4 Pro.

Produces: tokens/sec, peak RSS, energy profile across compute units.
Tests both the raw FP16-encoded ternary mlpackage and a 2-bit palettised
version (where the real speedup lives).

Methodology follows bench_coreml_ane.py: warmup + multi-run measurement,
powermetrics energy sampling, JSON + stdout output.
"""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.proto import FeatureTypes_pb2

# Relocated from tern-core/output/coreml_models/ to canonical models/coreml/mistral-7b/
# per disk audit 2026-04-18. See models/coreml/mistral-7b/meta.json for provenance.
MODEL_PATH = Path(
    "/Users/syn/synapticode/models/coreml/mistral-7b"
    "/mistral_7b_ternary.mlpackage"
)
PALETTISED_PATH = Path(
    "/Users/syn/synapticode/models/coreml/mistral-7b"
    "/mistral_7b_ternary_2bit.mlpackage"
)
RESULTS_PATH = Path(__file__).parent / "mistral7b_phase2.json"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 50
SEQ_LEN = 64

COMPUTE_UNITS = [
    ("ALL", ct.ComputeUnit.ALL),
    ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
]

WATCHDOG_COMPRESSOR_TRIP_PAGES = 1_800_000
WATCHDOG_POLL_S = 3.0

_VM_STAT_LINE = re.compile(r'^([^:]+):\s+(\d+)\.?\s*$')
_POWER_RE = re.compile(
    r'(?:Package|Combined)\s+Power.*?:\s*([\d.]+)\s*(?:m?W)',
    re.IGNORECASE,
)


# ── helpers ──────────────────────────────────────────────────────────

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


# ── watchdog ─────────────────────────────────────────────────────────

def start_watchdog(report: dict, dump_fn) -> dict:
    state = {"stop_event": threading.Event(), "last_sample": None}
    _lock = threading.Lock()

    def _run():
        while not state["stop_event"].is_set():
            try:
                s = read_vm_stat()
                comp = s.get("pages_occupied_by_compressor", 0)
                free = s.get("pages_free", 0)
                spec = s.get("pages_speculative", 0)
                available = free + spec
                state["last_sample"] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "comp": comp, "free": free, "spec": spec,
                }
                tripped = None
                if comp >= WATCHDOG_COMPRESSOR_TRIP_PAGES:
                    tripped = "compressor_trip"
                if tripped:
                    report["stage"] = "killed_presumed"
                    report["watchdog_trip"] = {
                        "reason": tripped,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "comp": comp, "free": free, "spec": spec,
                        "rss_mb": rss_mb(),
                    }
                    try:
                        dump_fn()
                    finally:
                        msg = (f"[watchdog] TRIPPED ({tripped}): "
                               f"comp={comp} free={free} spec={spec}\n")
                        sys.stdout.write(msg); sys.stdout.flush()
                        sys.stderr.write(msg); sys.stderr.flush()
                    os._exit(137)
            except Exception:
                pass
            state["stop_event"].wait(WATCHDOG_POLL_S)

    t = threading.Thread(target=_run, name="watchdog", daemon=True)
    t.start()
    return state


# ── benchmark core ───────────────────────────────────────────────────

def benchmark_cu(model_path: Path, cu_name: str, cu, input_dict: dict,
                 seq_len: int) -> dict:
    print(f"  [{cu_name}] loading...", flush=True)
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(model_path), compute_units=cu)
    load_s = time.perf_counter() - t0
    print(f"  [{cu_name}] loaded in {load_s:.2f}s, rss={rss_mb():.0f} MB",
          flush=True)

    print(f"  [{cu_name}] warmup {WARMUP_RUNS} runs...", flush=True)
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
          f"rss={peak:.0f} MB", flush=True)
    del model; gc.collect()
    return result


# ── energy measurement ───────────────────────────────────────────────

def sudo_available() -> bool:
    try:
        subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
             "-n", "1", "-i", "100"],
            capture_output=True, timeout=5,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def measure_energy(model_path: Path, cu, input_dict: dict,
                   label: str, duration_s: float = 15.0) -> dict:
    print(f"  [energy:{label}] sustained inference for {duration_s:.0f}s...",
          flush=True)
    model = ct.models.MLModel(str(model_path), compute_units=cu)
    for _ in range(WARMUP_RUNS):
        model.predict(input_dict)

    n_samples = max(int(duration_s), 10)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pm = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
             "-i", "1000", "-n", str(n_samples)],
            stdout=open(tmp_path, 'w'), stderr=subprocess.DEVNULL,
        )
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
        pm.terminate(); pm.wait(timeout=3)

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

    del model; gc.collect()
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
          f"{result['energy_per_inference_mj']:.2f} mJ/inference, "
          f"{n_inferences} inferences", flush=True)
    return result


# ── palettisation ────────────────────────────────────────────────────

def palettise_model(src_path: Path, dst_path: Path) -> float:
    from coremltools.optimize.coreml import (
        OpPalettizerConfig, OptimizationConfig, palettize_weights,
    )
    print(f"  Applying 2-bit palettisation...", flush=True)
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(src_path))
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=2, mode="kmeans")
    )
    pal = palettize_weights(model, config)
    pal.save(str(dst_path))
    elapsed = time.perf_counter() - t0
    size = model_size_mb(dst_path)
    print(f"  Palettised in {elapsed:.1f}s → {dst_path.name} "
          f"({size:.1f} MB)", flush=True)
    del model, pal; gc.collect()
    return elapsed


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    report = {
        "runner": "bench_mistral7b_phase2.py",
        "model_path": str(MODEL_PATH),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip(),
        "python": platform.python_version(),
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "seq_len": SEQ_LEN,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "stage": "start",
        "spec": {},
        "raw_benchmarks": {},
        "palettised_benchmarks": {},
        "model_sizes_mb": {},
        "energy": {},
        "error": None,
    }

    _dump_lock = threading.Lock()

    def dump():
        with _dump_lock:
            RESULTS_PATH.write_text(
                json.dumps(report, indent=2, default=str)
            )

    wd = start_watchdog(report, dump)

    try:
        # ── spec ──
        print("=" * 72, flush=True)
        print("  Mistral-7B Ternary — Apple Brief Benchmark", flush=True)
        print("=" * 72, flush=True)
        print(f"  model: {MODEL_PATH}", flush=True)

        spec = ct.utils.load_spec(str(MODEL_PATH))
        inp = spec.description.input[0]
        ma = inp.type.multiArrayType
        dtype_name = FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(
            ma.dataType
        )
        shape = list(ma.shape)
        report["spec"] = {
            "input_name": inp.name,
            "input_dtype": dtype_name,
            "input_shape": shape,
            "output_name": spec.description.output[0].name,
            "spec_version": spec.specificationVersion,
        }
        report["stage"] = "spec_loaded"
        dump()

        # ── build input ──
        np_dtype = {"INT32": np.int32, "FLOAT16": np.float16,
                    "FLOAT32": np.float32}[dtype_name]
        input_dict = {inp.name: np.zeros(shape, dtype=np_dtype)}
        print(f"  input: {inp.name} {dtype_name} {shape}", flush=True)

        # ── validation pass ──
        print(f"\n{'─'*72}", flush=True)
        print("  Phase A: validation pass (ComputeUnit.ALL)", flush=True)
        print(f"{'─'*72}", flush=True)
        t0 = time.perf_counter()
        model = ct.models.MLModel(
            str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL
        )
        load_s = time.perf_counter() - t0
        print(f"  loaded in {load_s:.2f}s, rss={rss_mb():.0f} MB",
              flush=True)

        t0 = time.perf_counter()
        out = model.predict(input_dict)
        pred_s = time.perf_counter() - t0
        out_shapes = {k: list(v.shape) if hasattr(v, 'shape') else None
                      for k, v in out.items()}
        report["validation"] = {
            "load_s": load_s, "predict_s": pred_s,
            "output_shapes": out_shapes,
            "rss_mb": rss_mb(),
        }
        report["stage"] = "validated"
        print(f"  predict: {pred_s*1000:.2f} ms, "
              f"output shapes: {out_shapes}", flush=True)
        del model; gc.collect()
        dump()

        # ── raw model benchmarks ──
        print(f"\n{'─'*72}", flush=True)
        print("  Phase B: raw FP16-encoded ternary benchmarks", flush=True)
        print(f"{'─'*72}", flush=True)
        report["model_sizes_mb"]["raw"] = model_size_mb(MODEL_PATH)
        print(f"  model size: {report['model_sizes_mb']['raw']:.1f} MB",
              flush=True)

        for cu_name, cu in COMPUTE_UNITS:
            r = benchmark_cu(MODEL_PATH, cu_name, cu, input_dict, SEQ_LEN)
            report["raw_benchmarks"][cu_name] = r
            dump()

        # ── palettise (non-fatal — Inf weights in FP16 tensors can
        #    cause sklearn KMeans to reject; proceed to Phase D) ──
        print(f"\n{'─'*72}", flush=True)
        print("  Phase C: 2-bit palettisation + benchmark", flush=True)
        print(f"{'─'*72}", flush=True)

        phase_c_ok = False
        try:
            if PALETTISED_PATH.exists():
                print(f"  Using cached: {PALETTISED_PATH}", flush=True)
            else:
                pal_time = palettise_model(MODEL_PATH, PALETTISED_PATH)
                report["palettise_seconds"] = pal_time
                dump()

            report["model_sizes_mb"]["palettised_2bit"] = model_size_mb(
                PALETTISED_PATH
            )
            compression = (report["model_sizes_mb"]["raw"]
                            / report["model_sizes_mb"]["palettised_2bit"])
            print(f"  palettised size: "
                  f"{report['model_sizes_mb']['palettised_2bit']:.1f} MB "
                  f"({compression:.1f}× compression)", flush=True)

            for cu_name, cu in COMPUTE_UNITS:
                r = benchmark_cu(
                    PALETTISED_PATH, f"2bit_{cu_name}", cu, input_dict,
                    SEQ_LEN,
                )
                report["palettised_benchmarks"][cu_name] = r
                dump()

            phase_c_ok = True
        except Exception as exc:
            report["phase_c_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            dump()
            print(f"  [Phase C] FAILED (non-fatal): {type(exc).__name__}: "
                  f"{exc}", flush=True)
            print("  Continuing to Phase D...", flush=True)

        # ── energy ──
        print(f"\n{'─'*72}", flush=True)
        print("  Phase D: energy profile", flush=True)
        print(f"{'─'*72}", flush=True)

        if sudo_available():
            best_raw = min(report["raw_benchmarks"].values(),
                           key=lambda r: r["latency_mean_ms"])
            best_raw_cu = best_raw["compute_units"]
            cu_map = {n: c for n, c in COMPUTE_UNITS}

            report["energy"]["raw_best"] = measure_energy(
                MODEL_PATH,
                cu_map.get(best_raw_cu, ct.ComputeUnit.ALL),
                input_dict,
                f"raw_{best_raw_cu}",
            )
            dump()

            if phase_c_ok and report["palettised_benchmarks"]:
                best_pal = min(
                    report["palettised_benchmarks"].values(),
                    key=lambda r: r["latency_mean_ms"],
                )
                best_pal_cu = best_pal["compute_units"].replace("2bit_", "")
                report["energy"]["pal_best"] = measure_energy(
                    PALETTISED_PATH,
                    cu_map.get(best_pal_cu, ct.ComputeUnit.ALL),
                    input_dict,
                    f"2bit_{best_pal_cu}",
                )
                dump()
            elif not phase_c_ok:
                print("  Skipping palettised energy (Phase C failed)",
                      flush=True)
                report["energy"]["pal_best_note"] = "skipped: Phase C failed"
                dump()
        else:
            print("  sudo powermetrics unavailable — skipping energy",
                  flush=True)
            report["energy"]["note"] = "sudo unavailable"

        # ── summary ──
        print(f"\n{'='*72}", flush=True)
        print("  RESULTS — Mistral-7B Ternary (Apple Brief)", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"  Hardware: {report['hardware']}", flush=True)
        print(f"  Input: {inp.name} {dtype_name} {shape}", flush=True)
        print(f"  Runs: {WARMUP_RUNS} warmup, {BENCHMARK_RUNS} measured",
              flush=True)
        print(flush=True)

        print(f"  Model sizes:", flush=True)
        for k, v in report["model_sizes_mb"].items():
            print(f"    {k:<20} {v:>10.1f} MB", flush=True)
        print(flush=True)

        hdr = (f"  {'Config':<25} {'Mean ms':>9} {'Min ms':>9} "
               f"{'tok/s':>8} {'RSS MB':>8}")
        print(hdr, flush=True)
        print(f"  {'─'*25} {'─'*9} {'─'*9} {'─'*8} {'─'*8}", flush=True)
        for label, bm in [("Raw FP16-tern", report["raw_benchmarks"]),
                           ("Palettised 2-bit", report["palettised_benchmarks"])]:
            for cu, r in bm.items():
                tag = f"{label} ({cu})" if "2bit" not in r["compute_units"] \
                    else f"{label} ({cu})"
                print(f"  {tag:<25} {r['latency_mean_ms']:>8.2f} "
                      f"{r['latency_min_ms']:>8.2f} "
                      f"{r['tokens_per_second']:>7.1f} "
                      f"{r['peak_rss_mb']:>7.0f}", flush=True)
        print(flush=True)

        if report["energy"].get("raw_best"):
            e = report["energy"]["raw_best"]
            print(f"  Energy (raw best):   {e.get('power_mean_w',0):.2f} W, "
                  f"{e.get('energy_per_inference_mj',0):.2f} mJ/infer",
                  flush=True)
        if report["energy"].get("pal_best"):
            e = report["energy"]["pal_best"]
            print(f"  Energy (2-bit best): {e.get('power_mean_w',0):.2f} W, "
                  f"{e.get('energy_per_inference_mj',0):.2f} mJ/infer",
                  flush=True)

        report["stage"] = "done"
        report["peak_rss_mb"] = rss_mb()
        wd["stop_event"].set()
        if wd.get("last_sample"):
            report["watchdog_last_sample"] = wd["last_sample"]
        dump()
        print(f"\n  Results: {RESULTS_PATH}", flush=True)
        print("=" * 72, flush=True)
        return 0

    except Exception as e:
        report["error"] = {
            "stage": report["stage"],
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        report["peak_rss_mb"] = rss_mb()
        dump()
        print(f"[phase2] ERROR at stage={report['stage']}: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
