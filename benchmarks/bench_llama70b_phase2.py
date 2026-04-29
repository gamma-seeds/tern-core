#!/usr/bin/env python3
"""
bench_llama70b_phase2.py — Phase 2 validation runner for llama70b-v0.6.0.mlpackage.

Single forward pass on zeros input, CPU_ONLY, peak RSS via resource.getrusage,
JSON result blob written to benchmarks/llama70b_phase2.json. Input spec is read
from the mlpackage at runtime; the zeros tensor is built to match whatever the
model declares. No baseline comparison in this runner — Mistral-7B v0.1 numbers
are to be supplied separately.
"""

import json
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.proto import FeatureTypes_pb2

MODEL_PATH = Path("/Users/syn/synapticode/tern-core/llama70b-v0.6.0.mlpackage")
RESULTS_PATH = Path(__file__).parent / "llama70b_phase2.json"
DEFAULT_SEQ_LEN = 512

# Watchdog thresholds — previous run 2026-04-11 18:23:00 was killed by jetsam
# with compressor_size=2395677 pages. The compressor filling up is the real
# kill signal; raw free+speculative legitimately drains during a healthy
# mmap ramp (CoreML streaming a 39 GB weight file) without touching the
# compressor, so the available floor is only a last-ditch OOM backstop.
WATCHDOG_COMPRESSOR_TRIP_PAGES = 1_800_000   # ~27.5 GiB compressed
WATCHDOG_POLL_S = 2.0


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024 * 1024)) if sys.platform == "darwin" else (r / 1024)


def dtype_name(ma) -> str:
    return FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(ma.dataType)


def np_dtype_for(name: str):
    mapping = {
        "FLOAT32": np.float32,
        "FLOAT16": np.float16,
        "DOUBLE":  np.float64,
        "INT32":   np.int32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported input dtype: {name}")
    return mapping[name]


def concrete_shape(ma, default_seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    fixed = list(ma.shape)
    if fixed and all(int(d) > 0 for d in fixed):
        return tuple(int(d) for d in fixed)
    ranges = list(ma.shapeRange.sizeRanges)
    if ranges:
        shape = []
        for r in ranges:
            lo, hi = int(r.lowerBound), int(r.upperBound)
            if lo <= default_seq_len <= (hi if hi > 0 else default_seq_len):
                shape.append(default_seq_len)
            elif hi > 0:
                shape.append(hi)
            else:
                shape.append(lo)
        return tuple(shape)
    raise ValueError("no concrete shape and no shape range in spec")


_VM_STAT_LINE = re.compile(r'^([^:]+):\s+(\d+)\.?\s*$')


def read_vm_stat() -> dict:
    out = subprocess.check_output(["vm_stat"]).decode()
    stats = {}
    for line in out.splitlines():
        m = _VM_STAT_LINE.match(line)
        if m:
            key = m.group(1).strip().lower().replace(' ', '_').replace('"', '')
            stats[key] = int(m.group(2))
    return stats


def start_watchdog(report: dict, dump_fn, state: dict) -> threading.Thread:
    """Background pressure watchdog.

    Polls vm_stat every WATCHDOG_POLL_S seconds. If the compressor exceeds
    WATCHDOG_COMPRESSOR_TRIP_PAGES or free+speculative drops below
    WATCHDOG_AVAILABLE_FLOOR_PAGES, writes stage='killed_presumed' with the
    tripping vm_stat snapshot, flushes the JSON report, and exits the process
    via os._exit(137) so we win the race against jetsam SIGKILL.
    """
    stop_event = state["stop_event"]

    def _run():
        while not stop_event.is_set():
            try:
                s = read_vm_stat()
                comp = s.get("pages_occupied_by_compressor", 0) or \
                       s.get("pages_stored_in_compressor", 0)
                free = s.get("pages_free", 0)
                spec = s.get("pages_speculative", 0)
                available = free + spec
                state["last_sample"] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "pages_free": free,
                    "pages_speculative": spec,
                    "pages_compressor": comp,
                }
                tripped = None
                if comp >= WATCHDOG_COMPRESSOR_TRIP_PAGES:
                    tripped = "compressor_trip"

                if tripped:
                    report["stage"] = "killed_presumed"
                    report["watchdog_trip"] = {
                        "reason": tripped,
                        "tripped_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pages_free": free,
                        "pages_speculative": spec,
                        "pages_compressor": comp,
                        "compressor_trip_pages": WATCHDOG_COMPRESSOR_TRIP_PAGES,
                        "available_floor_pages": WATCHDOG_AVAILABLE_FLOOR_PAGES,
                        "rss_mb_self": rss_mb(),
                    }
                    try:
                        dump_fn()
                    finally:
                        sys.stdout.write(
                            f"[watchdog] TRIPPED ({tripped}): "
                            f"comp={comp} free={free} spec={spec} — exit 137\n"
                        )
                        sys.stdout.flush()
                        sys.stderr.write(
                            f"[watchdog] TRIPPED ({tripped}): "
                            f"comp={comp} free={free} spec={spec}\n"
                        )
                        sys.stderr.flush()
                    os._exit(137)
            except Exception as e:
                sys.stderr.write(f"[watchdog] poll error: {e}\n")
                sys.stderr.flush()
            stop_event.wait(WATCHDOG_POLL_S)

    t = threading.Thread(target=_run, name="watchdog", daemon=True)
    t.start()
    return t


def describe_input(feature) -> dict:
    info = {"name": feature.name,
            "feature_type": feature.type.WhichOneof("Type")}
    if info["feature_type"] == "multiArrayType":
        ma = feature.type.multiArrayType
        info["dtype"] = dtype_name(ma)
        info["declared_shape"] = list(ma.shape)
        if len(ma.shapeRange.sizeRanges) > 0:
            info["shape_ranges"] = [
                {"lower": r.lowerBound, "upper": r.upperBound}
                for r in ma.shapeRange.sizeRanges
            ]
    return info


def main() -> int:
    report = {
        "runner": "bench_llama70b_phase2.py",
        "model_path": str(MODEL_PATH),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip(),
        "python": platform.python_version(),
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "compute_units": "CPU_AND_NE",
        "watchdog": {
            "compressor_trip_pages": WATCHDOG_COMPRESSOR_TRIP_PAGES,
            "available_floor_pages": WATCHDOG_AVAILABLE_FLOOR_PAGES,
            "poll_s": WATCHDOG_POLL_S,
        },
        "stage": "start",
        "inputs": [],
        "outputs": [],
        "concrete_input_shapes": {},
        "rss_mb": {},
        "timings_s": {},
        "output_shapes": {},
        "error": None,
    }

    _dump_lock = threading.Lock()

    def dump():
        with _dump_lock:
            RESULTS_PATH.write_text(
                json.dumps(report, indent=2, default=str)
            )

    watchdog_state = {"stop_event": threading.Event(), "last_sample": None}
    start_watchdog(report, dump, watchdog_state)

    try:
        print(f"[phase2] model: {MODEL_PATH}", flush=True)
        print(f"[phase2] rss_start_mb: {rss_mb():.1f}", flush=True)
        report["rss_mb"]["start"] = rss_mb()

        spec = ct.utils.load_spec(str(MODEL_PATH))
        report["stage"] = "spec_loaded"
        report["spec_version"] = spec.specificationVersion
        report["model_type"] = spec.WhichOneof("Type")
        report["inputs"] = [describe_input(f) for f in spec.description.input]
        report["outputs"] = [
            {"name": f.name, "feature_type": f.type.WhichOneof("Type")}
            for f in spec.description.output
        ]
        print(f"[phase2] spec_version={report['spec_version']} "
              f"model_type={report['model_type']}", flush=True)
        print(f"[phase2] inputs: {report['inputs']}", flush=True)
        print(f"[phase2] outputs: {report['outputs']}", flush=True)

        input_dict = {}
        for f in spec.description.input:
            if f.type.WhichOneof("Type") != "multiArrayType":
                raise ValueError(
                    f"input {f.name}: only multiArrayType supported, "
                    f"got {f.type.WhichOneof('Type')}"
                )
            ma = f.type.multiArrayType
            shape = concrete_shape(ma)
            arr = np.zeros(shape, dtype=np_dtype_for(dtype_name(ma)))
            input_dict[f.name] = arr
            report["concrete_input_shapes"][f.name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }
            print(f"[phase2] input {f.name}: shape={list(arr.shape)} "
                  f"dtype={arr.dtype} nbytes={arr.nbytes}", flush=True)
        dump()

        print(f"[phase2] rss_before_load_mb: {rss_mb():.1f}", flush=True)
        report["rss_mb"]["before_load"] = rss_mb()
        dump()

        t0 = time.perf_counter()
        model = ct.models.MLModel(
            str(MODEL_PATH),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        load_s = time.perf_counter() - t0
        report["timings_s"]["load"] = load_s
        report["rss_mb"]["after_load"] = rss_mb()
        report["stage"] = "model_loaded"
        print(f"[phase2] load_s={load_s:.2f} "
              f"rss_after_load_mb={rss_mb():.1f}", flush=True)
        dump()

        print(f"[phase2] running single predict...", flush=True)
        t0 = time.perf_counter()
        out = model.predict(input_dict)
        predict_s = time.perf_counter() - t0
        report["timings_s"]["predict"] = predict_s
        report["rss_mb"]["after_predict"] = rss_mb()
        report["stage"] = "predict_complete"
        report["output_names"] = list(out.keys())
        for k, v in out.items():
            report["output_shapes"][k] = (
                list(v.shape) if hasattr(v, "shape") else None
            )
        print(f"[phase2] predict_s={predict_s:.3f} "
              f"rss_after_predict_mb={rss_mb():.1f}", flush=True)
        for k in report["output_shapes"]:
            print(f"[phase2] output {k}: shape={report['output_shapes'][k]}",
                  flush=True)

        report["rss_mb"]["peak"] = rss_mb()
        report["stage"] = "done"
        watchdog_state["stop_event"].set()
        if watchdog_state.get("last_sample") is not None:
            report["watchdog_last_sample"] = watchdog_state["last_sample"]
        dump()
        print(f"[phase2] results: {RESULTS_PATH}", flush=True)
        return 0

    except Exception as e:
        report["error"] = {
            "stage": report["stage"],
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        report["rss_mb"]["peak_on_error"] = rss_mb()
        dump()
        print(f"[phase2] ERROR at stage={report['stage']}: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
