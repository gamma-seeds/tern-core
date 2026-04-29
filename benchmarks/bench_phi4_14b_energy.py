#!/usr/bin/env python3
"""
Focused Phase D energy measurement for Phi-4 14B.
Loads only the best compute unit config (ALL) and runs powermetrics for 15s.
"""

import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import coremltools as ct

MODEL_PATH = Path(
    "/Users/syn/synapticode/models/coreml/phi4-14b"
    "/phi4_14b_ternary_v0.1.0.mlpackage"
)
PALETTISED_PATH = Path(
    "/Users/syn/synapticode/models/coreml/phi4-14b"
    "/phi4_14b_ternary_v0.1.0_2bit.mlpackage"
)
RESULTS_PATH = Path(__file__).parent / "phi4_14b_phase2.json"

SEQ_LEN = 64
ENERGY_DURATION_S = 15

_POWER_RE = re.compile(
    r'(?:Package|Combined)\s+Power.*?:\s*([\d.]+)\s*(?:m?W)',
    re.IGNORECASE,
)


def measure_energy(model_path, compute_unit, input_dict, label):
    print(f"  [{label}] loading model...", flush=True)
    model = ct.models.MLModel(str(model_path), compute_units=compute_unit)
    print(f"  [{label}] sustained inference for {ENERGY_DURATION_S}s...", flush=True)

    watts = []
    stop = threading.Event()

    def _sample():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp = f.name
        try:
            proc = subprocess.Popen(
                ["sudo", "powermetrics", "--samplers", "cpu_power",
                 "-i", "1000", "-n", str(ENERGY_DURATION_S + 2)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for line in proc.stdout:
                m = _POWER_RE.search(line)
                if m:
                    val = float(m.group(1))
                    if val < 0.1:
                        val *= 1000
                    watts.append(val)
            proc.wait()
        except Exception as e:
            print(f"    powermetrics error: {e}", flush=True)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    time.sleep(1)

    inferences = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ENERGY_DURATION_S:
        model.predict(input_dict)
        inferences += 1

    sampler.join(timeout=5)
    del model

    if watts:
        import statistics
        mean_w = statistics.mean(watts)
        energy_mj = (mean_w * ENERGY_DURATION_S * 1000) / max(inferences, 1)
        result = {
            "label": label,
            "power_mean_w": mean_w,
            "power_median_w": statistics.median(watts),
            "power_stdev_w": statistics.stdev(watts) if len(watts) > 1 else 0,
            "power_samples": len(watts),
            "inferences": inferences,
            "energy_per_inference_mj": energy_mj,
        }
        print(f"  [{label}] {mean_w:.2f} W, {energy_mj:.2f} mJ/inference, "
              f"{inferences} inferences", flush=True)
        return result
    else:
        print(f"  [{label}] no power samples captured", flush=True)
        return {"label": label, "note": "no power samples parsed"}


def main():
    print("=" * 72, flush=True)
    print("  Phi-4 14B — Phase D Energy Measurement", flush=True)
    print("=" * 72, flush=True)

    input_dict = {"input_ids": np.random.randint(0, 100352, (1, 512)).astype(np.int32)}

    # Measure both raw and palettised on ALL (best config)
    results = {}

    results["raw_best"] = measure_energy(
        MODEL_PATH, ct.ComputeUnit.ALL, input_dict, "raw_ALL")

    if PALETTISED_PATH.exists():
        results["pal_best"] = measure_energy(
            PALETTISED_PATH, ct.ComputeUnit.ALL, input_dict, "2bit_ALL")
    else:
        print("  No palettised model found, skipping", flush=True)

    # Update existing JSON
    if RESULTS_PATH.exists():
        report = json.load(open(RESULTS_PATH))
    else:
        report = {}

    report["energy"] = results
    report["stage"] = "done"
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Updated: {RESULTS_PATH}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
