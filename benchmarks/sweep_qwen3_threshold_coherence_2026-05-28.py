"""
Qwen3-30B-A3B ternary threshold-coherence sweep (2026-05-28).

The canonical 0.7 artefact loads faithfully (placement verified vs the FP16
checkpoint) but generates incoherently — a quality-envelope property of
ternary at 0.7, same class as Phi-4 @0.7. This sweep maps where the
coherence cliff sits: recompress at 0.5 / 0.55 / 0.6 and, after each,
run a short greedy generation probe through the Milestone-1 runnable bank
model. Three points are cheap (~50 min each) and tell us whether the cliff
is reachable above 0.5 or whether the contingency (per-layer calibration /
different proving-ground artefact) is needed.

Sequential by design — each compression loads the ~57 GB FP16 base, so they
must not overlap. Outputs land under models/compressed/qwen3-30b-a3b/sweep/.

Usage:
    HF_HUB_OFFLINE=1 python benchmarks/sweep_qwen3_threshold_coherence_2026-05-28.py

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from terncore.convert import full_convert
from terncore.moe import build_runnable_qwen3_moe, load_moe_packed
from terncore.tern_model import TernModelReader

SOURCE = "Qwen/Qwen3-30B-A3B"
THRESHOLDS = [0.5, 0.55, 0.6]
SWEEP_ROOT = Path(
    "/Volumes/Syn Archive/models/compressed/qwen3-30b-a3b/sweep"
)
RESULTS = SWEEP_ROOT / "threshold_coherence_results.json"
PROMPT = "The capital of France is"
PROBE_TOKENS = 16


def probe_generation(manifest: str) -> dict:
    """Load the artefact into the runnable bank model and greedy-generate."""
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(SOURCE)
    packed = load_moe_packed(TernModelReader(manifest), spot_check_n=0)
    model = build_runnable_qwen3_moe(packed, cfg, device="cpu")
    tok = AutoTokenizer.from_pretrained(SOURCE)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=PROBE_TOKENS, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.perf_counter() - t0
    new = out[0][ids.shape[1]:].tolist()
    text = tok.decode(out[0], skip_special_tokens=True)
    result = {
        "generated_text": text,
        "unique_new_tokens": len(set(new)),
        "new_tokens": len(new),
        "tok_per_s": round(len(new) / dt, 3),
    }
    del model, packed, tok
    gc.collect()
    return result


def main() -> int:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for thr in THRESHOLDS:
        tag = f"t{thr:.2f}".replace(".", "")
        out_dir = SWEEP_ROOT / f"qwen3_30b_a3b_ternary_{tag}.tern-model"
        out_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        print(f"\n===== threshold {thr} → {out_dir} ({started}) =====", flush=True)
        t0 = time.perf_counter()
        entry = {"threshold": thr, "output_dir": str(out_dir), "started_at": started}
        try:
            report = full_convert(
                model_id=SOURCE,
                adapter_name="qwen3_moe",
                output_dir=str(out_dir),
                threshold=thr,
                verbose=True,
            )
            entry["compression_min"] = round((time.perf_counter() - t0) / 60, 1)
            entry["compression_vs_fp16"] = report.get("compression_vs_fp16")
            entry["ternary_layers"] = report.get("ternary_layers")
            entry["int4_layers"] = report.get("int4_layers")
            gc.collect()
            print(f"[probe] generating at threshold {thr}…", flush=True)
            entry["probe"] = probe_generation(str(out_dir / "model.tern-model"))
            entry["status"] = "ok"
            print(f"[result] thr={thr}  unique={entry['probe']['unique_new_tokens']}"
                  f"/{entry['probe']['new_tokens']}  text={entry['probe']['generated_text']!r}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — record + continue the sweep
            import traceback
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()[-1500:]
            print(f"[ERROR] threshold {thr}: {entry['error']}", flush=True)
            gc.collect()
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        all_results.append(entry)
        RESULTS.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"[written] {RESULTS}", flush=True)

    print("\n===== SWEEP COMPLETE =====", flush=True)
    for e in all_results:
        p = e.get("probe", {})
        print(f"  thr={e['threshold']}  status={e['status']}  "
              f"comp={e.get('compression_vs_fp16')}  "
              f"unique={p.get('unique_new_tokens')}/{p.get('new_tokens')}  "
              f"text={p.get('generated_text')!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
