"""
Standalone ternary sensitivity scan for MoE and dense artefacts.

Investigation provenance (2026-05-28):

1. The 0.5/0.55/0.6/0.7 threshold sweep on Qwen3-30B-A3B (every threshold
   collapses to single-token repetition) put us on the smeared branch.
2. The per-tensor scan on Qwen3 + Gemma-4-26B-A4B confirmed it: both
   distributions are tightly clustered (std ~0.01) around relative_error
   ~0.45, with no concentrated tail. Per-layer INT4 routing isn't the
   right lever for this kind of error.
3. ``full_convert``'s INT4 routing was dormant for Qwen3 because the
   sensitivity map was hard-coded to ``benchmarks/gemma4_e4b_dryrun.json``;
   that file's names matched only Gemma-4-family weights.

Two cheap measurements to scope the compression-quality fork before any
deep-dive:

- **per-channel relative error**: same weights, ``alpha`` per output channel
  instead of per tensor. Cheapest lever. Tells whether per-channel buys
  real headroom or marginal.
- **dense reference scan**: per-tensor relative-error distribution on a
  dense model. Mistral-7B isn't on disk; Phi-4 is THE documented dense
  collapse case (per docs/backlog.md "Phi-4 ternary recompression at
  lower threshold") and is cached.

Math (per layer, weight ``W`` of shape ``[out, in]``, threshold ``t``):

- per-tensor: ``alpha = mean(|W|)``, ``T = sign(W) * 1[|W| > t·alpha]``,
  reconstruction ``alpha·T``.
- per-channel: ``alpha_i = mean(|W[i, :]|)`` for each output row ``i``,
  ``T_i = sign(W[i, :]) * 1[|W[i, :]| > t·alpha_i]``, reconstruction
  ``alpha_i · T_i`` broadcast back.

Relative Frobenius error ``||W - recon||_F / ||W||_F`` is the
GPTQ/AWQ/SqueezeLLM per-layer perplexity-impact proxy.

Output JSON includes a ``tolerance_scan`` key (legacy schema) alongside
``layers`` so the result can feed ``full_convert``'s ``sensitivity_map_path``
parameter directly.

Usage:
    HF_HUB_OFFLINE=1 python benchmarks/sensitivity_scan_moe_2026-05-28.py \\
        --targets qwen3-30b-a3b,gemma4-26b-a4b --mode per_tensor

    HF_HUB_OFFLINE=1 python benchmarks/sensitivity_scan_moe_2026-05-28.py \\
        --targets qwen3-30b-a3b,phi-4 --mode per_channel

Copyright (c) 2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from terncore.adapters import get_adapter

ERROR_BANDS = [0.30, 0.40, 0.50, 0.54, 0.60, 0.70, 0.80, 0.90]
THIS_DIR = Path(__file__).resolve().parent

TARGETS = {
    "qwen3-30b-a3b":  ("Qwen/Qwen3-30B-A3B",                       "qwen3_moe"),
    "gemma4-26b-a4b": ("/Volumes/Syn Archive/models/source/gemma-4-26b-a4b-it",
                       "gemma4"),
    "phi-4":          ("microsoft/phi-4",                          "phi3"),
    "tinyllama":      ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",       "llama"),
}


@dataclass
class _LayerSens:
    name: str
    relative_error: float
    sparsity: float
    alpha: float        # scalar (per-tensor) or mean of per-channel alphas
    num_params: int


def _per_tensor_sensitivity(name: str, w: torch.Tensor, threshold: float) -> _LayerSens:
    w = w.float()
    abs_w = w.abs()
    alpha = abs_w.mean()
    tau = threshold * alpha
    T = torch.sign(w) * (abs_w > tau).float()
    recon = alpha * T
    w_norm = torch.norm(w).item()
    err = (torch.norm(w - recon).item() / w_norm) if w_norm > 0 else 0.0
    sparsity = (T == 0).float().mean().item()
    return _LayerSens(
        name=name, relative_error=err, sparsity=sparsity,
        alpha=float(alpha.item()), num_params=w.numel(),
    )


def _per_channel_sensitivity(name: str, w: torch.Tensor, threshold: float) -> _LayerSens:
    """alpha and threshold per output channel (row 0) of a 2-D weight."""
    w = w.float()
    if w.ndim != 2:
        # Fall back to per-tensor for non-2-D tensors (rare for ternary-eligible).
        return _per_tensor_sensitivity(name, w, threshold)
    abs_w = w.abs()
    alpha = abs_w.mean(dim=1, keepdim=True)                # [out, 1]
    tau = threshold * alpha
    T = torch.sign(w) * (abs_w > tau).float()
    recon = alpha * T                                       # broadcast back
    w_norm = torch.norm(w).item()
    err = (torch.norm(w - recon).item() / w_norm) if w_norm > 0 else 0.0
    sparsity = (T == 0).float().mean().item()
    return _LayerSens(
        name=name, relative_error=err, sparsity=sparsity,
        alpha=float(alpha.mean().item()), num_params=w.numel(),
    )


def _categorise(name: str) -> str:
    if ".mlp.experts." in name and "_proj" in name:
        return "expert"
    if ".mlp." in name and "_proj" in name and ".experts." not in name:
        return "dense_mlp"
    if ".self_attn." in name and "_proj" in name:
        return "attention"
    return "other"


def _resolve_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_id, allow_patterns=[
            "*.safetensors", "*.safetensors.index.json", "config.json",
        ],
    ))


def _iter_safetensors(model_dir: Path):
    idx = model_dir / "model.safetensors.index.json"
    w2f: dict[str, Path] = {}
    if idx.exists():
        for k, fname in json.load(open(idx))["weight_map"].items():
            w2f[k] = model_dir / fname
    else:
        for sf in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(sf), framework="pt") as f:
                for k in f.keys():
                    w2f[k] = sf
    file_to_keys: dict[Path, list[str]] = defaultdict(list)
    for k, p in w2f.items():
        file_to_keys[p].append(k)
    return file_to_keys


def scan(model_id: str, adapter_name: str, threshold: float, mode: str, label: str) -> dict:
    """Run a per-tensor or per-channel sensitivity scan on a model's weights."""
    if mode not in ("per_tensor", "per_channel"):
        raise ValueError(f"mode must be per_tensor|per_channel, got {mode!r}")
    compute = _per_tensor_sensitivity if mode == "per_tensor" else _per_channel_sensitivity

    print(f"\n===== sensitivity scan: {label}  mode={mode}  thr={threshold} =====", flush=True)
    adapter = get_adapter(adapter_name)
    model_dir = _resolve_model_dir(model_id)
    print(f"  adapter={adapter_name}  source={model_dir}", flush=True)

    file_to_keys = _iter_safetensors(model_dir)

    # Pass 1: gather shapes + stacked-tensor expansion plans.
    weight_shapes: dict[str, list[int]] = {}
    stacked_plans: dict[str, list] = {}
    for p, keys in file_to_keys.items():
        with safe_open(str(p), framework="pt") as f:
            for k in keys:
                shape = list(f.get_slice(k).get_shape())
                plan = adapter.expand_stacked(k, shape)
                if plan is None:
                    weight_shapes[k] = shape
                else:
                    stacked_plans[k] = plan
                    per_slice = [
                        s for i, s in enumerate(shape) if i != plan[0].slice_axis
                    ]
                    for sl in plan:
                        weight_shapes[sl.synthesised_name] = list(per_slice)
    classifications = adapter.classify_all(weight_shapes)
    eligible = {n for n, c in classifications.items() if c.category == "ternary_eligible"}
    print(
        f"  weights={len(weight_shapes)}  ternary-eligible={len(eligible)}  "
        f"stacked-parents={len(stacked_plans)}", flush=True,
    )

    # Pass 2: stream tensors, slice stacked parents, measure per-layer error.
    sensitivities: list[_LayerSens] = []
    t0 = time.perf_counter()
    for p, keys in file_to_keys.items():
        with safe_open(str(p), framework="pt") as f:
            for k in keys:
                if k in stacked_plans:
                    plan = stacked_plans[k]
                    parent = f.get_tensor(k)
                    axis = plan[0].slice_axis
                    for sl in plan:
                        if sl.synthesised_name not in eligible:
                            continue
                        idx = [slice(None)] * parent.ndim
                        idx[axis] = sl.slice_index
                        sensitivities.append(
                            compute(sl.synthesised_name, parent[tuple(idx)], threshold)
                        )
                    del parent
                else:
                    if k not in eligible:
                        continue
                    sensitivities.append(compute(k, f.get_tensor(k), threshold))
                if len(sensitivities) and len(sensitivities) % 2000 == 0:
                    print(
                        f"    [{len(sensitivities)}/{len(eligible)}] "
                        f"({time.perf_counter() - t0:.0f}s)", flush=True,
                    )
    dt = time.perf_counter() - t0
    print(f"  scanned {len(sensitivities)} layers in {dt:.0f}s", flush=True)

    errs = [s.relative_error for s in sensitivities]
    by_cat: dict[str, list[float]] = defaultdict(list)
    for s in sensitivities:
        by_cat[_categorise(s.name)].append(s.relative_error)

    summary = {
        "label": label,
        "model_id": model_id,
        "adapter": adapter_name,
        "mode": mode,
        "threshold": threshold,
        "n_layers": len(sensitivities),
        "elapsed_seconds": round(dt, 1),
        "relative_error": {
            "mean": round(statistics.mean(errs), 5),
            "median": round(statistics.median(errs), 5),
            "stdev": round(statistics.stdev(errs), 5) if len(errs) > 1 else 0.0,
            "min": round(min(errs), 5),
            "max": round(max(errs), 5),
            "p95": round(sorted(errs)[int(0.95 * len(errs))], 5),
            "p99": round(sorted(errs)[int(0.99 * len(errs))], 5),
        },
        "layers_above": {
            f">={b:.2f}": sum(1 for e in errs if e >= b) for b in ERROR_BANDS
        },
        "by_category": {
            cat: {
                "n": len(es),
                "mean": round(statistics.mean(es), 5),
                "median": round(statistics.median(es), 5),
                "max": round(max(es), 5),
                "above_0.54": sum(1 for e in es if e >= 0.54),
                "above_0.60": sum(1 for e in es if e >= 0.60),
            } for cat, es in by_cat.items()
        },
        "top_30_worst": [
            {
                "name": s.name,
                "relative_error": round(s.relative_error, 5),
                "sparsity": round(s.sparsity, 4),
                "num_params": s.num_params,
            } for s in sorted(sensitivities, key=lambda x: -x.relative_error)[:30]
        ],
        # Full per-layer list — durable evidence and the source full_convert
        # consumes via sensitivity_map_path.
        "layers": [
            {
                "name": s.name,
                "relative_error": round(s.relative_error, 5),
                "sparsity": round(s.sparsity, 4),
                "alpha": round(s.alpha, 5),
                "num_params": s.num_params,
                "category": _categorise(s.name),
            } for s in sensitivities
        ],
        # Legacy-schema compat (same {name, relative_error} list under a
        # ``tolerance_scan`` key) so ``full_convert``'s map loader accepts
        # this JSON without translation.
        "tolerance_scan": [
            {"name": s.name, "relative_error": round(s.relative_error, 5)}
            for s in sensitivities
        ],
    }
    return summary


def _print_summary(summary: dict) -> None:
    re = summary["relative_error"]
    print(f"\n  --- {summary['label']} ({summary['n_layers']} layers, "
          f"mode={summary['mode']}, thr={summary['threshold']}) ---", flush=True)
    print(f"  relative_error  mean={re['mean']:.4f}  median={re['median']:.4f}"
          f"  stdev={re['stdev']:.4f}  p95={re['p95']:.4f}  p99={re['p99']:.4f}"
          f"  max={re['max']:.4f}", flush=True)
    print("  layers_above:  "
          + "  ".join(f"{k}={v}" for k, v in summary["layers_above"].items()),
          flush=True)
    print("  by_category:", flush=True)
    for cat, st in summary["by_category"].items():
        print(f"    {cat:10s}  n={st['n']:5d}  mean={st['mean']:.4f}  "
              f"median={st['median']:.4f}  max={st['max']:.4f}  "
              f"≥0.54={st['above_0.54']:5d}  ≥0.60={st['above_0.60']:5d}",
              flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--targets", default="qwen3-30b-a3b,gemma4-26b-a4b",
        help=f"comma-separated; available: {','.join(TARGETS)}",
    )
    ap.add_argument(
        "--mode", default="per_tensor", choices=("per_tensor", "per_channel"),
    )
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args(argv)

    slugs = [s.strip() for s in args.targets.split(",") if s.strip()]
    suffix = "_per_channel" if args.mode == "per_channel" else ""
    summaries = []
    for slug in slugs:
        if slug not in TARGETS:
            print(f"  [skip] unknown target {slug!r}", flush=True)
            continue
        model_id, adapter = TARGETS[slug]
        out = THIS_DIR / f"sensitivity_scan_{slug}{suffix}_2026-05-28.json"
        try:
            s = scan(model_id, adapter, args.threshold, args.mode, slug)
            out.write_text(json.dumps(s, indent=1) + "\n")
            print(f"  wrote {out}", flush=True)
            _print_summary(s)
            summaries.append((slug, s))
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  [ERROR] {slug}: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    if len(summaries) >= 2:
        print("\n===== SIDE-BY-SIDE =====", flush=True)
        labels = [s[0] for s in summaries]
        print(f"  {'metric':22s}" + "".join(f"{l:>18s}" for l in labels), flush=True)
        rows = [
            ("n_layers",   lambda d: d["n_layers"]),
            ("mean re",    lambda d: f"{d['relative_error']['mean']:.4f}"),
            ("median re",  lambda d: f"{d['relative_error']['median']:.4f}"),
            ("stdev re",   lambda d: f"{d['relative_error']['stdev']:.4f}"),
            ("p95 re",     lambda d: f"{d['relative_error']['p95']:.4f}"),
            ("p99 re",     lambda d: f"{d['relative_error']['p99']:.4f}"),
            ("max re",     lambda d: f"{d['relative_error']['max']:.4f}"),
            (">= 0.54",    lambda d: d["layers_above"][">=0.54"]),
            (">= 0.60",    lambda d: d["layers_above"][">=0.60"]),
            (">= 0.70",    lambda d: d["layers_above"][">=0.70"]),
        ]
        for lbl, fn in rows:
            print(f"  {lbl:22s}" + "".join(f"{str(fn(s)):>18s}" for _, s in summaries),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
