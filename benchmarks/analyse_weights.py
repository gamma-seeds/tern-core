"""
Comprehensive weight analysis for TinyLlama-1.1B.

Loads the model ONCE, extracts detailed statistics from every Linear layer,
saves to JSON and CSV for offline analysis. Cross-references with Day 2
sensitivity data to discover patterns.

Usage:
    python benchmarks/analyse_weights.py
    python benchmarks/analyse_weights.py --extract-only
    python benchmarks/analyse_weights.py --analyse-only  # from saved JSON
    python benchmarks/analyse_weights.py --json-only

Patent 4: Progressive Compression — weight analysis for ternary tolerance.
Patent 7: Sparsity optimisation — zero-weight distribution analysis.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure tern-core is importable
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))

from terncore.arithmetic.quantizer import TernaryQuantizer

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = _BENCH_DIR.parent / "data"
JSON_PATH = DATA_DIR / "tinyllama_weight_analysis.json"
CSV_PATH = DATA_DIR / "tinyllama_layer_summary.csv"
REPORT_PATH = DATA_DIR / "tinyllama_weight_report.md"
THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 0.95]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
HISTOGRAM_BINS = 50

# Day 2 sensitivity data (layer_name -> ratio vs FP32 baseline)
# From eval_sensitivity.py results (4096 tokens, threshold 0.7)
# Full 155-layer data extracted from RESULTS.md
SENSITIVITY_DATA = {
    "model.layers.2.mlp.down_proj": 9609.3,
    "model.layers.5.self_attn.q_proj": 2.61,
    "model.layers.5.self_attn.k_proj": 2.47,
    "model.layers.4.self_attn.k_proj": 2.32,
    "model.layers.4.self_attn.q_proj": 2.06,
    "model.layers.6.self_attn.k_proj": 1.86,
    "model.layers.8.self_attn.k_proj": 1.57,
    "model.layers.6.self_attn.q_proj": 1.49,
    "model.layers.8.self_attn.q_proj": 1.43,
    "lm_head": 1.40,
    # Bottom-10 from RESULTS.md
    "model.layers.12.self_attn.o_proj": 1.003,
    "model.layers.1.mlp.gate_proj": 1.003,
    "model.layers.10.self_attn.o_proj": 1.003,
    "model.layers.20.self_attn.v_proj": 1.003,
    "model.layers.4.self_attn.o_proj": 1.003,
    "model.layers.13.self_attn.v_proj": 1.002,
    "model.layers.9.self_attn.o_proj": 1.002,
    "model.layers.17.self_attn.v_proj": 1.002,
    "model.layers.14.self_attn.v_proj": 1.002,
    "model.layers.3.self_attn.v_proj": 0.999,
}


# ═══════════════════════════════════════════════════════════════
# Layer name parsing
# ═══════════════════════════════════════════════════════════════


def parse_layer_name(name: str) -> tuple[str, int]:
    """Extract layer type and block index from layer name.

    Returns (layer_type, block_index). block_index is -1 for non-block layers.
    """
    # Match patterns like model.layers.0.self_attn.q_proj
    m = re.search(r"layers\.(\d+)\.(.*)", name)
    if m:
        block_idx = int(m.group(1))
        suffix = m.group(2)
        # Extract the projection type
        for proj_type in ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"):
            if proj_type in suffix:
                return proj_type, block_idx
        return suffix, block_idx
    if "lm_head" in name:
        return "lm_head", -1
    if "embed" in name:
        return "embed", -1
    return name, -1


# ═══════════════════════════════════════════════════════════════
# Weight extraction
# ═══════════════════════════════════════════════════════════════


def compute_effective_rank(weight: torch.Tensor, sample_size: int = 256) -> float:
    """Approximate effective rank via SVD on a random submatrix.

    Effective rank = exp(entropy of normalized singular values).
    Uses sampling for large matrices to keep runtime reasonable.
    """
    h, w = weight.shape
    if h > sample_size or w > sample_size:
        row_idx = torch.randperm(h)[:min(h, sample_size)]
        col_idx = torch.randperm(w)[:min(w, sample_size)]
        sub = weight[row_idx][:, col_idx]
    else:
        sub = weight

    try:
        s = torch.linalg.svdvals(sub.float())
        s = s[s > 1e-10]  # filter numerical zeros
        s_norm = s / s.sum()
        entropy = -(s_norm * torch.log(s_norm)).sum().item()
        return math.exp(entropy)
    except Exception:
        return 0.0


def extract_layer_stats(
    name: str, weight: torch.Tensor, sensitivity_data: dict,
) -> dict:
    """Extract comprehensive statistics from a single weight tensor."""
    w = weight.float()
    abs_w = w.abs()
    w_np = w.detach().numpy().ravel()

    layer_type, block_index = parse_layer_name(name)

    # Basic distribution statistics
    stats: dict[str, Any] = {
        "name": name,
        "type": layer_type,
        "block_index": block_index,
        "shape": list(weight.shape),
        "num_params": weight.numel(),
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
        "median": w.median().item(),
        "abs_mean": abs_w.mean().item(),
    }

    # Skewness and kurtosis (manual computation)
    m = w.mean()
    s = w.std()
    if s > 0:
        z = (w - m) / s
        stats["skewness"] = z.pow(3).mean().item()
        stats["kurtosis"] = z.pow(4).mean().item() - 3.0  # excess kurtosis
    else:
        stats["skewness"] = 0.0
        stats["kurtosis"] = 0.0

    # Ternary-relevant statistics across thresholds
    sparsity_at = {}
    alpha_at = {}
    quant_error_at = {}

    for t in THRESHOLDS:
        q = TernaryQuantizer(threshold=t)
        ternary, alpha = q.quantize(w)
        reconstructed = ternary * alpha

        n_zero = (ternary == 0).sum().item()
        sparsity_at[str(t)] = n_zero / w.numel()
        alpha_at[str(t)] = alpha.item()

        # Relative quantisation error: ||W - W_q|| / ||W||
        w_norm = torch.norm(w).item()
        if w_norm > 0:
            quant_error_at[str(t)] = torch.norm(w - reconstructed).item() / w_norm
        else:
            quant_error_at[str(t)] = 0.0

    stats["sparsity_at_threshold"] = sparsity_at
    stats["alpha_at_threshold"] = alpha_at
    stats["quantisation_error"] = quant_error_at

    # Percentiles
    pct_values = np.percentile(w_np, PERCENTILES)
    stats["percentiles"] = {str(p): float(v) for p, v in zip(PERCENTILES, pct_values)}

    # Histogram
    counts, edges = np.histogram(w_np, bins=HISTOGRAM_BINS)
    stats["histogram"] = {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }

    # Norms
    stats["weight_norm_l1"] = torch.norm(w, p=1).item()
    stats["weight_norm_l2"] = torch.norm(w, p=2).item()
    stats["weight_norm_inf"] = torch.norm(w, p=float("inf")).item()

    # Effective rank (sampled SVD)
    if w.dim() == 2:
        stats["effective_rank"] = compute_effective_rank(w)
    else:
        stats["effective_rank"] = 0.0

    # Cross-reference with Day 2 sensitivity
    if name in sensitivity_data:
        stats["sensitivity_ratio"] = sensitivity_data[name]
    else:
        stats["sensitivity_ratio"] = None

    # Ternary friendliness score
    qe = quant_error_at.get("0.7", 1.0)
    sr = stats["sensitivity_ratio"]
    if sr is not None and sr > 0 and qe < 1.0:
        stats["ternary_friendliness"] = (1.0 - qe) / sr
    else:
        stats["ternary_friendliness"] = None

    return stats


def extract_all(model_id: str = DEFAULT_MODEL_ID) -> list[dict]:
    """Load model and extract statistics from every Linear layer."""
    from transformers import AutoModelForCausalLM

    print(f"\n[1/2] Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    print(f"\n[2/2] Extracting weight statistics...")
    t0 = time.perf_counter()

    all_stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats = extract_layer_stats(name, module.weight.data, SENSITIVITY_DATA)
            all_stats.append(stats)
            print(f"  {len(all_stats):3d}. {name:<50s} "
                  f"shape={stats['shape']}  "
                  f"qe@0.7={stats['quantisation_error']['0.7']:.4f}  "
                  f"sparsity@0.7={stats['sparsity_at_threshold']['0.7']:.1%}")

    print(f"\n  Extracted {len(all_stats)} layers in {time.perf_counter() - t0:.1f}s")

    # Assign sensitivity ranks where we have data
    layers_with_sensitivity = [s for s in all_stats if s["sensitivity_ratio"] is not None]
    layers_with_sensitivity.sort(key=lambda s: s["sensitivity_ratio"], reverse=True)
    for rank, s in enumerate(layers_with_sensitivity, 1):
        s["sensitivity_rank"] = rank

    return all_stats


# ═══════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════


def save_json(data: list[dict], path: Path = JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def save_csv(data: list[dict], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "name", "type", "block_index", "num_params",
        "mean", "std", "abs_mean", "skewness", "kurtosis",
        "sparsity_0.7", "alpha_0.7", "quant_error_0.7",
        "weight_norm_l2", "effective_rank",
        "sensitivity_ratio", "ternary_friendliness",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for s in data:
            row = {
                "name": s["name"],
                "type": s["type"],
                "block_index": s["block_index"],
                "num_params": s["num_params"],
                "mean": f"{s['mean']:.6f}",
                "std": f"{s['std']:.6f}",
                "abs_mean": f"{s['abs_mean']:.6f}",
                "skewness": f"{s['skewness']:.4f}",
                "kurtosis": f"{s['kurtosis']:.4f}",
                "sparsity_0.7": f"{s['sparsity_at_threshold']['0.7']:.4f}",
                "alpha_0.7": f"{s['alpha_at_threshold']['0.7']:.6f}",
                "quant_error_0.7": f"{s['quantisation_error']['0.7']:.6f}",
                "weight_norm_l2": f"{s['weight_norm_l2']:.4f}",
                "effective_rank": f"{s['effective_rank']:.2f}",
                "sensitivity_ratio": (f"{s['sensitivity_ratio']:.4f}"
                                      if s['sensitivity_ratio'] is not None else ""),
                "ternary_friendliness": (f"{s['ternary_friendliness']:.4f}"
                                         if s['ternary_friendliness'] is not None else ""),
            }
            writer.writerow(row)
    print(f"  Saved {path}")


def load_json(path: Path = JSON_PATH) -> list[dict]:
    return json.loads(path.read_text())


# ═══════════════════════════════════════════════════════════════
# Analysis (works from extracted data only — no model needed)
# ═══════════════════════════════════════════════════════════════


def analyse_by_type(data: list[dict]) -> dict:
    """Group layers by type, compute aggregate statistics."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in data:
        groups[s["type"]].append(s)

    results = {}
    for layer_type, layers in sorted(groups.items()):
        abs_means = [s["abs_mean"] for s in layers]
        stds = [s["std"] for s in layers]
        sparsities = [s["sparsity_at_threshold"]["0.7"] for s in layers]
        quant_errors = [s["quantisation_error"]["0.7"] for s in layers]
        sensitivities = [s["sensitivity_ratio"] for s in layers if s["sensitivity_ratio"] is not None]
        kurtoses = [s["kurtosis"] for s in layers]
        eff_ranks = [s["effective_rank"] for s in layers if s["effective_rank"] > 0]

        results[layer_type] = {
            "count": len(layers),
            "abs_mean_avg": np.mean(abs_means),
            "abs_mean_std": np.std(abs_means),
            "std_avg": np.mean(stds),
            "sparsity_avg": np.mean(sparsities),
            "sparsity_std": np.std(sparsities),
            "quant_error_avg": np.mean(quant_errors),
            "quant_error_std": np.std(quant_errors),
            "sensitivity_avg": np.mean(sensitivities) if sensitivities else None,
            "sensitivity_count": len(sensitivities),
            "kurtosis_avg": np.mean(kurtoses),
            "eff_rank_avg": np.mean(eff_ranks) if eff_ranks else None,
            "homogeneous": np.std(quant_errors) < 0.01,
        }

    return results


def analyse_by_depth(data: list[dict]) -> dict:
    """Analyse statistics across transformer block depth."""
    depth_groups = {
        "blocks_0_5": [],
        "blocks_6_10": [],
        "blocks_11_15": [],
        "blocks_16_21": [],
    }

    for s in data:
        bi = s["block_index"]
        if bi < 0:
            continue
        if bi <= 5:
            depth_groups["blocks_0_5"].append(s)
        elif bi <= 10:
            depth_groups["blocks_6_10"].append(s)
        elif bi <= 15:
            depth_groups["blocks_11_15"].append(s)
        else:
            depth_groups["blocks_16_21"].append(s)

    results = {}
    for group_name, layers in depth_groups.items():
        if not layers:
            continue
        quant_errors = [s["quantisation_error"]["0.7"] for s in layers]
        sensitivities = [s["sensitivity_ratio"] for s in layers if s["sensitivity_ratio"] is not None]
        abs_means = [s["abs_mean"] for s in layers]
        stds = [s["std"] for s in layers]

        results[group_name] = {
            "count": len(layers),
            "quant_error_avg": np.mean(quant_errors),
            "sensitivity_avg": np.mean(sensitivities) if sensitivities else None,
            "abs_mean_avg": np.mean(abs_means),
            "std_avg": np.mean(stds),
        }

    return results


def analyse_correlation(data: list[dict]) -> dict:
    """Compute correlation between quantisation error and sensitivity."""
    pairs = [(s["quantisation_error"]["0.7"], s["sensitivity_ratio"])
             for s in data if s["sensitivity_ratio"] is not None]

    if len(pairs) < 3:
        return {"n": len(pairs), "pearson_r": None, "note": "insufficient data"}

    qe_arr = np.array([p[0] for p in pairs])
    sr_arr = np.array([p[1] for p in pairs])

    # Pearson correlation
    r = np.corrcoef(qe_arr, sr_arr)[0, 1]

    # Also try log(sensitivity) since layers.2.mlp.down_proj is extreme
    sr_log = np.log(sr_arr + 1e-10)
    r_log = np.corrcoef(qe_arr, sr_log)[0, 1]

    # Without the outlier (layers.2.mlp.down_proj has ratio 9609)
    no_outlier = [(q, s) for q, s in pairs if s < 100]
    if len(no_outlier) >= 3:
        qe_no = np.array([p[0] for p in no_outlier])
        sr_no = np.array([p[1] for p in no_outlier])
        r_no_outlier = np.corrcoef(qe_no, sr_no)[0, 1]
    else:
        r_no_outlier = None

    return {
        "n": len(pairs),
        "pearson_r": float(r),
        "pearson_r_log_sensitivity": float(r_log),
        "pearson_r_no_outlier": float(r_no_outlier) if r_no_outlier is not None else None,
        "outlier_layer": "model.layers.2.mlp.down_proj",
        "outlier_ratio": 9609.3,
    }


def rank_friendliness(data: list[dict]) -> list[dict]:
    """Rank layers by ternary friendliness score."""
    scored = [(s["name"], s["ternary_friendliness"], s["type"], s["block_index"])
              for s in data if s["ternary_friendliness"] is not None]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"name": n, "score": s, "type": t, "block": b} for n, s, t, b in scored]


def find_bimodal_layers(data: list[dict]) -> list[dict]:
    """Identify layers with bimodal or trimodal weight distributions.

    Uses kurtosis: bimodal distributions tend to have negative excess kurtosis
    (platykurtic). Also checks if the histogram has two peaks.
    """
    bimodal = []
    for s in data:
        hist_counts = s["histogram"]["counts"]
        # Find peaks (bins higher than both neighbors)
        peaks = 0
        for i in range(1, len(hist_counts) - 1):
            if hist_counts[i] > hist_counts[i-1] and hist_counts[i] > hist_counts[i+1]:
                if hist_counts[i] > max(hist_counts) * 0.1:  # significant peak
                    peaks += 1

        if peaks >= 2 or s["kurtosis"] < -0.5:
            bimodal.append({
                "name": s["name"],
                "type": s["type"],
                "block": s["block_index"],
                "kurtosis": s["kurtosis"],
                "peaks": peaks,
                "std": s["std"],
            })

    return bimodal


def find_outlier_weights(data: list[dict]) -> list[dict]:
    """Find layers with extreme weight values (>5 std from mean)."""
    outlier_layers = []
    for s in data:
        # Check if max or min is far from mean
        spread = s["max"] - s["min"]
        if s["std"] > 0:
            max_z = (s["max"] - s["mean"]) / s["std"]
            min_z = (s["mean"] - s["min"]) / s["std"]
            if max_z > 5.0 or min_z > 5.0:
                outlier_layers.append({
                    "name": s["name"],
                    "type": s["type"],
                    "block": s["block_index"],
                    "max_z": max_z,
                    "min_z": min_z,
                    "max_val": s["max"],
                    "min_val": s["min"],
                    "std": s["std"],
                })

    outlier_layers.sort(key=lambda x: max(x["max_z"], x["min_z"]), reverse=True)
    return outlier_layers


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════


def print_report(data: list[dict]) -> str:
    """Generate comprehensive analysis report. Returns markdown string."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 70)
    p("  TinyLlama-1.1B Weight Analysis")
    p("=" * 70)

    # ── Type profiles ──────────────────────────────────────────
    type_stats = analyse_by_type(data)
    p()
    p("LAYER TYPE PROFILES")
    p("-" * 70)
    header = (f"{'Type':<12s} | {'Count':>5s} | {'Mean|W|':>9s} | {'Std':>9s} | "
              f"{'Spars@0.7':>9s} | {'QE@0.7':>8s} | {'Kurtosis':>8s} | "
              f"{'EffRank':>7s}")
    p(header)
    p("-" * len(header))

    # Order by quant error (friendliest first)
    for layer_type in sorted(type_stats, key=lambda t: type_stats[t]["quant_error_avg"]):
        ts = type_stats[layer_type]
        er = f"{ts['eff_rank_avg']:.1f}" if ts['eff_rank_avg'] else "N/A"
        p(f"{layer_type:<12s} | {ts['count']:5d} | {ts['abs_mean_avg']:9.6f} | "
          f"{ts['std_avg']:9.6f} | {ts['sparsity_avg']:8.1%} | "
          f"{ts['quant_error_avg']:8.4f} | {ts['kurtosis_avg']:8.2f} | {er:>7s}")

    # Check homogeneity within types
    p()
    p("Type homogeneity (are all layers of a type similar?):")
    for layer_type, ts in sorted(type_stats.items()):
        label = "YES" if ts["homogeneous"] else "NO"
        p(f"  {layer_type:<12s}: QE std={ts['quant_error_std']:.4f}  "
          f"sparsity std={ts['sparsity_std']:.4f}  → {label}")

    # ── Depth analysis ─────────────────────────────────────────
    depth_stats = analyse_by_depth(data)
    p()
    p("BLOCK DEPTH ANALYSIS")
    p("-" * 70)
    for group_name in ["blocks_0_5", "blocks_6_10", "blocks_11_15", "blocks_16_21"]:
        if group_name not in depth_stats:
            continue
        ds = depth_stats[group_name]
        sens_str = f"{ds['sensitivity_avg']:.2f}x" if ds['sensitivity_avg'] else "N/A"
        p(f"  {group_name:<16s}: avg QE={ds['quant_error_avg']:.4f}  "
          f"avg sens={sens_str}  "
          f"avg |W|={ds['abs_mean_avg']:.6f}  "
          f"({ds['count']} layers)")

    # ── Correlation analysis ───────────────────────────────────
    corr = analyse_correlation(data)
    p()
    p("QUANT ERROR vs SENSITIVITY CORRELATION")
    p("-" * 70)
    if corr["pearson_r"] is not None:
        p(f"  Pearson r (all data):        {corr['pearson_r']:.4f}  (n={corr['n']})")
        p(f"  Pearson r (log sensitivity):  {corr['pearson_r_log_sensitivity']:.4f}")
        if corr["pearson_r_no_outlier"] is not None:
            p(f"  Pearson r (excl. outlier):    {corr['pearson_r_no_outlier']:.4f}")
        p(f"  Outlier: {corr['outlier_layer']} (ratio {corr['outlier_ratio']:.1f}x)")

        r = abs(corr['pearson_r_no_outlier'] or corr['pearson_r'])
        if r > 0.8:
            p("  Interpretation: STRONG predictor — quant error can proxy for sensitivity")
        elif r > 0.5:
            p("  Interpretation: MODERATE predictor — some predictive value")
        elif r > 0.3:
            p("  Interpretation: WEAK predictor — limited correlation")
        else:
            p("  Interpretation: NO correlation — quant error does not predict sensitivity")
    else:
        p("  Insufficient data for correlation analysis")

    # ── Friendliness ranking ───────────────────────────────────
    friendliness = rank_friendliness(data)
    p()
    p("TERNARY FRIENDLINESS RANKING")
    p("-" * 70)
    if friendliness:
        p("Top 10 most ternary-friendly:")
        for i, f in enumerate(friendliness[:10], 1):
            p(f"  {i:2d}. {f['name']:<50s} score={f['score']:.4f} ({f['type']})")
        p()
        p("Bottom 5 least ternary-friendly:")
        for f in friendliness[-5:]:
            p(f"      {f['name']:<50s} score={f['score']:.4f} ({f['type']})")

    # ── Bimodal layer detection ────────────────────────────────
    bimodal = find_bimodal_layers(data)
    p()
    p("BIMODAL/TRIMODAL DISTRIBUTION DETECTION")
    p("-" * 70)
    if bimodal:
        p(f"  Found {len(bimodal)} layers with multi-peak distributions:")
        for b in bimodal[:10]:
            p(f"    {b['name']:<50s} kurtosis={b['kurtosis']:.2f}  peaks={b['peaks']}")
    else:
        p("  No bimodal distributions detected (all unimodal)")

    # ── Outlier weights ────────────────────────────────────────
    outliers = find_outlier_weights(data)
    p()
    p("OUTLIER WEIGHT DETECTION (>5 std from mean)")
    p("-" * 70)
    if outliers:
        p(f"  Found {len(outliers)} layers with extreme outlier weights:")
        for o in outliers[:10]:
            p(f"    {o['name']:<50s} max_z={o['max_z']:.1f}  "
              f"min_z={o['min_z']:.1f}  range=[{o['min_val']:.4f}, {o['max_val']:.4f}]")
    else:
        p("  No layers with >5 std outlier weights")

    # ── Discovery summary ──────────────────────────────────────
    p()
    p("=" * 70)
    p("  TOP DISCOVERIES")
    p("=" * 70)

    # Find the most interesting patterns
    discoveries = []

    # 1. Check if all v_proj layers are similar
    if "v_proj" in type_stats:
        ts = type_stats["v_proj"]
        if ts["homogeneous"]:
            discoveries.append(
                f"v_proj layers are HOMOGENEOUS (QE std={ts['quant_error_std']:.4f}). "
                f"All 22 behave similarly under quantisation → safe to treat as a group."
            )

    # 2. Compare v_proj vs q_proj
    if "v_proj" in type_stats and "q_proj" in type_stats:
        v_qe = type_stats["v_proj"]["quant_error_avg"]
        q_qe = type_stats["q_proj"]["quant_error_avg"]
        ratio = q_qe / v_qe if v_qe > 0 else 0
        if ratio > 1.1:
            discoveries.append(
                f"q_proj has {ratio:.1f}x higher quant error than v_proj "
                f"({q_qe:.4f} vs {v_qe:.4f}). Weight statistics explain "
                f"why v_proj is more ternary-tolerant."
            )

    # 3. Depth pattern
    if "blocks_0_5" in depth_stats and "blocks_16_21" in depth_stats:
        early_qe = depth_stats["blocks_0_5"]["quant_error_avg"]
        late_qe = depth_stats["blocks_16_21"]["quant_error_avg"]
        if early_qe > late_qe * 1.05:
            discoveries.append(
                f"Early blocks (0-5) have {early_qe/late_qe:.2f}x higher quant error "
                f"than late blocks (16-21). Consistent with Day 2 finding that "
                f"early attention layers are more sensitive."
            )

    # 4. Correlation strength
    if corr["pearson_r_no_outlier"] is not None:
        r = corr["pearson_r_no_outlier"]
        if abs(r) > 0.5:
            discoveries.append(
                f"Quant error correlates with sensitivity (r={r:.3f} excl. outlier). "
                f"Can be used as a cheap proxy for sensitivity analysis."
            )
        else:
            discoveries.append(
                f"Quant error does NOT strongly correlate with sensitivity "
                f"(r={r:.3f} excl. outlier). Layer sensitivity is driven by "
                f"functional role, not weight distribution shape."
            )

    for i, d in enumerate(discoveries, 1):
        p(f"\n  {i}. {d}")

    p()
    p("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyLlama weight analysis")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract stats only, skip analysis")
    parser.add_argument("--analyse-only", action="store_true",
                        help="Analyse from saved JSON (no model loading)")
    parser.add_argument("--json-only", action="store_true",
                        help="Print raw JSON only")
    args = parser.parse_args()

    if args.analyse_only:
        print(f"Loading from {JSON_PATH}...")
        data = load_json(JSON_PATH)
    else:
        data = extract_all(args.model)
        save_json(data)
        save_csv(data)

    if args.json_only:
        print(json.dumps(data, indent=2))
        return

    if not args.extract_only:
        report_text = print_report(data)
        # Save markdown report
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report_text + "\n")
        print(f"\n  Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
