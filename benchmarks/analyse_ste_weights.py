"""
Pre/Post STE weight comparison.

Compares weight distributions before and after STE training to understand
what STE actually does to the weights. Answers:
  - Did STE move weights closer to {-alpha, 0, +alpha}?
  - Did all layer types shift equally?
  - Which layers changed most during STE?
  - Did distributions become more bimodal/trimodal?

Since Day 4 didn't save model weights, this script runs a SHORT STE
training (50 steps, ~25 min) to collect before/after weight snapshots.

Saves comparison data to data/tinyllama_ste_comparison.json.

Usage:
    python benchmarks/analyse_ste_weights.py
    python benchmarks/analyse_ste_weights.py --steps 50
    python benchmarks/analyse_ste_weights.py --analyse-only  # from saved JSON

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.ste import TernaryLinearSTE
from terncore.ste_trainer import STETrainer

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = _BENCH_DIR.parent / "data"
JSON_PATH = DATA_DIR / "tinyllama_ste_comparison.json"
DEFAULT_STEPS = 50
DEFAULT_LR = 1e-4
DEFAULT_SEQ_LEN = 256
THRESHOLD = 0.7
SEED = 42


def parse_layer_type(name: str) -> str:
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        if proj in name:
            return proj
    if "lm_head" in name:
        return "lm_head"
    return name


def compute_layer_stats(weight: torch.Tensor, threshold: float = THRESHOLD) -> dict:
    """Compute weight statistics relevant to ternary quantisation."""
    w = weight.float()
    abs_w = w.abs()

    q = TernaryQuantizer(threshold=threshold)
    ternary, alpha = q.quantize(w)
    reconstructed = ternary * alpha
    w_norm = torch.norm(w).item()

    n_zero = (ternary == 0).sum().item()
    n_pos = (ternary == 1).sum().item()
    n_neg = (ternary == -1).sum().item()

    return {
        "mean": w.mean().item(),
        "std": w.std().item(),
        "abs_mean": abs_w.mean().item(),
        "sparsity": n_zero / w.numel(),
        "pos_frac": n_pos / w.numel(),
        "neg_frac": n_neg / w.numel(),
        "alpha": alpha.item(),
        "quant_error": torch.norm(w - reconstructed).item() / w_norm if w_norm > 0 else 0,
        "weight_norm_l2": w_norm,
    }


def run_ste_comparison(
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> list[dict]:
    """Run short STE training and compare before/after weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.manual_seed(SEED)

    # Load model
    print(f"\n[1/5] Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Snapshot BEFORE weights
    print(f"\n[2/5] Snapshotting pre-STE weights...")
    pre_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pre_weights[name] = module.weight.data.clone()
            pre_stats = compute_layer_stats(module.weight.data)
            pre_weights[name + "_stats"] = pre_stats

    # Load training data
    print(f"\n[3/5] Loading WikiText-2 train split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(e["text"] for e in dataset if e["text"].strip())
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    import random
    rng = random.Random(SEED)
    chunks = []
    for i in range(0, len(full_ids) - seq_len, seq_len):
        chunks.append(full_ids[i:i+seq_len].unsqueeze(0))
    rng.shuffle(chunks)
    print(f"  {len(chunks)} chunks x {seq_len} tokens")

    # Train with STE
    print(f"\n[4/5] Running STE training ({steps} steps)...")
    trainer = STETrainer(model=model, threshold=THRESHOLD, lr=lr, log_every=max(1, steps // 5))
    converted, protected = trainer.setup()
    print(f"  Converted: {converted}, Protected: {protected}")

    result = trainer.train(data_iterator=chunks, num_steps=steps)
    print(f"  Loss: {result.initial_loss:.4f} -> {result.final_loss:.4f}")
    print(f"  Time: {result.total_time_s:.1f}s")

    # Snapshot AFTER weights and compare
    print(f"\n[5/5] Comparing pre/post weights...")
    comparisons = []

    for name, module in model.named_modules():
        if not isinstance(module, TernaryLinearSTE):
            continue

        pre_w = pre_weights.get(name)
        if pre_w is None:
            continue

        post_w = module.weight.data.clone()
        pre_s = pre_weights[name + "_stats"]
        post_s = compute_layer_stats(post_w)

        # Weight change metrics
        weight_diff = post_w - pre_w
        layer_type = parse_layer_type(name)
        m = re.search(r"layers\.(\d+)", name)
        block_idx = int(m.group(1)) if m else -1

        comparison = {
            "name": name,
            "type": layer_type,
            "block_index": block_idx,
            "num_params": post_w.numel(),
            "pre": pre_s,
            "post": post_s,
            "changes": {
                "mean_shift": post_s["mean"] - pre_s["mean"],
                "std_change": post_s["std"] - pre_s["std"],
                "std_change_pct": 100 * (post_s["std"] - pre_s["std"]) / pre_s["std"] if pre_s["std"] > 0 else 0,
                "abs_mean_change": post_s["abs_mean"] - pre_s["abs_mean"],
                "sparsity_change": post_s["sparsity"] - pre_s["sparsity"],
                "quant_error_change": post_s["quant_error"] - pre_s["quant_error"],
                "alpha_change": post_s["alpha"] - pre_s["alpha"],
                "weight_diff_norm": torch.norm(weight_diff).item(),
                "weight_diff_relative": torch.norm(weight_diff).item() / torch.norm(pre_w).item() if torch.norm(pre_w).item() > 0 else 0,
                "max_abs_change": weight_diff.abs().max().item(),
                "mean_abs_change": weight_diff.abs().mean().item(),
            },
        }
        comparisons.append(comparison)

    # Sort by relative weight change (most changed first)
    comparisons.sort(key=lambda c: c["changes"]["weight_diff_relative"], reverse=True)

    return comparisons


def print_report(comparisons: list[dict]) -> None:
    """Print formatted comparison report."""
    print()
    print("=" * 70)
    print("  Pre/Post STE Weight Comparison")
    print("=" * 70)

    # Per-type summary
    type_changes: dict[str, list] = defaultdict(list)
    for c in comparisons:
        type_changes[c["type"]].append(c)

    print()
    print("PER-TYPE CHANGES")
    print("-" * 70)
    header = (f"{'Type':<12s} | {'Count':>5s} | {'Rel Diff':>8s} | {'QE Change':>9s} | "
              f"{'Spars Change':>12s} | {'Std Change':>10s}")
    print(header)
    print("-" * len(header))

    for layer_type in sorted(type_changes):
        layers = type_changes[layer_type]
        rel_diffs = [c["changes"]["weight_diff_relative"] for c in layers]
        qe_changes = [c["changes"]["quant_error_change"] for c in layers]
        sp_changes = [c["changes"]["sparsity_change"] for c in layers]
        std_changes = [c["changes"]["std_change_pct"] for c in layers]

        print(f"{layer_type:<12s} | {len(layers):5d} | "
              f"{np.mean(rel_diffs):8.4f} | {np.mean(qe_changes):+9.4f} | "
              f"{np.mean(sp_changes):+11.4f} | {np.mean(std_changes):+9.2f}%")

    # Top 10 most changed
    print()
    print("TOP 10 MOST CHANGED LAYERS")
    print("-" * 70)
    for i, c in enumerate(comparisons[:10], 1):
        ch = c["changes"]
        print(f"  {i:2d}. {c['name']:<50s}")
        print(f"      rel_diff={ch['weight_diff_relative']:.4f}  "
              f"QE: {c['pre']['quant_error']:.4f} -> {c['post']['quant_error']:.4f}  "
              f"sparsity: {c['pre']['sparsity']:.3f} -> {c['post']['sparsity']:.3f}")

    # Bottom 5 least changed
    print()
    print("BOTTOM 5 LEAST CHANGED LAYERS")
    print("-" * 70)
    for c in comparisons[-5:]:
        ch = c["changes"]
        print(f"      {c['name']:<50s} rel_diff={ch['weight_diff_relative']:.4f}")

    # Key findings
    print()
    print("KEY FINDINGS")
    print("-" * 70)

    # Did quant error decrease?
    qe_decreased = sum(1 for c in comparisons if c["changes"]["quant_error_change"] < 0)
    print(f"  Quant error decreased: {qe_decreased}/{len(comparisons)} layers "
          f"({100*qe_decreased/len(comparisons):.0f}%)")

    # Did sparsity change?
    sp_increased = sum(1 for c in comparisons if c["changes"]["sparsity_change"] > 0.001)
    sp_decreased = sum(1 for c in comparisons if c["changes"]["sparsity_change"] < -0.001)
    print(f"  Sparsity increased: {sp_increased}, decreased: {sp_decreased}, unchanged: {len(comparisons)-sp_increased-sp_decreased}")

    # Did std change?
    avg_std_change = np.mean([c["changes"]["std_change_pct"] for c in comparisons])
    print(f"  Average std change: {avg_std_change:+.2f}%")

    # Which type changed most?
    type_avg_diff = {t: np.mean([c["changes"]["weight_diff_relative"] for c in cs])
                     for t, cs in type_changes.items()}
    most_changed = max(type_avg_diff, key=type_avg_diff.get)
    least_changed = min(type_avg_diff, key=type_avg_diff.get)
    print(f"  Most changed type:  {most_changed} (avg rel_diff={type_avg_diff[most_changed]:.4f})")
    print(f"  Least changed type: {least_changed} (avg rel_diff={type_avg_diff[least_changed]:.4f})")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/Post STE weight comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--analyse-only", action="store_true",
                        help="Load from saved JSON")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    if args.analyse_only:
        comparisons = json.loads(JSON_PATH.read_text())
    else:
        comparisons = run_ste_comparison(
            model_id=args.model, steps=args.steps, lr=args.lr,
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(comparisons, indent=2) + "\n")
        print(f"\n  Saved to {JSON_PATH}")

    if args.json_only:
        print(json.dumps(comparisons, indent=2))
    else:
        print_report(comparisons)


if __name__ == "__main__":
    main()
