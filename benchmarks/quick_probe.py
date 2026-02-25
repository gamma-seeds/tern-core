"""
Universal smoke test for ternary model quality.

Loads a HuggingFace model, evaluates FP32 and ternary perplexity on a
small token subset, ranks layers by gradient-based sensitivity (Fisher
information approximation), and provides a fast assessment.

Target runtime: <2 minutes.

Usage:
    python benchmarks/quick_probe.py
    python benchmarks/quick_probe.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/quick_probe.py --json-only

Patent 4: Progressive Compression — fast sensitivity estimation.
Patent 36: Deterministic execution guarantee.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Ensure tern-core is importable
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.engine.inference import TernaryInferenceEngine

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_THRESHOLD = 0.7
EVAL_TOKENS = 512  # tiny subset for speed
STRIDE = 512
SEED = 42

# Day 2 brute-force top-10 for validation
DAY2_TOP10 = [
    "model.layers.2.mlp.down_proj",
    "model.layers.5.self_attn.q_proj",
    "model.layers.5.self_attn.k_proj",
    "model.layers.4.self_attn.k_proj",
    "model.layers.4.self_attn.q_proj",
    "model.layers.6.self_attn.k_proj",
    "model.layers.8.self_attn.k_proj",
    "model.layers.6.self_attn.q_proj",
    "model.layers.8.self_attn.q_proj",
    "lm_head",
]


# ═══════════════════════════════════════════════════════════════
# Perplexity evaluation (minimal, fast)
# ═══════════════════════════════════════════════════════════════


def quick_ppl(
    model: nn.Module, input_ids: torch.Tensor, max_length: int,
) -> float:
    """Compute perplexity on a small token set."""
    model.eval()
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0

    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            nlls.append(outputs.loss.item() * trg_len)

        prev_end = end_loc
        if end_loc == seq_len:
            break

    try:
        return math.exp(sum(nlls) / prev_end)
    except OverflowError:
        return float("inf")


# ═══════════════════════════════════════════════════════════════
# Gradient-based sensitivity (Fisher information approximation)
# ═══════════════════════════════════════════════════════════════


def gradient_sensitivity(
    model: nn.Module, input_ids: torch.Tensor,
) -> list[tuple[str, float]]:
    """Rank layers by gradient magnitude from one forward-backward pass.

    Uses the Fisher information approximation:
        sensitivity(layer) = mean(|grad| * |weight|)

    This captures how much a small weight perturbation affects the loss —
    layers where gradients are large AND weights are large are the most
    sensitive to quantisation.
    """
    model.train()
    model.zero_grad()

    # Enable gradients for all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    # One forward-backward pass
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

    # Collect sensitivity scores for Linear layers
    sensitivities = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            # Fisher information approximation: E[grad^2] ≈ grad * weight
            fisher = (module.weight.grad.abs() * module.weight.data.abs()).mean().item()
            grad_norm = module.weight.grad.norm().item()
            weight_norm = module.weight.data.norm().item()
            sensitivities.append((name, fisher, grad_norm, weight_norm))

    # Disable gradients again
    for p in model.parameters():
        p.requires_grad_(False)
    model.zero_grad()
    model.eval()

    # Sort by Fisher sensitivity (highest first)
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    return sensitivities


# ═══════════════════════════════════════════════════════════════
# Main probe
# ═══════════════════════════════════════════════════════════════


def run_probe(
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Run the complete quick probe."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.manual_seed(SEED)
    t_total = time.perf_counter()

    # Load model
    print(f"\n[1/5] Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    max_length = getattr(model.config, "max_position_embeddings", 2048)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Load small eval set
    print(f"\n[2/5] Loading WikiText-2 ({EVAL_TOKENS} tokens)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(e["text"] for e in dataset if e["text"].strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :EVAL_TOKENS]
    print(f"  Tokens: {input_ids.shape[1]}")

    # FP32 baseline PPL
    print(f"\n[3/5] Evaluating FP32 perplexity...")
    t0 = time.perf_counter()
    fp32_ppl = quick_ppl(model, input_ids, max_length)
    print(f"  FP32 PPL ({EVAL_TOKENS} tokens): {fp32_ppl:.2f} ({time.perf_counter() - t0:.1f}s)")

    # Gradient sensitivity (one forward-backward pass)
    print(f"\n[4/5] Computing gradient sensitivity...")
    t0 = time.perf_counter()
    sensitivities = gradient_sensitivity(model, input_ids)
    grad_time = time.perf_counter() - t0
    print(f"  Computed in {grad_time:.1f}s")

    # Ternary PPL (convert all layers)
    print(f"\n[5/5] Evaluating ternary perplexity...")
    t0 = time.perf_counter()
    engine = TernaryInferenceEngine(threshold=threshold, protect_lm_head=True)
    report = engine.convert(model, sensitivity_analysis=False)
    ternary_ppl = quick_ppl(model, input_ids, max_length)
    print(f"  Ternary PPL ({EVAL_TOKENS} tokens): {ternary_ppl:.2f} ({time.perf_counter() - t0:.1f}s)")

    total_time = time.perf_counter() - t_total

    # Sparsity
    from terncore.arithmetic.linear import TernaryLinear
    sparsities = []
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            sparsities.append(m.sparsity)
    avg_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0

    # Validate gradient ranking vs Day 2
    grad_top10 = [s[0] for s in sensitivities[:10]]
    overlap = len(set(grad_top10) & set(DAY2_TOP10))

    # Recommendation
    gap_pct = 100 * (ternary_ppl - fp32_ppl) / fp32_ppl if fp32_ppl > 0 else float("inf")
    if gap_pct < 5:
        recommendation = "TERNARY_READY"
    elif gap_pct < 50:
        recommendation = "NEEDS_MIXED_PRECISION"
    else:
        recommendation = "NEEDS_STE_TRAINING"

    result = {
        "model_id": model_id,
        "threshold": threshold,
        "eval_tokens": EVAL_TOKENS,
        "fp32_ppl": round(fp32_ppl, 4),
        "ternary_ppl": round(ternary_ppl, 4) if math.isfinite(ternary_ppl) else None,
        "gap_pct": round(gap_pct, 2) if math.isfinite(gap_pct) else None,
        "total_layers": report.total_layers,
        "converted_layers": report.converted_layers,
        "sparsity": round(avg_sparsity, 4),
        "compression_ratio": round(report.compression_ratio, 2),
        "top_10_gradient_sensitivity": [
            {"name": s[0], "fisher": round(s[1], 6), "grad_norm": round(s[2], 6)}
            for s in sensitivities[:10]
        ],
        "gradient_vs_day2_overlap": overlap,
        "gradient_vs_day2_top10_match": f"{overlap}/10",
        "recommendation": recommendation,
        "total_time_s": round(total_time, 1),
    }

    return result


def print_report(result: dict) -> None:
    """Print formatted probe report."""
    print()
    print("=" * 70)
    print(f"  Quick Probe: {result['model_id']}")
    print("=" * 70)
    print(f"  FP32 PPL ({result['eval_tokens']} tokens):     {result['fp32_ppl']:.2f}")
    ternary_str = f"{result['ternary_ppl']:.2f}" if result['ternary_ppl'] else "inf"
    print(f"  Ternary PPL ({result['eval_tokens']} tokens):  {ternary_str}")
    gap_str = f"+{result['gap_pct']:.1f}%" if result['gap_pct'] else "+inf"
    print(f"  Gap vs FP32:            {gap_str}")
    print(f"  Total layers:           {result['total_layers']}")
    print(f"  Converted layers:       {result['converted_layers']}")
    print(f"  Sparsity (threshold {result['threshold']}): {result['sparsity']:.1%}")
    print(f"  Compression ratio:      {result['compression_ratio']:.1f}x")
    print()

    print("  Top-10 sensitive (gradient magnitude):")
    for i, s in enumerate(result["top_10_gradient_sensitivity"], 1):
        print(f"    {i:2d}. {s['name']:<50s}  fisher={s['fisher']:.6f}")

    print()
    print(f"  Gradient vs Day 2 ranking overlap: {result['gradient_vs_day2_top10_match']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Total time: {result['total_time_s']:.1f}s")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick probe for ternary model quality")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = run_probe(model_id=args.model, threshold=args.threshold)

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
