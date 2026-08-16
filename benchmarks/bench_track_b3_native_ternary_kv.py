"""BUILD 6 Track B3 — Native-ternary KV cache compression.

Bypasses TurboQuant entirely. Quantises KV cache entries to {-1, 0, +1}
using the same symmetric threshold as weight quantisation:
  threshold = t × mean(|V|) per vector
  entries above threshold → sign, below → 0
  scale = mean(|non-zero entries|) per vector

Stored as 2-bit packed (4 values/byte) + FP16 scale per vector.
Compression ratio from actual packed bytes vs FP16/FP32 uncompressed.

Quality gate: argmax match rate (does the next-token prediction change?).
- 93% = floor (no sweep point below this)
- 99% = bell mean target (average across tokens)

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import json
import math
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PROMPTS = [
    "The capital of France is",
    "In machine learning, the gradient descent algorithm",
    "The ternary number system uses three values",
    "Quantum computing relies on superposition",
    "The fastest sorting algorithm for random data",
]
GENERATE_TOKENS = 100
OUTPUT_PATH = Path(__file__).parent / "Benchmark_Report_B3_Native_Ternary_KV.json"


def ternary_quantise_vector(v: torch.Tensor, threshold_factor: float):
    """Quantise a vector to {-1, 0, +1} with symmetric threshold.

    Returns: (codes int8, scale float, zero_ratio float, n_nonzero int)
    """
    abs_v = v.abs()
    mean_abs = abs_v.mean()
    thresh = threshold_factor * mean_abs

    codes = torch.zeros_like(v, dtype=torch.int8)
    mask_pos = v > thresh
    mask_neg = v < -thresh
    codes[mask_pos] = 1
    codes[mask_neg] = -1

    nonzero_mask = codes != 0
    n_nonzero = nonzero_mask.sum().item()
    scale = abs_v[nonzero_mask].mean().item() if n_nonzero > 0 else 0.0
    zero_ratio = 1.0 - (n_nonzero / v.numel())

    return codes, scale, zero_ratio, n_nonzero


def ternary_dequantise_vector(codes: torch.Tensor, scale: float):
    return codes.float() * scale


def pack_ternary_2bit(codes: torch.Tensor):
    """Pack ternary codes to 2-bit: 00=0, 01=+1, 10=-1.
    Returns packed bytes and byte count.
    """
    mapped = torch.zeros_like(codes, dtype=torch.uint8)
    mapped[codes == 1] = 1
    mapped[codes == -1] = 2
    n = mapped.numel()
    padded = n + (4 - n % 4) % 4
    if padded > n:
        mapped = torch.cat([mapped, torch.zeros(padded - n, dtype=torch.uint8)])
    mapped = mapped.reshape(-1, 4)
    packed = (mapped[:, 0] | (mapped[:, 1] << 2) | (mapped[:, 2] << 4) | (mapped[:, 3] << 6))
    return packed, packed.numel()


def compress_kv_cache(past_key_values, threshold_factor: float):
    """Compress entire KV cache to ternary. Returns compressed cache and metrics."""
    _TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    from tern_infer import _extract_kv_pairs

    kv_pairs = _extract_kv_pairs(past_key_values)
    compressed_layers = []
    total_uncompressed_bytes = 0
    total_compressed_bytes = 0
    zero_ratios = []

    for k, v in kv_pairs:
        batch, n_heads, seq_len, head_dim = k.shape
        total_uncompressed_bytes += k.nelement() * k.element_size()
        total_uncompressed_bytes += v.nelement() * v.element_size()

        layer_k_codes = []
        layer_v_codes = []
        layer_k_scales = []
        layer_v_scales = []

        for h in range(n_heads):
            for s in range(seq_len):
                k_vec = k[0, h, s, :]
                v_vec = v[0, h, s, :]

                k_codes, k_scale, k_zr, _ = ternary_quantise_vector(k_vec, threshold_factor)
                v_codes, v_scale, v_zr, _ = ternary_quantise_vector(v_vec, threshold_factor)

                layer_k_codes.append(k_codes)
                layer_v_codes.append(v_codes)
                layer_k_scales.append(k_scale)
                layer_v_scales.append(v_scale)
                zero_ratios.extend([k_zr, v_zr])

        all_k = torch.stack(layer_k_codes)
        all_v = torch.stack(layer_v_codes)
        packed_k, k_bytes = pack_ternary_2bit(all_k.flatten())
        packed_v, v_bytes = pack_ternary_2bit(all_v.flatten())

        scales_bytes = len(layer_k_scales) * 2 + len(layer_v_scales) * 2  # FP16 per scale
        total_compressed_bytes += k_bytes + v_bytes + scales_bytes

        compressed_layers.append({
            "k_codes": all_k,
            "v_codes": all_v,
            "k_scales": torch.tensor(layer_k_scales, dtype=torch.float16),
            "v_scales": torch.tensor(layer_v_scales, dtype=torch.float16),
        })

    return compressed_layers, {
        "uncompressed_bytes": total_uncompressed_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": total_uncompressed_bytes / total_compressed_bytes if total_compressed_bytes > 0 else float('inf'),
        "mean_zero_ratio": float(np.mean(zero_ratios)),
        "min_zero_ratio": float(np.min(zero_ratios)),
        "max_zero_ratio": float(np.max(zero_ratios)),
    }


def decompress_kv_cache(compressed_layers, original_past_key_values):
    """Reconstruct KV cache from ternary codes + scales."""
    _TOOLS = str(Path(__file__).resolve().parent.parent / "tools")
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    from tern_infer import _extract_kv_pairs
    kv_pairs = _extract_kv_pairs(original_past_key_values)
    new_past = []

    for layer_idx, (k, v) in enumerate(kv_pairs):
        batch, n_heads, seq_len, head_dim = k.shape
        cl = compressed_layers[layer_idx]
        new_k = torch.zeros_like(k)
        new_v = torch.zeros_like(v)
        idx = 0
        for h in range(n_heads):
            for s in range(seq_len):
                new_k[0, h, s, :] = ternary_dequantise_vector(
                    cl["k_codes"][idx], cl["k_scales"][idx].item()
                ).to(k.dtype)
                new_v[0, h, s, :] = ternary_dequantise_vector(
                    cl["v_codes"][idx], cl["v_scales"][idx].item()
                ).to(v.dtype)
                idx += 1
        new_past.append((new_k, new_v))

    from transformers.cache_utils import DynamicCache
    return DynamicCache(tuple(new_past))


def run_quality_test(model, tokenizer, prompt, n_tokens, threshold_factor, device="cpu"):
    """Generate with and without KV compression, measure argmax match."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    baseline_tokens = []
    compressed_tokens = []

    with torch.no_grad():
        # Baseline generation (no compression)
        past_kv = None
        gen_ids = input_ids.clone()
        for step in range(n_tokens):
            if past_kv is None:
                out = model(input_ids=gen_ids, use_cache=True)
            else:
                out = model(input_ids=gen_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
            baseline_tokens.append(next_tok.item())

        # Compressed generation (ternary KV round-trip each step)
        past_kv = None
        gen_ids = input_ids.clone()
        for step in range(n_tokens):
            if past_kv is None:
                out = model(input_ids=gen_ids, use_cache=True)
            else:
                out = model(input_ids=gen_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

            # Compress and decompress KV cache (closed-loop)
            compressed, metrics = compress_kv_cache(past_kv, threshold_factor)
            past_kv = decompress_kv_cache(compressed, past_kv)

            next_tok = out.logits[:, -1:].argmax(dim=-1)
            gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
            compressed_tokens.append(next_tok.item())

    matches = sum(1 for b, c in zip(baseline_tokens, compressed_tokens) if b == c)
    match_rate = matches / len(baseline_tokens)

    return {
        "prompt": prompt,
        "threshold": threshold_factor,
        "n_tokens": n_tokens,
        "argmax_matches": matches,
        "argmax_match_rate": match_rate,
        "compression_ratio": metrics["compression_ratio"],
        "mean_zero_ratio": metrics["mean_zero_ratio"],
        "baseline_text": tokenizer.decode(baseline_tokens),
        "compressed_text": tokenizer.decode(compressed_tokens),
    }


def main():
    print("BUILD 6 Track B3 — Native-Ternary KV Compression")
    print(f"Model: {MODEL_ID}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Quality gates: 93% floor, 99% bell mean")
    print()

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Architecture: {n_layers}L × {n_kv_heads}KV × {head_dim}d")
    print()

    all_results = []

    for t in THRESHOLDS:
        print(f"--- Threshold {t} ---")
        prompt_results = []
        for prompt in PROMPTS:
            r = run_quality_test(model, tokenizer, prompt, GENERATE_TOKENS, t, device)
            prompt_results.append(r)
            match_pct = r["argmax_match_rate"] * 100
            status = "PASS" if match_pct >= 93 else "FAIL"
            print(f"  [{status}] {match_pct:5.1f}% match, {r['compression_ratio']:.1f}× compression, "
                  f"{r['mean_zero_ratio']*100:.0f}% zeros — \"{prompt[:30]}...\"")

        match_rates = [r["argmax_match_rate"] for r in prompt_results]
        mean_match = float(np.mean(match_rates))
        min_match = float(np.min(match_rates))
        compression_ratios = [r["compression_ratio"] for r in prompt_results]
        mean_compression = float(np.mean(compression_ratios))
        zero_ratios = [r["mean_zero_ratio"] for r in prompt_results]
        mean_zero = float(np.mean(zero_ratios))

        floor_pass = min_match >= 0.93
        mean_pass = mean_match >= 0.99

        verdict = "PASS" if floor_pass and mean_pass else "FLOOR_FAIL" if not floor_pass else "MEAN_FAIL"

        sweep_point = {
            "threshold": t,
            "mean_match_rate": mean_match,
            "min_match_rate": min_match,
            "max_match_rate": float(np.max(match_rates)),
            "std_match_rate": float(np.std(match_rates)),
            "mean_compression_ratio": mean_compression,
            "mean_zero_ratio": mean_zero,
            "floor_pass": floor_pass,
            "mean_pass": mean_pass,
            "verdict": verdict,
            "per_prompt": prompt_results,
        }
        all_results.append(sweep_point)
        print(f"  → Mean: {mean_match*100:.1f}%, Min: {min_match*100:.1f}%, "
              f"Compression: {mean_compression:.1f}×, Zeros: {mean_zero*100:.0f}% — {verdict}")
        print()

    # Find best operating point (highest compression that passes both gates)
    passing = [r for r in all_results if r["floor_pass"] and r["mean_pass"]]
    if passing:
        best = max(passing, key=lambda r: r["mean_compression_ratio"])
        print(f"=== BEST OPERATING POINT: threshold={best['threshold']}, "
              f"{best['mean_compression_ratio']:.1f}× compression, "
              f"{best['mean_match_rate']*100:.1f}% mean match ===")
    else:
        best = None
        print("=== NO THRESHOLD PASSES BOTH GATES ===")
        closest = min(all_results, key=lambda r: abs(r["mean_match_rate"] - 0.99))
        print(f"  Closest: threshold={closest['threshold']}, "
              f"{closest['mean_match_rate']*100:.1f}% mean, {closest['min_match_rate']*100:.1f}% min")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    report = {
        "benchmark": "BUILD 6 Track B3 — Native-Ternary KV Compression",
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "chip": "Apple M4 Pro",
            "machine": platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {
            "id": MODEL_ID,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
        },
        "method": "Native ternary: {-1, 0, +1} per-vector symmetric threshold, 2-bit packed + FP16 scale",
        "quality_gates": {
            "floor": 0.93,
            "bell_mean": 0.99,
            "metric": "argmax_match_rate (next-token prediction match vs uncompressed baseline)",
        },
        "sweep_results": [{
            "threshold": r["threshold"],
            "mean_match_rate": r["mean_match_rate"],
            "min_match_rate": r["min_match_rate"],
            "mean_compression_ratio": r["mean_compression_ratio"],
            "mean_zero_ratio": r["mean_zero_ratio"],
            "verdict": r["verdict"],
        } for r in all_results],
        "best_operating_point": {
            "threshold": best["threshold"],
            "mean_match_rate": best["mean_match_rate"],
            "min_match_rate": best["min_match_rate"],
            "mean_compression_ratio": best["mean_compression_ratio"],
            "mean_zero_ratio": best["mean_zero_ratio"],
        } if best else None,
        "full_results": all_results,
        "peak_rss_mb": peak_rss_mb,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {OUTPUT_PATH}")

    md_path = OUTPUT_PATH.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# BUILD 6 Track B3 — Native-Ternary KV Compression Report\n\n")
        f.write(f"> **Hardware:** Apple M4 Pro · 64 GB · {platform.system()} {platform.release()}\n")
        f.write(f"> **Date:** {report['timestamp']}\n")
        f.write(f"> **Model:** {MODEL_ID}\n")
        f.write(f"> **Method:** Native ternary {{-1, 0, +1}} quantisation of KV cache entries\n\n")
        f.write("## Quality gates\n\n")
        f.write("- **93% floor**: no individual prompt below 93% argmax match\n")
        f.write("- **99% bell mean**: average across prompts ≥ 99% argmax match\n\n")
        f.write("## Threshold sweep\n\n")
        f.write("| Threshold | Mean match | Min match | Compression | Zero ratio | Verdict |\n")
        f.write("|:---------:|:----------:|:---------:|:-----------:|:----------:|:-------:|\n")
        for r in all_results:
            f.write(f"| {r['threshold']} | {r['mean_match_rate']*100:.1f}% | "
                    f"{r['min_match_rate']*100:.1f}% | {r['mean_compression_ratio']:.1f}× | "
                    f"{r['mean_zero_ratio']*100:.0f}% | {r['verdict']} |\n")
        f.write("\n")
        if best:
            f.write(f"## Best operating point: threshold = {best['threshold']}\n\n")
            f.write(f"- Compression: **{best['mean_compression_ratio']:.1f}×**\n")
            f.write(f"- Mean argmax match: **{best['mean_match_rate']*100:.1f}%**\n")
            f.write(f"- Min argmax match: **{best['min_match_rate']*100:.1f}%**\n")
            f.write(f"- Zero ratio: **{best['mean_zero_ratio']*100:.0f}%**\n")
        else:
            f.write("## No threshold passes both gates\n\n")
            f.write("Quality degrades below the 93% floor or 99% mean at all thresholds tested.\n")

    print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
