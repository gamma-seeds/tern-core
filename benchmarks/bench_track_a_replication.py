"""BUILD 6 Track A — Replication of the pioneering 5.3× KV cache compression run.

Reruns the original TurboQuant KV experiment **exactly as first designed**:
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Method: IncrementalTQCompressor, open-loop (compress as side effect)
- Config: b_mse=3, mixed_precision=True, head_dim=64
- Measurement: compression ratio from byte accounting, encode overhead, PPL baseline

Open-loop means: the compressor.append() records compressed KV state, but the
model's next forward pass uses the original uncompressed past_key_values.
The compression ratio is MEASURED from actual compressed output bytes.
The PPL is the MODEL BASELINE (post-load), not KV-compression quality.

This is an honest replication of the original measurement methodology.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))
from tern_infer import IncrementalTQCompressor, _extract_kv_pairs  # noqa: E402

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
B_MSE = 3
N_LAYERS = 22
N_HEADS = 4  # KV heads (GQA); 32 = attention heads
HEAD_DIM = 64
PROMPT = "The capital of France is"
GENERATE_TOKENS = 50
N_RUNS = 3
OUTPUT_PATH = Path(__file__).parent / "Benchmark_Report_A_Replication.json"


def measure_kv_bytes(past_key_values):
    total = 0
    kv_pairs = _extract_kv_pairs(past_key_values)
    for k, v in kv_pairs:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total


def _count_object_bytes(obj):
    total = 0
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        val = getattr(obj, attr)
        if callable(val):
            continue
        if hasattr(val, 'nelement') and hasattr(val, 'element_size'):
            total += val.nelement() * val.element_size()
        elif hasattr(val, 'shape') and hasattr(val, 'dtype'):
            import numpy as np
            total += val.size * val.itemsize if isinstance(val, np.ndarray) else 0
        elif hasattr(val, 'pq') or hasattr(val, 'norm') or hasattr(val, 'indices'):
            total += _count_object_bytes(val)
    return total


def measure_compressed_bytes(compressor):
    total = 0
    for layer_data in compressor.compressed:
        if layer_data is None:
            continue
        for head_data in layer_data:
            for k_c, v_c in head_data:
                total += _count_object_bytes(k_c)
                total += _count_object_bytes(v_c)
    return total


def run_generation_with_tq(model, tokenizer, prompt, n_tokens, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    compressor = IncrementalTQCompressor(
        n_layers=N_LAYERS, n_heads=N_HEADS, head_dim=HEAD_DIM,
        device=device, b_mse=B_MSE,
    )

    past_key_values = None
    generated_ids = input_ids.clone()
    encode_times = []
    uncompressed_bytes_final = 0
    compressed_bytes_final = 0

    with torch.no_grad():
        t_prefill_start = time.perf_counter()
        out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        t_prefill_model = time.perf_counter() - t_prefill_start

        t_encode_start = time.perf_counter()
        compressor.append(past_key_values, encode_from=0)
        t_prefill_encode = time.perf_counter() - t_encode_start

        for step in range(n_tokens):
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values

            t0 = time.perf_counter()
            compressor.append(past_key_values)
            encode_times.append(time.perf_counter() - t0)

        uncompressed_bytes_final = measure_kv_bytes(past_key_values)
        compressed_bytes_final = measure_compressed_bytes(compressor)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    compression_ratio = uncompressed_bytes_final / compressed_bytes_final if compressed_bytes_final > 0 else float('inf')
    mean_encode_ms = (sum(encode_times) / len(encode_times) * 1000) if encode_times else 0

    return {
        "generated_text": generated_text,
        "n_tokens_generated": n_tokens,
        "prefill_model_ms": t_prefill_model * 1000,
        "prefill_encode_ms": t_prefill_encode * 1000,
        "per_token_encode_ms_mean": mean_encode_ms,
        "per_token_encode_ms_all": [t * 1000 for t in encode_times],
        "uncompressed_bytes": uncompressed_bytes_final,
        "compressed_bytes": compressed_bytes_final,
        "compression_ratio": compression_ratio,
        "compressor_seq_len": compressor.seq_len,
    }


def compute_baseline_ppl(model, tokenizer, device="cpu"):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        text = (
            "The capital of France is Paris. "
            "Neural networks use matrix multiplication for inference. "
            "Ternary quantization maps weights to negative one, zero, and positive one. "
        ) * 200

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs.loss.item()


def main():
    print(f"BUILD 6 Track A — TurboQuant KV Replication")
    print(f"Model: {MODEL_ID}")
    print(f"Config: b_mse={B_MSE}, n_layers={N_LAYERS}, n_heads={N_HEADS}, head_dim={HEAD_DIM}")
    print(f"Tokens: {GENERATE_TOKENS}, Runs: {N_RUNS}")
    print()

    device = "cpu"

    print(f"Loading model from HuggingFace cache...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")

    actual_n_layers = model.config.num_hidden_layers
    actual_n_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else model.config.num_attention_heads
    actual_head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Model architecture: {actual_n_layers} layers, {actual_n_heads} KV heads, {actual_head_dim} head_dim")
    print(f"Original config:    {N_LAYERS} layers, {N_HEADS} KV heads, {HEAD_DIM} head_dim")
    if actual_n_layers != N_LAYERS or actual_head_dim != HEAD_DIM:
        print("WARNING: Architecture mismatch with original config!")
    print()

    print("Computing baseline perplexity (model-only, no KV compression)...")
    t0 = time.perf_counter()
    ppl_loss = compute_baseline_ppl(model, tokenizer, device)
    import math
    ppl = math.exp(ppl_loss)
    ppl_time = time.perf_counter() - t0
    print(f"Baseline PPL: {ppl:.4f} (loss={ppl_loss:.4f}, computed in {ppl_time:.1f}s)")
    print()

    all_runs = []
    for run_idx in range(N_RUNS):
        print(f"--- Run {run_idx + 1}/{N_RUNS} ---")
        result = run_generation_with_tq(model, tokenizer, PROMPT, GENERATE_TOKENS, device)
        print(f"  Compression: {result['compression_ratio']:.2f}×")
        print(f"  Prefill encode: {result['prefill_encode_ms']:.2f} ms")
        print(f"  Per-token encode: {result['per_token_encode_ms_mean']:.2f} ms")
        print(f"  Uncompressed: {result['uncompressed_bytes']:,} bytes")
        print(f"  Compressed:   {result['compressed_bytes']:,} bytes")
        all_runs.append(result)

    median_idx = N_RUNS // 2
    sorted_by_ratio = sorted(all_runs, key=lambda r: r['compression_ratio'])
    median_run = sorted_by_ratio[median_idx]

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    report = {
        "benchmark": "BUILD 6 Track A — TurboQuant KV Replication",
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "chip": platform.processor() or "Apple M4 Pro",
            "machine": platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {
            "id": MODEL_ID,
            "n_layers": actual_n_layers,
            "n_kv_heads": actual_n_heads,
            "n_attn_heads": model.config.num_attention_heads,
            "head_dim": actual_head_dim,
            "params": sum(p.numel() for p in model.parameters()),
        },
        "original_config": {
            "source": "benchmarks/originals/tq_bench_results.json",
            "b_mse": B_MSE,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "mixed_precision": True,
            "method": "IncrementalTQCompressor open-loop (compress as side effect)",
        },
        "replication_config": {
            "prompt": PROMPT,
            "generate_tokens": GENERATE_TOKENS,
            "n_runs": N_RUNS,
            "device": device,
            "dtype": "float32",
        },
        "original_result": {
            "kv_cache_compression_ratio": 5.3,
            "prefill_overhead_ms": 12.4,
            "per_token_encode_ms": 0.38,
            "peak_memory_uncompressed_mb": 412.0,
            "peak_memory_compressed_mb": 78.0,
            "perplexity_baseline_fp16": 7.82,
            "perplexity_with_tq": 8.14,
            "tag": "Measured (2026-03-30)",
        },
        "replication_result": {
            "compression_ratio": median_run["compression_ratio"],
            "prefill_encode_ms": median_run["prefill_encode_ms"],
            "per_token_encode_ms_mean": median_run["per_token_encode_ms_mean"],
            "uncompressed_bytes": median_run["uncompressed_bytes"],
            "compressed_bytes": median_run["compressed_bytes"],
            "baseline_ppl": ppl,
            "baseline_ppl_loss": ppl_loss,
            "peak_rss_mb": peak_rss_mb,
            "tag": "Measured",
        },
        "all_runs": [{
            "run": i + 1,
            "compression_ratio": r["compression_ratio"],
            "prefill_encode_ms": r["prefill_encode_ms"],
            "per_token_encode_ms_mean": r["per_token_encode_ms_mean"],
            "uncompressed_bytes": r["uncompressed_bytes"],
            "compressed_bytes": r["compressed_bytes"],
        } for i, r in enumerate(all_runs)],
        "verdict": None,
        "notes": [
            "Open-loop measurement: compressor.append() records compressed state as side effect; "
            "model uses uncompressed past_key_values for next forward pass.",
            "Compression ratio is MEASURED from byte accounting (uncompressed vs compressed output).",
            "PPL is model baseline, NOT KV-compression quality impact.",
            "R12 sweep (2026-05-18) attempted closed-loop round-trip and got NaN PPL across all b_mse values — "
            "the open-loop measurement is the only methodology that produces finite quality numbers.",
        ],
    }

    ratios = [r["compression_ratio"] for r in all_runs]
    original_ratio = 5.3
    median_ratio = median_run["compression_ratio"]
    delta_pct = abs(median_ratio - original_ratio) / original_ratio * 100

    if delta_pct < 5:
        report["verdict"] = f"REPLICATED — median {median_ratio:.2f}× vs original {original_ratio}× (delta {delta_pct:.1f}%)"
    elif delta_pct < 15:
        report["verdict"] = f"PARTIAL — median {median_ratio:.2f}× vs original {original_ratio}× (delta {delta_pct:.1f}%); investigate cause"
    else:
        report["verdict"] = f"DRIFT — median {median_ratio:.2f}× vs original {original_ratio}× (delta {delta_pct:.1f}%); root cause needed"

    print()
    print(f"=== VERDICT: {report['verdict']} ===")
    print(f"Median compression: {median_ratio:.2f}× (original: {original_ratio}×)")
    print(f"Baseline PPL: {ppl:.4f} (original: 7.82)")
    print()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {OUTPUT_PATH}")

    md_path = OUTPUT_PATH.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# BUILD 6 Track A — TurboQuant KV Replication Report\n\n")
        f.write(f"> **Verdict:** {report['verdict']}\n")
        f.write(f"> **Date:** {report['timestamp']}\n")
        f.write(f"> **Hardware:** {report['hardware']['chip']} · {report['hardware']['os']}\n\n")
        f.write("## Original vs Replication\n\n")
        f.write("| Metric | Original (2026-03-30) | Replication | Tag |\n")
        f.write("|--------|:---------------------:|:-----------:|:---:|\n")
        f.write(f"| KV compression ratio | 5.3× | {median_ratio:.2f}× | Measured |\n")
        f.write(f"| Prefill encode overhead | 12.4 ms | {median_run['prefill_encode_ms']:.1f} ms | Measured |\n")
        f.write(f"| Per-token encode | 0.38 ms | {median_run['per_token_encode_ms_mean']:.2f} ms | Measured |\n")
        f.write(f"| Baseline PPL | 7.82 | {ppl:.2f} | Measured |\n")
        f.write(f"| Peak RSS | 412 MB | {peak_rss_mb:.0f} MB | Measured |\n\n")
        f.write("## Config\n\n")
        f.write(f"- Model: `{MODEL_ID}`\n")
        f.write(f"- b_mse: {B_MSE}\n")
        f.write(f"- Method: IncrementalTQCompressor open-loop\n")
        f.write(f"- Prompt: \"{PROMPT}\"\n")
        f.write(f"- Generated tokens: {GENERATE_TOKENS}\n")
        f.write(f"- Runs: {N_RUNS} (median reported)\n")
        f.write(f"- Device: {device}, dtype: float32\n\n")
        f.write("## Run-by-run\n\n")
        f.write("| Run | Compression | Prefill encode | Per-token encode |\n")
        f.write("|:---:|:-----------:|:--------------:|:----------------:|\n")
        for i, r in enumerate(all_runs):
            f.write(f"| {i+1} | {r['compression_ratio']:.2f}× | {r['prefill_encode_ms']:.1f} ms | {r['per_token_encode_ms_mean']:.2f} ms |\n")
        f.write("\n## Notes\n\n")
        for note in report["notes"]:
            f.write(f"- {note}\n")
    print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
