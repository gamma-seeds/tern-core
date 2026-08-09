"""BUILD 6 Track B1 — FP16 + INT8 weight baselines.

Measures baseline inference performance for three models in two
precision modes:

  1. FP16 (float16 on MPS / CPU)
  2. INT8 (torch.ao dynamic quantisation, CPU-only)

Per-model outputs: model size on disk, peak RSS, prefill latency,
decode tok/s, generated text sample. Each measurement repeated n>=3
times; median reported.

Models:
  - unsloth/Llama-3.2-1B-Instruct
  - unsloth/Llama-3.2-3B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3

Copyright (c) 2025-2026 Synapticode. All rights reserved.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HOME = os.environ.get("HF_HOME", "/Volumes/Syn Archive/cache/huggingface")
os.environ["HF_HOME"] = HF_HOME

MODELS = [
    {
        "id": "unsloth/Llama-3.2-1B-Instruct",
        "short": "llama32-1b",
        "fp16_size_gb_expected": 2.1,
    },
    {
        "id": "unsloth/Llama-3.2-3B-Instruct",
        "short": "llama32-3b",
        "fp16_size_gb_expected": 6.0,
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "mistral-7b",
        "fp16_size_gb_expected": 14.0,
    },
]

PROMPTS = [
    "The capital of France is",
    "Explain ternary computing in one paragraph:",
    "Write a Python function that sorts a list:",
]
GENERATE_TOKENS = 64
N_RUNS = 3
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "Benchmark_Report_B1_FP16_INT8_Baselines.json"


def get_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def model_size_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total


def run_generation(model, tokenizer, prompt, n_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        t_prefill = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        t_prefill = time.perf_counter() - t_prefill

        generated_ids = input_ids.clone()
        t_decode_start = time.perf_counter()
        for _ in range(n_tokens):
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
        t_decode = time.perf_counter() - t_decode_start

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    tok_s = n_tokens / t_decode if t_decode > 0 else 0

    return {
        "prompt_len": prompt_len,
        "n_tokens": n_tokens,
        "prefill_ms": round(t_prefill * 1000, 2),
        "decode_ms": round(t_decode * 1000, 2),
        "tok_s": round(tok_s, 2),
        "text_sample": text[:200],
    }


def benchmark_fp16(model_cfg):
    model_id = model_cfg["id"]
    short = model_cfg["short"]
    print(f"\n{'='*60}")
    print(f"FP16 baseline: {model_id}")
    print(f"{'='*60}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device
    )
    model.eval()

    size_bytes = model_size_bytes(model)
    size_mb = size_bytes / (1024 * 1024)
    peak_rss = get_peak_rss_mb()

    print(f"  Model size: {size_mb:.1f} MB ({size_bytes:,} bytes)")
    print(f"  Peak RSS: {peak_rss:.0f} MB")

    all_runs = []
    for prompt in PROMPTS:
        runs = []
        for r in range(N_RUNS):
            result = run_generation(model, tokenizer, prompt, GENERATE_TOKENS, device)
            runs.append(result)
            print(f"  [{short}] FP16 run {r+1}/{N_RUNS} prompt='{prompt[:30]}...' "
                  f"tok/s={result['tok_s']:.1f}")
        median_toks = statistics.median([r["tok_s"] for r in runs])
        all_runs.append({
            "prompt": prompt,
            "runs": runs,
            "median_tok_s": round(median_toks, 2),
        })

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_id": model_id,
        "short": short,
        "precision": "FP16",
        "device": device,
        "model_size_bytes": size_bytes,
        "model_size_mb": round(size_mb, 1),
        "peak_rss_mb": round(peak_rss, 0),
        "generate_tokens": GENERATE_TOKENS,
        "n_runs": N_RUNS,
        "prompts": all_runs,
        "overall_median_tok_s": round(
            statistics.median([p["median_tok_s"] for p in all_runs]), 2
        ),
    }


def benchmark_int8(model_cfg):
    model_id = model_cfg["id"]
    short = model_cfg["short"]
    print(f"\n{'='*60}")
    print(f"INT8 baseline: {model_id}")
    print(f"{'='*60}")

    torch.backends.quantized.engine = "qnnpack"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32
    )
    model.eval()

    fp32_size = model_size_bytes(model)
    print(f"  FP32 size before quantisation: {fp32_size / (1024**2):.1f} MB")

    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    int8_size = model_size_bytes(model)
    int8_mb = int8_size / (1024 * 1024)
    peak_rss = get_peak_rss_mb()
    compression = fp32_size / int8_size if int8_size > 0 else 0

    print(f"  INT8 model size: {int8_mb:.1f} MB")
    print(f"  Compression vs FP32: {compression:.2f}×")
    print(f"  Peak RSS: {peak_rss:.0f} MB")

    device = "cpu"

    all_runs = []
    for prompt in PROMPTS:
        runs = []
        for r in range(N_RUNS):
            result = run_generation(model, tokenizer, prompt, GENERATE_TOKENS, device)
            runs.append(result)
            print(f"  [{short}] INT8 run {r+1}/{N_RUNS} prompt='{prompt[:30]}...' "
                  f"tok/s={result['tok_s']:.1f}")
        median_toks = statistics.median([r["tok_s"] for r in runs])
        all_runs.append({
            "prompt": prompt,
            "runs": runs,
            "median_tok_s": round(median_toks, 2),
        })

    del model
    gc.collect()

    return {
        "model_id": model_id,
        "short": short,
        "precision": "INT8_dynamic",
        "device": device,
        "fp32_size_bytes": fp32_size,
        "model_size_bytes": int8_size,
        "model_size_mb": round(int8_mb, 1),
        "compression_vs_fp32": round(compression, 2),
        "peak_rss_mb": round(peak_rss, 0),
        "generate_tokens": GENERATE_TOKENS,
        "n_runs": N_RUNS,
        "prompts": all_runs,
        "overall_median_tok_s": round(
            statistics.median([p["median_tok_s"] for p in all_runs]), 2
        ),
    }


def main():
    print("BUILD 6 Track B1 — FP16 + INT8 Weight Baselines")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Platform: {platform.machine()} / {platform.system()} / Python {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"HF_HOME: {HF_HOME}")

    results = {
        "benchmark": "BUILD6_Track_B1_FP16_INT8_Baselines",
        "started": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps": torch.backends.mps.is_available(),
            "processor": platform.processor(),
        },
        "config": {
            "generate_tokens": GENERATE_TOKENS,
            "n_runs": N_RUNS,
            "prompts": PROMPTS,
        },
        "models": [],
    }

    for model_cfg in MODELS:
        model_results = {"model_id": model_cfg["id"], "short": model_cfg["short"]}

        try:
            model_results["fp16"] = benchmark_fp16(model_cfg)
        except Exception as e:
            print(f"  FP16 FAILED for {model_cfg['id']}: {e}")
            model_results["fp16"] = {"error": str(e)}

        try:
            model_results["int8"] = benchmark_int8(model_cfg)
        except Exception as e:
            print(f"  INT8 FAILED for {model_cfg['id']}: {e}")
            model_results["int8"] = {"error": str(e)}

        results["models"].append(model_results)

        results["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [saved intermediate → {OUTPUT_PATH.name}]")

    results["completed"] = datetime.now(timezone.utc).isoformat()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m in results["models"]:
        print(f"\n{m['model_id']}:")
        for prec in ["fp16", "int8"]:
            d = m.get(prec, {})
            if "error" in d:
                print(f"  {prec.upper()}: ERROR — {d['error']}")
            elif "overall_median_tok_s" in d:
                print(f"  {prec.upper()}: {d['model_size_mb']} MB, "
                      f"{d['overall_median_tok_s']} tok/s (median)")
            else:
                print(f"  {prec.upper()}: no data")

    print(f"\nResults → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
