"""
Interactive ternary inference demo.

Loads TinyLlama-1.1B, applies mixed-precision ternary conversion
(v_proj_late3: 3 ternary layers in blocks 19-21), and generates text.

Usage:
    python tools/tern_infer.py --prompt "The future of computing lies in"
    python tools/tern_infer.py --interactive
    python tools/tern_infer.py --prompt "Once upon a time" --max-tokens 100

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Ensure tern-core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.mixed_precision import MixedPrecisionConverter

# v_proj_late3 config from Day 3 (3 ternary layers, estimated +4.1% PPL gap)
V_PROJ_LATE3_LAYERS = [
    "model.layers.19.self_attn.v_proj",
    "model.layers.20.self_attn.v_proj",
    "model.layers.21.self_attn.v_proj",
]

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_TOKENS = 50


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load model and apply v_proj_late3 mixed-precision ternary conversion."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Build protection list: protect ALL layers EXCEPT v_proj_late3
    all_linears = [name for name, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear)]
    protection_list = [name for name in all_linears
                       if name not in V_PROJ_LATE3_LAYERS]

    print(f"Converting to mixed-precision ternary (v_proj_late3)...")
    t0 = time.perf_counter()
    converter = MixedPrecisionConverter(
        threshold=0.7,
        protection_list=protection_list,
    )
    report = converter.convert(model)
    conv_time = time.perf_counter() - t0
    print(f"  Converted {report.converted_layers} layers in {conv_time:.1f}s")
    print(f"  Protected: {report.skipped_layers}, Compression: {report.compression_ratio:.2f}x")

    model.eval()
    return model, tokenizer


def generate_streaming(
    model, tokenizer, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, float, int]:
    """Generate text token by token with streaming output.

    Returns (full_text, tokens_per_second, num_tokens).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()

    print(f"\n{prompt}", end="", flush=True)

    t_start = time.perf_counter()
    tokens_generated = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            tokens_generated += 1

            # Decode and print just the new token
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(new_token, end="", flush=True)

    t_elapsed = time.perf_counter() - t_start
    tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0

    print()  # newline
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text, tps, tokens_generated


def interactive_mode(model, tokenizer, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    """Interactive prompt loop."""
    print()
    print("=" * 60)
    print("  Ternary Inference Demo (v_proj_late3, 3 ternary layers)")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        _, tps, n_tokens = generate_streaming(model, tokenizer, prompt, max_tokens)
        print(f"  [{n_tokens} tokens, {tps:.1f} tok/s]")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ternary inference demo")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_tokens)
    elif args.prompt:
        full_text, tps, n_tokens = generate_streaming(
            model, tokenizer, args.prompt, args.max_tokens,
        )
        print(f"\n  [{n_tokens} tokens, {tps:.1f} tok/s]")
    else:
        # Default demo prompt
        full_text, tps, n_tokens = generate_streaming(
            model, tokenizer,
            "The future of computing lies in",
            args.max_tokens,
        )
        print(f"\n  [{n_tokens} tokens, {tps:.1f} tok/s]")


if __name__ == "__main__":
    main()
