"""
Load a .tern-model file and run inference or inspect metadata.

Companion to tools/tern_infer.py (which loads from HuggingFace and converts
on-the-fly). This tool loads pre-serialised .tern-model files directly.

Usage:
    python tools/tern_loader.py model.tern-model --info
    python tools/tern_loader.py model.tern-model --verify
    python tools/tern_loader.py model.tern-model --prompt "The future of"
    python tools/tern_loader.py model.tern-model --prompt "Once upon" --max-tokens 50

Patent 6: Model format specification.
Patent 8: Serialisation and integrity verification.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure tern-core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.tern_model import TernModelReader


def cmd_info(reader: TernModelReader) -> None:
    """Print header and manifest summary."""
    h = reader.header
    print(f"  Magic:        {h['magic'].decode()}")
    print(f"  Version:      {h['version']}")
    print(f"  Header size:  {h['header_size']} bytes")
    print(f"  Layers:       {h['num_layers']} "
          f"({h['num_ternary']} ternary, {h['num_protected']} protected)")
    print(f"  Weights:      offset={h['weights_offset']}, "
          f"size={h['weights_size']:,} bytes")

    meta = reader.manifest.get("model_metadata", {})
    print(f"\n  Source:        {meta.get('source', 'unknown')}")
    print(f"  Created:       {meta.get('created_at', 'unknown')}")
    print(f"  Notes:         {meta.get('notes', '')}")

    print(f"\n  Layer details:")
    for entry in reader.manifest["layers"]:
        dtype_tag = entry["dtype"]
        shape_str = "x".join(str(s) for s in entry["shape"])
        extras = ""
        if dtype_tag == "ternary2":
            extras = (f"  sparsity={entry.get('sparsity', 0):.1%}  "
                      f"alpha={entry.get('alpha', 0):.6f}")
        print(f"    {entry['name']:50s}  {dtype_tag:10s}  {shape_str:>12s}{extras}")


def cmd_verify(reader: TernModelReader) -> None:
    """Run CRC32 integrity check."""
    t0 = time.perf_counter()
    ok = reader.verify()
    dt = time.perf_counter() - t0
    if ok:
        print(f"  Integrity check PASSED ({dt*1000:.1f}ms)")
    else:
        print(f"  Integrity check FAILED ({dt*1000:.1f}ms)")
        sys.exit(1)


def cmd_infer(
    reader: TernModelReader,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> None:
    """Load .tern-model into a HuggingFace model and generate text."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  ERROR: pip install transformers sentencepiece accelerate")
        sys.exit(1)

    # Load tokenizer and empty model
    print(f"  Loading model architecture from {model_id}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"  Architecture loaded in {load_time:.1f}s")

    # Reconstruct state_dict from .tern-model
    print(f"  Loading weights from .tern-model...")
    t0 = time.perf_counter()
    state_dict = reader.reconstruct_all()
    reconstruct_time = time.perf_counter() - t0
    print(f"  Reconstructed {len(state_dict)} tensors in {reconstruct_time:.1f}s")

    # Load into model (strict=False since we only store Linear layers)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict)} matched, "
          f"{len(missing)} missing (non-Linear), {len(unexpected)} unexpected")

    # Generate
    print(f"\n  Prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    gen_time = time.perf_counter() - t0
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tok_s = n_tokens / gen_time if gen_time > 0 else 0

    print(f"  Output ({n_tokens} tokens, {tok_s:.1f} tok/s):")
    print(f"  {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect .tern-model files"
    )
    parser.add_argument("model_path", help="Path to .tern-model file")
    parser.add_argument("--info", action="store_true",
                        help="Show header and manifest summary")
    parser.add_argument("--verify", action="store_true",
                        help="Run CRC32 integrity check")
    parser.add_argument("--prompt", type=str,
                        help="Run inference with this prompt")
    parser.add_argument("--max-tokens", type=int, default=30,
                        help="Max tokens to generate (default: 30)")
    parser.add_argument("--model-id", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID for architecture")
    args = parser.parse_args()

    path = Path(args.model_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Loading {path.name}...")
    t0 = time.perf_counter()
    reader = TernModelReader(str(path))
    dt = time.perf_counter() - t0
    print(f"  Header + manifest parsed in {dt*1000:.1f}ms")

    if args.info:
        print("\n[Info]")
        cmd_info(reader)

    if args.verify:
        print("\n[Verify]")
        cmd_verify(reader)

    if args.prompt:
        print("\n[Inference]")
        cmd_infer(reader, args.model_id, args.prompt, args.max_tokens)

    if not (args.info or args.verify or args.prompt):
        print("\nNo action specified. Use --info, --verify, or --prompt.")
        parser.print_help()


if __name__ == "__main__":
    main()
