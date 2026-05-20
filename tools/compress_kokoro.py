"""
Compress Kokoro 82M to the canonical .tern-model artefact via the
KokoroAdapter routing policy + TernModelWriter pipeline.

First downstream consumer integration of integration³ Provider³ Protocol
(per surgeon's dispatch 2026-05-19; Phase 0 brief
``2026-05-19_kokoro_82m_integration3_attachment_phase0.md``).

Pipeline:
  1. Load ``kokoro-v1_0.pth`` (dict-of-dicts: bert / bert_encoder /
     predictor / decoder / text_encoder).
  2. Build composite keys ``"<section>/<tensor_name>"`` so the
     KokoroAdapter's vocoder + acoustic-stack pattern matchers fire.
  3. Per-tensor route via ``KokoroAdapter.classify_weight`` — ternary
     for 2-D / 3-D weight tensors (incl. ResBlock conv kernels and
     AdaIN modulator FCs); INT4 for 2-D F0_proj / N_proj; FP16 for
     embeddings + norms + weight_g + 1-D.
  4. Pack via TernModelWriter add_ternary_layer / add_int4_layer /
     _add_fp16_layer helpers. Skips per-weight sparsity bitmap on
     ternary entries — bitmap is reader-side optional and would add
     ~10 MB to a 82M-param compressed artefact (1 bit / weight). At
     execution time the block-level bitmap can be regenerated via
     ``TernModelWriter.generate_sparsity_bitmap`` from packed bytes.
  5. Write ``kokoro_82m_ternary_v0.6.0.tern-model`` to the canonical
     model-library directory.
  6. Report size + F-8 ≤22 MB ceiling pass/fail.

Phase 0 OQ-6 ratification: ≤22 MB demo-shipped artefact (main model
+ 6 demo voices @ FP16). Voices stay as separate .pt files at the
filesystem (loaded per-call by KokoroProvider3) — not embedded in
the .tern-model artefact.

Usage:
    /path/to/venv/bin/python tools/compress_kokoro.py \\
        [--source <path>] [--output <path>] [--threshold 0.7]

Defaults align with the canonical model-library convention:
    source: ~/synapticode/model-library/Kokoro-82M/kokoro-v1_0.pth
    output: ~/synapticode/model-library/Kokoro-82M/
            kokoro_82m_ternary_v0.6.0.tern-model

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Route through this worktree's src/ rather than the venv editable
# install pointing at the R12 sprint working tree.
_HERE = Path(__file__).resolve().parent
_WORKTREE_SRC = _HERE.parent / "src"
if _WORKTREE_SRC.exists() and str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))
for _mod in list(sys.modules):
    if _mod == "terncore" or _mod.startswith("terncore."):
        del sys.modules[_mod]

import torch  # noqa: E402

from terncore.adapters import get_adapter  # noqa: E402
from terncore.tern_model import TernModelWriter  # noqa: E402

DEFAULT_SOURCE = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/kokoro-v1_0.pth"
)
DEFAULT_OUTPUT = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/"
    "kokoro_82m_ternary_v0.6.0.tern-model"
)
DEMO_VOICES_DIR = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/voices"
)
DEMO_VOICES = (
    "af_heart", "am_adam", "bf_alice",
    "bm_daniel", "jf_alpha", "zf_xiaobei",
)
F8_CEILING_BYTES = 22 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress Kokoro 82M to .tern-model artefact."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--voices-dir", default=DEMO_VOICES_DIR)
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    voices_dir = Path(args.voices_dir)
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        return 2

    print(f"Loading {src}...")
    t0 = time.time()
    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    print(f"  Loaded in {time.time() - t0:.1f}s; "
          f"sections: {list(sd.keys())}")

    adapter = get_adapter("kokoro")
    info = adapter.info()
    print(f"Adapter: {info.name}; architectures={info.architectures}")
    adapter.validate_architecture("KokoroForTTS")

    writer = TernModelWriter({
        "source": "hexgrad/Kokoro-82M",
        "source_format": "kokoro-v1_0.pth",
        "adapter": "kokoro",
        "threshold": args.threshold,
        "pipeline": "kokoro-adapter-convert-v0.6.0",
        "demo_voices": list(DEMO_VOICES),
        "sample_rate_hz": 24000,
        "n_token": 178,
        "license": "Apache-2.0",
    })

    stats = {
        "ternary": 0, "int4": 0, "fp16": 0,
        "ternary_params": 0, "int4_params": 0, "fp16_params": 0,
    }

    n_total = sum(
        1 for _section, sub in sd.items()
        for _k, v in sub.items()
        if isinstance(v, torch.Tensor)
    )
    print(f"\nProcessing {n_total} tensors...")
    t1 = time.time()

    for section, sub_sd in sd.items():
        for key, weight in sub_sd.items():
            if not isinstance(weight, torch.Tensor):
                continue
            composite_name = f"{section}/{key}"
            shape = list(weight.shape)
            classification = adapter.classify_weight(composite_name, shape)
            cat = classification.category

            w_fp32 = weight.detach().to(torch.float32)

            if cat == "ternary_eligible":
                # Skip per-weight bitmap (1 bit/weight = ~10 MB at 82M
                # params); reader-side optional + block-level bitmap can
                # be regenerated post-hoc.
                packed, alpha, _bitmap, sparsity = TernModelWriter.pack_ternary(
                    w_fp32, threshold=args.threshold
                )
                writer.add_ternary_layer(
                    name=composite_name,
                    packed_weights=packed,
                    alpha=alpha,
                    shape=list(w_fp32.shape),
                    sparsity_bitmap=None,
                    threshold=args.threshold,
                    sparsity=sparsity,
                )
                stats["ternary"] += 1
                stats["ternary_params"] += w_fp32.numel()
            elif cat == "int4_eligible":
                writer.add_layer(
                    name=composite_name,
                    weights=w_fp32,
                    dtype="int4_block32",
                )
                stats["int4"] += 1
                stats["int4_params"] += w_fp32.numel()
            else:  # "fp16_retain"
                writer.add_layer(
                    name=composite_name,
                    weights=w_fp32,
                    dtype="float16",
                )
                stats["fp16"] += 1
                stats["fp16_params"] += w_fp32.numel()

    print(f"  Quantised in {time.time() - t1:.1f}s")
    print(f"  Ternary: {stats['ternary']:>4d} layers, "
          f"{stats['ternary_params']:>11,d} params")
    print(f"  INT4:    {stats['int4']:>4d} layers, "
          f"{stats['int4_params']:>11,d} params")
    print(f"  FP16:    {stats['fp16']:>4d} layers, "
          f"{stats['fp16_params']:>11,d} params")

    print(f"\nWriting {out}...")
    out.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.time()
    write_stats = writer.write(str(out))
    print(f"  File size: {write_stats['file_size'] / 1024 / 1024:.2f} MB")
    print(f"  Wrote in {time.time() - t2:.1f}s")

    model_bytes = out.stat().st_size
    print(f"\nOn-disk size: {model_bytes / 1024 / 1024:.2f} MB "
          f"({model_bytes:,} bytes)")

    # Demo-subset voice footprint at FP16 (half of FP32 on-disk size).
    voice_bytes = 0
    for v in DEMO_VOICES:
        vf = voices_dir / f"{v}.pt"
        if vf.exists():
            voice_bytes += vf.stat().st_size // 2

    print(f"6 demo voices @ FP16 (est): "
          f"{voice_bytes / 1024 / 1024:.2f} MB")
    total = model_bytes + voice_bytes
    print(f"Total demo artefact: {total / 1024 / 1024:.2f} MB")
    print(f"F-8 ≤22 MB ceiling: "
          f"{'PASS' if total <= F8_CEILING_BYTES else 'FAIL'}")
    return 0 if total <= F8_CEILING_BYTES else 1


if __name__ == "__main__":
    sys.exit(main())
