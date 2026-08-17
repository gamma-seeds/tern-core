"""Canonical WikiText-2 PPL loader for Gemma 4 *Unified* (.tern-model) artefacts.

``tools/tern_ppl_bench.py --tern-model-path`` loads a ternary variant through
``TernModelReader.load_packed_model``, whose module-tree walker raises on the
Gemma 4 Unified (``gemma4_unified``, ``Gemma4UnifiedForConditionalGeneration``)
manifest:

* The manifest stores text-tower tensors under **logical** Gemma names
  (``model.layers.*`` / ``model.embed_tokens.*`` / ``model.norm.*``), while
  transformers 5.10 nests the decoder under ``model.language_model.*``.
* The FP16-retained inline multimodal projectors
  (``model.embed_vision.*`` / ``model.embed_audio.*`` / ``model.vision_embedder.*``)
  have no matching submodule when the checkpoint is instantiated as a text
  ``AutoModelForCausalLM`` — they are irrelevant to a text-only WikiText-2 forward.

This module supplies the canonical text-only PPL path for Unified artefacts:

1. Materialise the FP32 skeleton via ``AutoModelForCausalLM``.
2. Overlay the reconstructed weights by **flat state-dict key**, streaming one
   layer at a time (bounded memory — no full second state-dict), translating
   logical names → the transformers-5.10 tree via the canonical
   ``GEMMA4_MULTIMODAL_TRANSFORMERS_5_5`` preset.
3. Skip manifest keys absent from the text model, while **hard-failing if any
   ternary / INT4 entry fails to place** (a missing decoder-tower entry would
   silently corrupt the measurement).
4. Compute PPL with the harness's canonical R7-A ``evaluate_ppl`` so the number
   is methodology-identical to the FP baseline produced by ``tern_ppl_bench``.

The text-tower name bridge and the strict-quant overlay gate are the two pieces
the stock loader lacks; everything downstream reuses ``tern_ppl_bench``.

Usage::

    python tools/gemma4_unified_ppl.py \
        --tern-model-path model.tern-model \
        --source-model-id google/gemma-4-12B \
        --device mps --seq-len 2048 --stride 2048 \
        --baseline-ppl 8.4638

Patents 10-12: automated binary-to-ternary conversion + faithful reconstruction.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import terncore  # noqa: E402
from terncore.tern_model import (  # noqa: E402
    TernModelReader,
    GEMMA4_MULTIMODAL_TRANSFORMERS_5_5 as TEXT_TOWER_MAP,
)
import tern_ppl_bench as B  # noqa: E402

# Manifest dtypes that MUST place — a decoder-tower entry that fails to overlay
# would silently corrupt the measurement, so the overlay gate aborts on these.
QUANT_DTYPES = {"ternary2", "ternary_g128", "int4_block32"}


def translate_name(name: str) -> str:
    """Bridge logical manifest names → the transformers 5.10 module tree.

    The Unified manifest stores text-tower tensors under logical Gemma names
    (``model.layers.*`` / ``model.embed_tokens.*`` / ``model.norm.*``);
    transformers 5.10 nests them under ``model.language_model.*``. First prefix
    match wins; unmatched names (the vision/audio embedder) pass through
    unchanged.
    """
    for src, dst in TEXT_TOWER_MAP.items():
        if name.startswith(src):
            return dst + name[len(src):]
    return name


def overlay(reader: TernModelReader, model: torch.nn.Module) -> dict:
    """Stream-overlay reconstructed weights onto ``model`` by flat key.

    Returns a report dict with placement counts. Caller must treat a non-empty
    ``skipped_quant`` (any ternary/INT4 entry that failed to place) or ``fail``
    (shape mismatch) as fatal.
    """
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    keyset = set(params.keys())

    placed: list[str] = []
    skipped: list[tuple[str, str]] = []
    fail: list[tuple] = []
    counts: dict[str, int] = {}

    for entry in reader.manifest["layers"]:
        raw_name = entry["name"]            # manifest key — reconstruct with this
        mapped = translate_name(raw_name)   # model-tree key — match/copy with this
        dtype = entry.get("dtype", "float16")
        counts[dtype] = counts.get(dtype, 0) + 1
        tensors = reader.reconstruct_layer(raw_name)
        for sub, tensor in tensors.items():
            if mapped in keyset:
                key = mapped
            elif f"{mapped}.{sub}" in keyset:
                key = f"{mapped}.{sub}"
            elif sub == "bias" and mapped.endswith(".weight") and \
                    mapped[:-7] + ".bias" in keyset:
                key = mapped[:-7] + ".bias"
            else:
                key = mapped if sub == "weight" else f"{mapped}.{sub}"

            if key in keyset:
                p = params[key]
                if tuple(p.shape) != tuple(tensor.shape):
                    fail.append((key, tuple(p.shape), tuple(tensor.shape)))
                    continue
                with torch.no_grad():
                    p.copy_(tensor.to(p.dtype))
                placed.append(key)
            else:
                skipped.append((key, dtype))
        del tensors

    skipped_quant = [(k, d) for (k, d) in skipped if d in QUANT_DTYPES]
    return {
        "placed": len(placed), "skipped": skipped, "fail": fail,
        "skipped_quant": skipped_quant, "counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WikiText-2 PPL for Gemma 4 Unified .tern-model artefacts"
    )
    ap.add_argument("--tern-model-path", required=True)
    ap.add_argument("--source-model-id", default="google/gemma-4-12B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--baseline-ppl", type=float, default=None)
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "wikitext2_ppl"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gemma4-ppl] loading fp32 skeleton {args.source_model_id} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.source_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    print(f"[gemma4-ppl]   skeleton loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    reader = TernModelReader(Path(args.tern_model_path))
    print("[gemma4-ppl] overlaying reconstructed weights (streaming) ...", flush=True)
    t1 = time.perf_counter()
    rep = overlay(reader, model)
    print(f"[gemma4-ppl]   overlay done in {time.perf_counter()-t1:.1f}s", flush=True)
    print(f"[gemma4-ppl]   manifest dtype counts: {rep['counts']}", flush=True)
    print(f"[gemma4-ppl]   placed={rep['placed']} skipped={len(rep['skipped'])} "
          f"fail={len(rep['fail'])}", flush=True)
    if rep["skipped"]:
        print(f"[gemma4-ppl]   skipped (text-irrelevant) first 8: "
              f"{rep['skipped'][:8]}", flush=True)
    if rep["fail"]:
        print(f"[gemma4-ppl]   FAIL (shape mismatch): {rep['fail'][:8]}", flush=True)
        sys.exit(2)
    if rep["skipped_quant"]:
        print(f"[gemma4-ppl]   ABORT — ternary/INT4 entries did not place: "
              f"{rep['skipped_quant'][:8]}", flush=True)
        sys.exit(3)

    model = model.to(args.device)
    model.eval()

    print("[gemma4-ppl] loading WikiText-2 test split ...", flush=True)
    test_text, hf_rev = B.load_wikitext2_test_text()
    bos = tok.bos_token_id
    tokens = B.prepare_tokens(test_text, tok, bos)
    print(f"[gemma4-ppl]   tokens: {tokens.shape[0]:,} "
          f"(bos_prepended={bos is not None})", flush=True)

    print(f"[gemma4-ppl] eval seq_len={args.seq_len} stride={args.stride} ...", flush=True)
    t2 = time.perf_counter()
    res = B.evaluate_ppl(model, tokens, args.seq_len, args.stride, args.device)
    print(f"[gemma4-ppl]   PPL = {res.ppl:.4f} (windows={res.windows_evaluated}, "
          f"scored={res.tokens_scored:,}, {time.perf_counter()-t2:.1f}s)", flush=True)

    headroom = None
    if args.baseline_ppl:
        headroom = (res.ppl - args.baseline_ppl) / args.baseline_ppl
        print(f"[gemma4-ppl]   baseline={args.baseline_ppl} "
              f"ppl_headroom={headroom:.4f} "
              f"band={B.classify_ppl_headroom_band(headroom)}", flush=True)

    out = {
        "schema_version": "wikitext2_ppl/1.0-unified-overlay",
        "run_id": B.utc_now_compact(),
        "timestamp_utc": B.utc_now_iso(),
        "tern_core_version": terncore.__version__,
        "tern_core_git_commit": B.git_commit_short(),
        "model": {"model_id": args.source_model_id, "variant": "ternary",
                  "source_path": args.tern_model_path,
                  "load_path": "streaming_overlay_strict_quant"},
        "manifest_dtype_counts": rep["counts"],
        "overlay": {"placed": rep["placed"], "skipped": len(rep["skipped"])},
        "tokeniser": {"bos_token_id": bos, "bos_prepended": bos is not None},
        "dataset": {"name": B.WIKITEXT_CONFIG, "split": B.WIKITEXT_SPLIT,
                    "huggingface_revision": hf_rev,
                    "total_tokens": int(tokens.shape[0]),
                    "tokens_discarded": res.tokens_discarded},
        "methodology": {"spec_version": B.SPEC_VERSION, "seq_len": args.seq_len,
                        "stride": args.stride},
        "hardware": {"device": args.device, "dtype_activation": "float32",
                     "dtype_loss": "float32"},
        "results": {"windows_evaluated": res.windows_evaluated,
                    "tokens_scored": res.tokens_scored,
                    "mean_loss": round(res.mean_loss, 6), "ppl": round(res.ppl, 4)},
        "comparison": {"baseline_ppl": args.baseline_ppl,
                       "ppl_headroom": round(headroom, 4) if headroom is not None else None,
                       "ppl_headroom_band": (B.classify_ppl_headroom_band(headroom)
                                             if headroom is not None else None)},
        "notes": args.notes,
    }
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"ppl_gemma4_unified_overlay_{out['run_id']}.json"
    outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[gemma4-ppl]   wrote {outpath}", flush=True)
    print(f"[gemma4-ppl]   ppl = {out['results']['ppl']}", flush=True)


if __name__ == "__main__":
    main()
