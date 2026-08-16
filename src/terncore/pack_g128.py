"""
Model-level per-group symmetric (``ternary_g128``) pack.

Ingests the genuinely-ternary FP16 weights an MLX-2bit-derived
``*-unpacked`` repo publishes (e.g. PrismML ``Ternary-Bonsai-*-unpacked``)
and packs them into tern-core's lossless per-group (128) symmetric
``ternary_g128`` format — the recommended path for MLX-2bit ternary
imports.

This is the **lossless** ingest: ternary-eligible weights route through
:func:`terncore.mlx_ingest.pack_group_symmetric` (per-group scale =
``max|w|``, ``trit = round(w/scale)``), preserving the per-group scale
granularity. It is distinct from :func:`terncore.convert.full_convert`,
whose threshold per-layer-α ``pack_ternary`` path collapses the
inter-group scale spread and is lossy on per-group-authored weights.

Per-layer hard equivalence gate: :func:`pack_group_symmetric` aborts with
the offending tensor name + group index on any group whose FP16
reconstruction drifts beyond two ULP at the group's scale. A single miss
aborts the whole pack — no lossy approximation is written.

The architecture adapter (e.g. ``qwen3``) decides which weights are
ternary-eligible (2-D transformer-block projections) versus FP16-retained
(embeddings, norms, QK-Norm, LM head).

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from terncore.mlx_ingest import (
    GROUP_SIZE_DEFAULT,
    IngestEquivalenceError,
    pack_group_symmetric,
)


def _printer(verbose: bool):
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)
    return _log


def _resolve_model_dir(model_id: str, log) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    log(f"  Resolving {model_id} from HuggingFace Hub …")
    local = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json",
                        "config.json"],
    )
    return Path(local)


def _discover_weights(model_dir: Path, log) -> dict:
    import json as _json
    from safetensors import safe_open

    weight_to_file: dict = {}
    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index_path.exists():
        wmap = _json.loads(index_path.read_text())["weight_map"]
        for wname, shard in wmap.items():
            weight_to_file[wname] = model_dir / shard
        log(f"  Sharded: {len(set(wmap.values()))} shards, "
            f"{len(weight_to_file)} weights")
    elif single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_to_file[k] = single
        log(f"  Single shard: {len(weight_to_file)} weights")
    else:
        for st in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(st), framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_to_file[k] = st
        log(f"  {len(weight_to_file)} weights across loose shards")
    if not weight_to_file:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")
    return weight_to_file


def pack_g128_model(
    model_id: str,
    adapter_name: str,
    output_path: str,
    *,
    group_size: int = GROUP_SIZE_DEFAULT,
    name: str = "model",
    verbose: bool = True,
) -> dict:
    """Pack an unpacked-FP16 ternary model into a ``ternary_g128`` .tern-model.

    Args:
        model_id:     HuggingFace model ID or local directory of the
                      ``*-unpacked`` FP16 source.
        adapter_name: Architecture adapter (e.g. ``"qwen3"``) — decides
                      ternary-eligible vs FP16-retained weights.
        output_path:  Path for the ``.tern-model`` output.
        group_size:   Per-group size along the input axis (default 128).
        name:         Logical model name recorded in the manifest.
        verbose:      Print progress.

    Returns:
        A report dict (layer census, compression, per-layer gate errors).

    Raises:
        IngestEquivalenceError: any group fails the per-group equivalence
            gate (the source is not cleanly ternary in that group).
    """
    import torch
    from safetensors import safe_open
    from terncore.adapters import get_adapter
    from terncore.tern_model import TernModelWriter

    log = _printer(verbose)
    t0 = time.perf_counter()
    adapter = get_adapter(adapter_name)

    log("=" * 68)
    log(f"  ternary_g128 pack — {adapter.info().name} adapter")
    log("=" * 68)
    log(f"  Model:  {model_id}")
    log(f"  Output: {output_path}")

    model_dir = _resolve_model_dir(model_id, log)

    # Validate adapter routing against the HF config.
    cfg = json.loads((model_dir / "config.json").read_text())
    hf_arch = (cfg.get("architectures") or [cfg.get("model_type", "")])[0]
    adapter.validate_architecture(hf_arch)
    log(f"  Arch '{hf_arch}' validated against adapter '{adapter.info().name}'")

    weight_to_file = _discover_weights(model_dir, log)
    file_to_keys: dict = {}
    for wname, fpath in weight_to_file.items():
        file_to_keys.setdefault(fpath, []).append(wname)

    # Read shapes + classify.
    weight_shapes: dict = {}
    for fpath, keys in file_to_keys.items():
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for key in keys:
                parent_shape = list(f.get_slice(key).get_shape())
                if adapter.expand_stacked(key, parent_shape) is not None:
                    raise RuntimeError(
                        f"Stacked/MoE tensor '{key}' is out of scope for the "
                        f"dense g128 pack; use a dense adapter."
                    )
                weight_shapes[key] = parent_shape

    classifications = adapter.classify_all(weight_shapes)
    eligible = {n for n, c in classifications.items()
                if c.category == "ternary_eligible"}
    log(f"  Classified {len(weight_shapes)} weights: "
        f"{len(eligible)} ternary-eligible, "
        f"{len(weight_shapes) - len(eligible)} FP16-retain")

    writer = TernModelWriter({
        "source": model_id,
        "adapter": adapter_name,
        "pack": "ternary_g128",
        "group_size": group_size,
        "pipeline": "pack_g128_model / per-group lossless ingest",
    })

    stats = {"g128_layers": 0, "g128_params": 0,
             "fp16_layers": 0, "fp16_params": 0}
    per_layer_gate: list = []
    global_max_err = 0.0
    total = len(weight_shapes)
    done = 0

    for fpath, keys in file_to_keys.items():
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for wname in sorted(keys):
                canonical = adapter.normalize_name(wname)
                tensor = f.get_tensor(wname)
                num_params = tensor.numel()

                if wname in eligible:
                    w_np = tensor.to(torch.float32).numpy()
                    pg = pack_group_symmetric(w_np, name=canonical,
                                              group_size=group_size)
                    writer.add_ternary_g128_layer(
                        name=canonical,
                        packed_weights=pg.packed_weights,
                        scales=pg.scales,
                        shape=pg.shape,
                        scale_shape=pg.scale_shape,
                        group_size=pg.group_size,
                        quant_error=pg.max_abs_error,
                    )
                    global_max_err = max(global_max_err, pg.max_abs_error)
                    per_layer_gate.append({
                        "name": canonical, "shape": pg.shape,
                        "scale_shape": pg.scale_shape,
                        "max_abs_error": pg.max_abs_error,
                    })
                    stats["g128_layers"] += 1
                    stats["g128_params"] += num_params
                    del w_np
                else:
                    writer.add_layer(name=canonical,
                                     weights=tensor.float(), dtype="float16")
                    stats["fp16_layers"] += 1
                    stats["fp16_params"] += num_params

                del tensor
                done += 1
                if verbose and (done % 50 == 0 or done == total):
                    log(f"    [{done}/{total}] packed "
                        f"(g128={stats['g128_layers']}, "
                        f"fp16={stats['fp16_layers']})")
        gc.collect()

    log(f"  Equivalence gate GREEN across {stats['g128_layers']} ternary "
        f"layers (global max_abs_error={global_max_err:.3e})")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wstats = writer.write(out_file)
    size = wstats["file_size"]
    total_params = stats["g128_params"] + stats["fp16_params"]
    compression = (total_params * 2 / size) if size else 0.0
    elapsed = time.perf_counter() - t0

    log(f"  Wrote {out_file} — {size/1e6:.1f} MB ({compression:.2f}× vs FP16)")

    return {
        "status": "PACKED_GATE_GREEN",
        "model_id": model_id,
        "adapter": adapter_name,
        "pack_format": "ternary_g128",
        "group_size": group_size,
        "output_path": str(out_file),
        "file_size_bytes": size,
        "compression_vs_fp16": round(compression, 2),
        "total_params": total_params,
        "global_max_abs_error": global_max_err,
        **stats,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_layer_gate": per_layer_gate,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pack an unpacked-FP16 ternary model into ternary_g128.")
    ap.add_argument("model", help="HuggingFace model ID or local directory")
    ap.add_argument("-o", "--output", required=True, help=".tern-model path")
    ap.add_argument("--adapter", default="qwen3",
                    help="architecture adapter (default: qwen3)")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE_DEFAULT)
    ap.add_argument("--name", default="model")
    ap.add_argument("--report", default=None, help="optional JSON report path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = pack_g128_model(
            args.model, args.adapter, args.output,
            group_size=args.group_size, name=args.name,
            verbose=not args.quiet,
        )
    except IngestEquivalenceError as e:
        print(f"EQUIVALENCE GATE MISS — refusing to pack:\n  {e}",
              file=sys.stderr, flush=True)
        return 2

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
