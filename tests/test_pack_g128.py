"""
Integration test for the model-level ternary_g128 pack
(``terncore.pack_g128.pack_g128_model``).

Synthesises a tiny dense Qwen3 model (config.json + safetensors) whose
2-D transformer-block projections are genuinely per-group ternary
(trits × per-128-group scale) and whose embeddings / norms / QK-Norm /
LM head are FP16. Packs it end-to-end and asserts: the adapter routes
the expected layer census, the per-group equivalence gate passes
bit-exact, and the written .tern-model reconstructs to the source
weights within FP16 tolerance.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from terncore.mlx_ingest import GROUP_SIZE_DEFAULT, IngestEquivalenceError
from terncore.pack_g128 import pack_g128_model
from terncore.tern_model import TernModelReader

GS = GROUP_SIZE_DEFAULT

# Tiny dense-Qwen3 geometry; every ternary-eligible input dim is a
# multiple of the group size (128): hidden = 128 (1 group), intermediate
# = 256 (2 groups), o_proj in = num_heads*head_dim = 128.
N_LAYERS = 2
HIDDEN = 128
INTER = 256
N_HEADS = 2
N_KV = 1
HEAD_DIM = 64
VOCAB = 32


def _ternary_weight(out_f, in_f, seed):
    """A genuinely per-group-ternary FP16 weight: trits × per-group scale."""
    rng = np.random.default_rng(seed)
    num_groups = in_f // GS
    trits = rng.integers(-1, 2, size=(out_f, in_f)).astype(np.float32)
    trits[:, 0::GS] = 1.0  # each group has a +1 so scale == that magnitude
    scales = (rng.random((out_f, num_groups)) * 0.04 + 0.01).astype(np.float16)
    w = (trits.reshape(out_f, num_groups, GS)
         * scales[:, :, None].astype(np.float32)).reshape(out_f, in_f)
    return torch.from_numpy(w.astype(np.float16))


def _fp16(*shape, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.standard_normal(shape) * 0.02).astype(np.float16))


def _build_tiny_qwen3(model_dir):
    q_out = N_HEADS * HEAD_DIM   # 128
    kv_out = N_KV * HEAD_DIM     # 64
    tensors = {
        "model.embed_tokens.weight": _fp16(VOCAB, HIDDEN, seed=1),
        "model.norm.weight": _fp16(HIDDEN, seed=2),
        "lm_head.weight": _fp16(VOCAB, HIDDEN, seed=3),
    }
    s = 10
    for i in range(N_LAYERS):
        p = f"model.layers.{i}"
        # ternary-eligible 2-D projections
        tensors[f"{p}.self_attn.q_proj.weight"] = _ternary_weight(q_out, HIDDEN, s); s += 1
        tensors[f"{p}.self_attn.k_proj.weight"] = _ternary_weight(kv_out, HIDDEN, s); s += 1
        tensors[f"{p}.self_attn.v_proj.weight"] = _ternary_weight(kv_out, HIDDEN, s); s += 1
        tensors[f"{p}.self_attn.o_proj.weight"] = _ternary_weight(HIDDEN, q_out, s); s += 1
        tensors[f"{p}.mlp.gate_proj.weight"] = _ternary_weight(INTER, HIDDEN, s); s += 1
        tensors[f"{p}.mlp.up_proj.weight"] = _ternary_weight(INTER, HIDDEN, s); s += 1
        tensors[f"{p}.mlp.down_proj.weight"] = _ternary_weight(HIDDEN, INTER, s); s += 1
        # FP16-retained norms (incl. per-head QK-Norm)
        tensors[f"{p}.input_layernorm.weight"] = _fp16(HIDDEN, seed=s); s += 1
        tensors[f"{p}.post_attention_layernorm.weight"] = _fp16(HIDDEN, seed=s); s += 1
        tensors[f"{p}.self_attn.q_norm.weight"] = _fp16(HEAD_DIM, seed=s); s += 1
        tensors[f"{p}.self_attn.k_norm.weight"] = _fp16(HEAD_DIM, seed=s); s += 1

    save_file(tensors, str(model_dir / "model.safetensors"))
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return tensors


def test_pack_g128_tiny_qwen3_end_to_end(tmp_path):
    model_dir = tmp_path / "tiny-qwen3-unpacked"
    model_dir.mkdir()
    src = _build_tiny_qwen3(model_dir)
    out = tmp_path / "tiny.tern-model"

    report = pack_g128_model(
        str(model_dir), "qwen3", str(out), name="tiny", verbose=False)

    # Census: 7 projections × 2 layers ternary; the rest FP16.
    assert report["status"] == "PACKED_GATE_GREEN"
    assert report["g128_layers"] == 7 * N_LAYERS
    assert report["fp16_layers"] == len(src) - 7 * N_LAYERS
    # Genuine ternary substrate → gate passes bit-exact.
    assert report["global_max_abs_error"] == 0.0
    assert out.exists()

    # Reconstruct a ternary layer and an FP16 layer; both match the source.
    reader = TernModelReader(out)
    tern_name = "model.layers.0.mlp.down_proj.weight"
    recon = reader.reconstruct_layer(tern_name)["weight"].to(torch.float32).numpy()
    assert np.allclose(recon, src[tern_name].to(torch.float32).numpy(),
                       atol=1e-3, rtol=0)
    entry = reader._get_manifest_entry(tern_name)
    assert entry["dtype"] == "ternary_g128"
    assert entry["scale_shape"] == [HIDDEN, INTER // GS]

    fp16_name = "model.embed_tokens.weight"
    assert reader._get_manifest_entry(fp16_name)["dtype"] == "float16"


def test_pack_g128_rejects_non_ternary(tmp_path):
    """A non-ternary eligible weight aborts the pack at the gate."""
    model_dir = tmp_path / "bad-qwen3-unpacked"
    model_dir.mkdir()
    _build_tiny_qwen3(model_dir)
    # Overwrite one eligible projection with dense Gaussian (non-ternary).
    from safetensors.torch import load_file
    tensors = load_file(str(model_dir / "model.safetensors"))
    rng = np.random.default_rng(99)
    tensors["model.layers.0.mlp.gate_proj.weight"] = torch.from_numpy(
        rng.standard_normal((INTER, HIDDEN)).astype(np.float16))
    save_file(tensors, str(model_dir / "model.safetensors"))

    with pytest.raises(IngestEquivalenceError, match="group"):
        pack_g128_model(str(model_dir), "qwen3",
                        str(tmp_path / "bad.tern-model"), verbose=False)
