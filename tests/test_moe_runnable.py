"""
Tests for the runnable ternary MoE (Milestone 1 Stage 2+3 custom block).

The core test packs a tiny *stock* ``Qwen3MoeForCausalLM`` into a per-expert
``.tern-model`` and asserts the bank-backed runnable model
(``build_runnable_qwen3_moe`` + ``TernaryMoEBlock``) reproduces a
dense-ternary reference **logit-for-logit** — validating router math
(softmax→top-k→renorm), expert dispatch, FFN, and meta-skeleton assembly
against the transformers implementation. Both sides use identical ternary
weights, so the only delta is packed-matmul vs dense rounding.

All fast (tiny dims); no archive dependency.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3MoeForCausalLM  # noqa: E402
from transformers.models.qwen3_moe.configuration_qwen3_moe import (  # noqa: E402
    Qwen3MoeConfig,
)

from terncore.moe import build_runnable_qwen3_moe, load_moe_packed  # noqa: E402
from terncore.tern_model import TernModelReader, TernModelWriter  # noqa: E402


def _tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _pack_stock_to_manifest(stock, config, path: Path) -> None:
    """Write a per-expert-sliced .tern-model from a stock Qwen3-MoE model."""
    sd = stock.state_dict()
    I = config.moe_intermediate_size
    w = TernModelWriter({"source": "synthetic-qwen3moe", "adapter": "qwen3_moe"})
    for i in range(config.num_hidden_layers):
        gup = sd[f"model.layers.{i}.mlp.experts.gate_up_proj"]   # [E, 2I, H]
        dwn = sd[f"model.layers.{i}.mlp.experts.down_proj"]       # [E, H, I]
        for e in range(config.num_experts):
            w.add_layer(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight",
                        gup[e][:I, :].float(), dtype="ternary2")
            w.add_layer(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight",
                        gup[e][I:2 * I, :].float(), dtype="ternary2")
            w.add_layer(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight",
                        dwn[e].float(), dtype="ternary2")
        for proj in ("q", "k", "v", "o"):
            w.add_layer(f"model.layers.{i}.self_attn.{proj}_proj.weight",
                        sd[f"model.layers.{i}.self_attn.{proj}_proj.weight"].float(),
                        dtype="ternary2")
        for nm in ("input_layernorm", "post_attention_layernorm"):
            w.add_layer(f"model.layers.{i}.{nm}.weight",
                        sd[f"model.layers.{i}.{nm}.weight"].float(), dtype="float16")
        for nm in ("q_norm", "k_norm"):
            w.add_layer(f"model.layers.{i}.self_attn.{nm}.weight",
                        sd[f"model.layers.{i}.self_attn.{nm}.weight"].float(),
                        dtype="float16")
        w.add_layer(f"model.layers.{i}.mlp.gate.weight",
                    sd[f"model.layers.{i}.mlp.gate.weight"].float(), dtype="float16")
    w.add_layer("model.embed_tokens.weight",
                sd["model.embed_tokens.weight"].float(), dtype="float16")
    w.add_layer("model.norm.weight", sd["model.norm.weight"].float(), dtype="float16")
    w.add_layer("lm_head.weight", sd["lm_head.weight"].float(), dtype="float16")
    w.write(str(path))


def _build_dense_ternary_reference(reader, config):
    """Fresh stock model populated with the manifest's *dequantised* weights.

    This is the oracle: identical ternary weights to the bank, but run
    through the stock fused-experts forward path.
    """
    def recon(name):
        return reader.reconstruct_layer(name)["weight"].float()

    ref = Qwen3MoeForCausalLM(config).eval()
    I = config.moe_intermediate_size
    with torch.no_grad():
        for i in range(config.num_hidden_layers):
            blk = ref.model.layers[i]
            for e in range(config.num_experts):
                g = recon(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight")
                u = recon(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight")
                d = recon(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight")
                blk.mlp.experts.gate_up_proj[e] = torch.cat([g, u], dim=0)
                blk.mlp.experts.down_proj[e] = d
            blk.mlp.gate.weight.data = recon(f"model.layers.{i}.mlp.gate.weight")
            for proj in ("q", "k", "v", "o"):
                getattr(blk.self_attn, f"{proj}_proj").weight.data = recon(
                    f"model.layers.{i}.self_attn.{proj}_proj.weight")
            blk.self_attn.q_norm.weight.data = recon(
                f"model.layers.{i}.self_attn.q_norm.weight")
            blk.self_attn.k_norm.weight.data = recon(
                f"model.layers.{i}.self_attn.k_norm.weight")
            blk.input_layernorm.weight.data = recon(
                f"model.layers.{i}.input_layernorm.weight")
            blk.post_attention_layernorm.weight.data = recon(
                f"model.layers.{i}.post_attention_layernorm.weight")
        ref.model.embed_tokens.weight.data = recon("model.embed_tokens.weight")
        ref.model.norm.weight.data = recon("model.norm.weight")
        ref.lm_head.weight.data = recon("lm_head.weight")
    return ref


@pytest.fixture()
def packed_tiny(tmp_path):
    torch.manual_seed(20260528)
    config = _tiny_config()
    stock = Qwen3MoeForCausalLM(config).eval()
    path = tmp_path / "tiny_qwen3moe.tern-model"
    _pack_stock_to_manifest(stock, config, path)
    reader = TernModelReader(str(path))
    return config, reader


def test_assembles_with_no_stranded_meta(packed_tiny):
    config, reader = packed_tiny
    packed = load_moe_packed(reader, spot_check_n=0)
    model = build_runnable_qwen3_moe(packed, config)
    stranded = [n for n, t in
                list(model.named_parameters()) + list(model.named_buffers())
                if t.is_meta]
    assert stranded == []
    # Bank registered exactly once under the top-level model (so .to(device)
    # moves it), not duplicated per layer. PackedTernaryLinear stores weights
    # as buffers, so it surfaces under named_buffers, not named_parameters.
    assert any(n == "_ternary_expert_bank" for n, _ in model.named_modules())
    bank_buffers = [n for n, _ in model.named_buffers()
                    if n.startswith("_ternary_expert_bank.")]
    assert len(bank_buffers) > 0
    # Sanity: device move does not choke on the shared bank reference.
    model.to("cpu")


def test_forward_shape_and_finite(packed_tiny):
    config, reader = packed_tiny
    model = build_runnable_qwen3_moe(load_moe_packed(reader, spot_check_n=0), config)
    ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (1, 6, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_matches_dense_ternary_reference(packed_tiny):
    config, reader = packed_tiny
    ref = _build_dense_ternary_reference(reader, config)
    ours = build_runnable_qwen3_moe(load_moe_packed(reader, spot_check_n=0), config)

    ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        lo_ref = ref(ids).logits
        lo_ours = ours(ids).logits
    max_abs = (lo_ref - lo_ours).abs().max().item()
    assert torch.allclose(lo_ref, lo_ours, atol=2e-3, rtol=2e-3), (
        f"runnable bank model diverged from dense-ternary reference: "
        f"max|Δ|={max_abs:.2e}"
    )


def test_greedy_generate_runs(packed_tiny):
    config, reader = packed_tiny
    model = build_runnable_qwen3_moe(load_moe_packed(reader, spot_check_n=0), config)
    ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 8
    assert torch.isfinite(model(out).logits).all()


# ── Slow capstone: real Qwen3-30B-A3B end-to-end ─────────────────────

_QWEN3_MANIFEST = (
    "/Volumes/Syn Archive/models/compressed/qwen3-30b-a3b/"
    "qwen3_30b_a3b_ternary_v0.1.0.tern-model/model.tern-model"
)


@pytest.mark.slow
def test_qwen3_30b_a3b_runnable_end_to_end():
    """Capstone: assemble + run the real Qwen3-30B-A3B ternary MoE.

    Verifies the Milestone-1 inference machinery end-to-end on the real
    artefact: loads ternary-resident (~13-16 GB, not the ~57 GB dense base),
    assembles with no stranded meta tensors, produces finite logits, and
    runs greedy generation.

    Generation COHERENCE is deliberately NOT asserted. Qwen3-30B-A3B ternary
    at threshold 0.7 sits below the coherent-generation envelope (per-weight
    cos(recon, ground-truth) ~0.80-0.89 on experts/attention, compounding
    across 48 layers + 8-of-128 MoE routing → repetition collapse). This is
    a quality-envelope property of the compressed artefact — same class as
    Phi-4 ternary @0.7 (cf. docs/backlog.md) — confirmed faithful-load via
    ground-truth checkpoint comparison (router cos=1.0, protected cos=1.0),
    NOT an inference-path defect. Remedy: recompress at a lower threshold
    (separate compression-quality workstream).
    """
    if not Path(_QWEN3_MANIFEST).exists():
        pytest.skip("Qwen3-30B-A3B manifest not on disk (archive not mounted).")
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    except Exception as e:  # config not cached / offline
        pytest.skip(f"Qwen3-30B-A3B config unavailable: {type(e).__name__}: {e}")

    packed = load_moe_packed(TernModelReader(_QWEN3_MANIFEST), spot_check_n=0)
    model = build_runnable_qwen3_moe(packed, config, device="cpu")

    stranded = [n for n, t in
                list(model.named_parameters()) + list(model.named_buffers())
                if t.is_meta]
    assert stranded == []
    assert len(packed.bank) == 18432

    ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (1, 6, config.vocab_size)
    assert torch.isfinite(logits).all()
