"""
Tests for the dense Qwen3 adapter (``terncore.adapters.qwen3``).

Q1 recon (2026-05-31) confirmed PrismML Ternary Bonsai 1.7B/8B are
``Qwen3ForCausalLM`` — GQA + SwiGLU + RMSNorm with per-head QK-Norm
(``self_attn.q_norm`` / ``self_attn.k_norm``), and tied embeddings on
the 1.7B checkpoint. This suite pins the weight-classification policy
against the real Bonsai tensor names and guards the dense/MoE routing
boundary (dense Qwen3 must not resolve to the qwen3_moe adapter, and
vice-versa).

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters import get_adapter
from terncore.adapters.base import ArchitectureAdapter, ArchitectureMismatch
from terncore.adapters.qwen3 import Qwen3Adapter

# Representative dense-Qwen3 (Bonsai) tensor names + shapes, taken from
# the 1.7B safetensors index pulled during Q1.
_PROJECTIONS = {
    "model.layers.0.self_attn.q_proj.weight": [2048, 2048],
    "model.layers.0.self_attn.k_proj.weight": [1024, 2048],
    "model.layers.0.self_attn.v_proj.weight": [1024, 2048],
    "model.layers.0.self_attn.o_proj.weight": [2048, 2048],
    "model.layers.0.mlp.gate_proj.weight": [6144, 2048],
    "model.layers.0.mlp.up_proj.weight": [6144, 2048],
    "model.layers.0.mlp.down_proj.weight": [2048, 6144],
}

_PROTECTED = {
    "model.embed_tokens.weight": [151669, 2048],
    "lm_head.weight": [151669, 2048],  # present on untied 8B
    "model.norm.weight": [2048],
    "model.layers.0.input_layernorm.weight": [2048],
    "model.layers.0.post_attention_layernorm.weight": [2048],
    "model.layers.0.self_attn.q_norm.weight": [128],  # per-head QK-Norm
    "model.layers.0.self_attn.k_norm.weight": [128],  # per-head QK-Norm
}


@pytest.fixture
def adapter() -> Qwen3Adapter:
    return Qwen3Adapter()


# ── Registry / identity ───────────────────────────────────────────────
def test_get_adapter_returns_dense_qwen3():
    a = get_adapter("qwen3")
    assert isinstance(a, ArchitectureAdapter)
    assert isinstance(a, Qwen3Adapter)
    assert a.info().name == "qwen3"


def test_info_declares_dense_architecture():
    info = Qwen3Adapter().info()
    assert info.architectures == ["Qwen3ForCausalLM"]
    assert info.model_type == "qwen3"
    assert info.multimodal is False
    # Dense adapter is not MoE — no expert pattern.
    assert info.expert_pattern is None


# ── Architecture routing boundary (dense vs MoE) ──────────────────────
def test_validate_accepts_dense_qwen3(adapter):
    adapter.validate_architecture("Qwen3ForCausalLM")  # must not raise


def test_dense_adapter_rejects_moe_architecture(adapter):
    """The sparse Qwen3MoeForCausalLM must route to qwen3_moe, never
    to the dense adapter."""
    with pytest.raises(ArchitectureMismatch):
        adapter.validate_architecture("Qwen3MoeForCausalLM")


def test_dense_adapter_rejects_llama(adapter):
    with pytest.raises(ArchitectureMismatch):
        adapter.validate_architecture("LlamaForCausalLM")


def test_moe_adapter_rejects_dense_architecture():
    """Symmetric guard: the MoE adapter must not absorb dense Qwen3."""
    moe = get_adapter("qwen3_moe")
    with pytest.raises(ArchitectureMismatch):
        moe.validate_architecture("Qwen3ForCausalLM")


# ── Weight classification ─────────────────────────────────────────────
@pytest.mark.parametrize("name, shape", list(_PROJECTIONS.items()))
def test_projection_weights_are_ternary_eligible(adapter, name, shape):
    cls = adapter.classify_weight(name, shape)
    assert cls.category == "ternary_eligible"
    assert cls.component == "language"


@pytest.mark.parametrize("name, shape", list(_PROTECTED.items()))
def test_protected_weights_are_fp16_retained(adapter, name, shape):
    cls = adapter.classify_weight(name, shape)
    assert cls.category == "fp16_retain"


def test_qk_norm_protected_via_norm_pattern(adapter):
    """The Qwen3-specific per-head QK-Norm weights are FP16-retained —
    the distinguishing architectural feature vs Llama."""
    for name in (
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ):
        cls = adapter.classify_weight(name, [128])
        assert cls.category == "fp16_retain"
        assert "norm" in cls.reason.lower()


def test_one_dimensional_weight_retained_even_without_protected_name(adapter):
    cls = adapter.classify_weight("model.layers.0.some_scale", [2048])
    assert cls.category == "fp16_retain"
    assert "1-D" in cls.reason


# ── Integration over a full representative weight set ──────────────────
def test_get_ternary_eligible_selects_only_projections(adapter):
    all_shapes = {**_PROJECTIONS, **_PROTECTED}
    eligible = set(adapter.get_ternary_eligible(all_shapes))
    assert eligible == set(_PROJECTIONS)


def test_tied_embedding_checkpoint_has_no_lm_head_to_classify(adapter):
    """On tied 1.7B there is no separate lm_head.weight; embed_tokens
    protection alone covers the head. Classification needs no special
    case — absence of the tensor is simply nothing to classify."""
    tied_set = {k: v for k, v in {**_PROJECTIONS, **_PROTECTED}.items()
                if k != "lm_head.weight"}
    eligible = set(adapter.get_ternary_eligible(tied_set))
    assert eligible == set(_PROJECTIONS)
    assert "model.embed_tokens.weight" not in eligible


# ── Block helpers ─────────────────────────────────────────────────────
def test_block_index_and_membership(adapter):
    name = "model.layers.7.self_attn.q_proj.weight"
    assert adapter.is_block_weight(name) is True
    assert adapter.block_index(name) == 7
    assert adapter.is_block_weight("model.embed_tokens.weight") is False
