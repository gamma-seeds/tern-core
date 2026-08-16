"""
Tests for the HRM-Text dual-timescale recurrent adapter
(``terncore.adapters.hrm_text``).

Probe HRM_PROBE_20260613T011321Z (2026-06-13) confirmed
``sapientinc/HRM-Text-1B`` is ``HrmTextForCausalLM`` — two separately-
parameterised H/L block stacks with **fused** ``attn.gqkv_proj`` (gated
Q/K/V) and ``mlp.gate_up_proj`` projections in the safetensors, and
parameterless MagicNorm (no block-norm weights). The only protected
tensors are the embeddings, the untied LM head, and the 1-D recurrent
initialiser ``model.z_L_init``. This suite pins the weight-classification
policy against the real fused tensor names + shapes, the H/L stack
tagging, and the architecture routing boundary.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters import get_adapter
from terncore.adapters.base import ArchitectureAdapter, ArchitectureMismatch
from terncore.adapters.hrm_text import HrmTextAdapter

# Real fused HRM-Text-1B projection tensors (raw safetensors layout),
# one representative layer per stack. gqkv_proj = fused gated Q/K/V
# [4*hidden, hidden]; gate_up_proj = fused MLP gate+up [2*inter, hidden].
_PROJECTIONS = {
    "model.H_module.layers.0.attn.gqkv_proj.weight": [6144, 1536],
    "model.H_module.layers.0.attn.o_proj.weight": [1536, 1536],
    "model.H_module.layers.0.mlp.gate_up_proj.weight": [8192, 1536],
    "model.H_module.layers.0.mlp.down_proj.weight": [1536, 4096],
    "model.L_module.layers.0.attn.gqkv_proj.weight": [6144, 1536],
    "model.L_module.layers.0.attn.o_proj.weight": [1536, 1536],
    "model.L_module.layers.0.mlp.gate_up_proj.weight": [8192, 1536],
    "model.L_module.layers.0.mlp.down_proj.weight": [1536, 4096],
}

# Protected: untied embeddings + LM head, and the 1-D recurrent init.
_PROTECTED = {
    "model.embed_tokens.weight": [65536, 1536],
    "lm_head.weight": [65536, 1536],
    "model.z_L_init": [1536],
}


@pytest.fixture
def adapter() -> HrmTextAdapter:
    return HrmTextAdapter()


# ── Registry / identity ───────────────────────────────────────────────
def test_get_adapter_returns_hrm_text():
    a = get_adapter("hrm_text")
    assert isinstance(a, ArchitectureAdapter)
    assert isinstance(a, HrmTextAdapter)
    assert a.info().name == "hrm_text"


def test_info_declares_recurrent_architecture():
    info = HrmTextAdapter().info()
    assert info.architectures == ["HrmTextForCausalLM"]
    assert info.model_type == "hrm_text"
    assert info.multimodal is False
    # Dense recurrent, not MoE — no expert pattern / stacking.
    assert info.expert_pattern is None


# ── Architecture routing boundary ─────────────────────────────────────
def test_validate_accepts_hrm_text(adapter):
    adapter.validate_architecture("HrmTextForCausalLM")  # must not raise


def test_hrm_adapter_rejects_other_architectures(adapter):
    for arch in ("Qwen3ForCausalLM", "LlamaForCausalLM", "Gemma4ForConditionalGeneration"):
        with pytest.raises(ArchitectureMismatch):
            adapter.validate_architecture(arch)


def test_other_adapters_reject_hrm_text():
    """Symmetric guard: a standard-transformer adapter must not absorb HRM."""
    for name in ("qwen3", "llama", "gemma4_unified"):
        with pytest.raises(ArchitectureMismatch):
            get_adapter(name).validate_architecture("HrmTextForCausalLM")


# ── Weight classification (fused projections) ─────────────────────────
@pytest.mark.parametrize("name, shape", list(_PROJECTIONS.items()))
def test_fused_projection_weights_are_ternary_eligible(adapter, name, shape):
    cls = adapter.classify_weight(name, shape)
    assert cls.category == "ternary_eligible"
    assert cls.component == "language"


@pytest.mark.parametrize("name, shape", list(_PROTECTED.items()))
def test_protected_weights_are_fp16_retained(adapter, name, shape):
    cls = adapter.classify_weight(name, shape)
    assert cls.category == "fp16_retain"


def test_z_l_init_protected_by_explicit_name(adapter):
    """The 1-D recurrent initialiser is named explicitly in the protection
    patterns (and the 1-D rule would retain it regardless)."""
    cls = adapter.classify_weight("model.z_L_init", [1536])
    assert cls.category == "fp16_retain"
    assert "z_l_init" in cls.reason.lower()


def test_one_dimensional_weight_retained_even_without_protected_name(adapter):
    cls = adapter.classify_weight("model.H_module.layers.0.some_scale", [1536])
    assert cls.category == "fp16_retain"
    assert "1-D" in cls.reason


# ── H/L stack tagging ─────────────────────────────────────────────────
def test_stack_of_tags_h_l_and_shared(adapter):
    assert adapter.stack_of("model.H_module.layers.3.attn.gqkv_proj.weight") == "H"
    assert adapter.stack_of("model.L_module.layers.3.mlp.down_proj.weight") == "L"
    assert adapter.stack_of("model.embed_tokens.weight") == "shared"
    assert adapter.stack_of("lm_head.weight") == "shared"
    assert adapter.stack_of("model.z_L_init") == "shared"


def test_h_and_l_layer_zero_are_distinct_entries(adapter):
    """H and L share `.layers.N.` numbering but are separate stacks — the
    converter must not collide them (probe verified 64 H + 64 L)."""
    h_name = "model.H_module.layers.0.attn.gqkv_proj.weight"
    l_name = "model.L_module.layers.0.attn.gqkv_proj.weight"
    assert h_name != l_name
    assert adapter.stack_of(h_name) == "H"
    assert adapter.stack_of(l_name) == "L"
    assert adapter.classify_weight(h_name, [6144, 1536]).category == "ternary_eligible"
    assert adapter.classify_weight(l_name, [6144, 1536]).category == "ternary_eligible"


# ── Integration over a full representative weight set ──────────────────
def test_get_ternary_eligible_selects_only_fused_projections(adapter):
    all_shapes = {**_PROJECTIONS, **_PROTECTED}
    eligible = set(adapter.get_ternary_eligible(all_shapes))
    assert eligible == set(_PROJECTIONS)
    assert "model.embed_tokens.weight" not in eligible
    assert "model.z_L_init" not in eligible


# ── Block helpers ─────────────────────────────────────────────────────
def test_block_index_and_membership(adapter):
    name = "model.L_module.layers.7.attn.o_proj.weight"
    assert adapter.is_block_weight(name) is True
    assert adapter.block_index(name) == 7
    assert adapter.is_block_weight("model.embed_tokens.weight") is False
