"""
Tests for the Gemma 4 Unified (encoder-free 12B) architecture adapter.

Covers registration/resolution, the architecture allow-list, and the
weight-classification contract that matters for a text-only ternary pass:
the language_model decoder tower is ternary-eligible while every inline
encoder-free multimodal embedder tensor is FP16-retained.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

import pytest

from terncore.adapters import get_adapter
from terncore.adapters.base import ArchitectureMismatch


@pytest.fixture
def adapter():
    return get_adapter("gemma4_unified")


def test_resolves_and_reports_unified_identity(adapter):
    info = adapter.info()
    assert info.name == "gemma4_unified"
    assert info.model_type == "gemma4_unified"
    assert info.architectures == ["Gemma4UnifiedForConditionalGeneration"]
    assert info.multimodal is True
    # 12B Unified is dense — no MoE expert handling.
    assert info.expert_pattern is None


def test_architecture_allow_list(adapter):
    # Exact match passes.
    adapter.validate_architecture("Gemma4UnifiedForConditionalGeneration")
    # The encoder-based variant must NOT silently route here.
    with pytest.raises(ArchitectureMismatch):
        adapter.validate_architecture("Gemma4ForConditionalGeneration")


@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", [4096, 3840]),
        ("model.language_model.layers.0.self_attn.v_proj.weight", [2048, 3840]),
        ("model.language_model.layers.0.mlp.gate_proj.weight", [15360, 3840]),
        ("model.language_model.layers.0.mlp.down_proj.weight", [3840, 15360]),
        # A global layer (idx 5) still ternarises the projections it DOES have
        # (q/k/o); it simply lacks v_proj — handled by absence, not by a rule.
        ("model.language_model.layers.5.self_attn.q_proj.weight", [4096, 3840]),
    ],
)
def test_decoder_tower_is_ternary_eligible(adapter, name, shape):
    c = adapter.classify_weight(name, shape)
    assert c.category == "ternary_eligible", c.reason
    assert c.component == "language"


@pytest.mark.parametrize(
    "name,shape,component",
    [
        # Encoder-free inline embedder — all FP16-retained.
        ("model.embed_vision.embedding_projection.weight", [3840, 1152], "vision"),
        ("model.embed_audio.embedding_projection.weight", [3840, 640], "audio"),
        # vision_embedder.* is unique to the Unified topology; its 2-D
        # patch_dense.weight would fall through to ternary without the
        # adapter's extended vision pattern — this is the regression guard.
        ("model.vision_embedder.patch_dense.weight", [1152, 588], "vision"),
        ("model.vision_embedder.pos_embedding", [1, 256, 1152], "vision"),
    ],
)
def test_multimodal_embedder_is_fp16_retained(adapter, name, shape, component):
    c = adapter.classify_weight(name, shape)
    assert c.category == "fp16_retain", c.reason
    assert c.component == component


@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.language_model.layers.0.input_layernorm.weight", [3840]),
        ("model.language_model.layers.0.layer_scalar", [3840]),
        ("model.language_model.embed_tokens.weight", [262144, 3840]),
        ("model.language_model.norm.weight", [3840]),
    ],
)
def test_protected_language_weights_stay_fp16(adapter, name, shape):
    c = adapter.classify_weight(name, shape)
    assert c.category == "fp16_retain", c.reason
