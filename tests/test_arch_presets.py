"""
Tests for ``terncore.arch_presets`` — declarative topology presets
resolved from a HuggingFace ``config.json``.

Phase 1 of the qwen3 arch-preset work (Q1 recon confirmed Bonsai is
``Qwen3ForCausalLM``: GQA 16/8, RMSNorm, SwiGLU, per-head QK-Norm,
YaRN RoPE, tied embeddings on 1.7B but not 8B). These tests pin the
topology resolution against the real Bonsai 1.7B / 8B configs pulled
during Q1, and guard the Llama path against regression.

Runtime QK-Norm / YaRN operators are deferred to Phase 2; this suite
asserts the declarative fields only.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terncore.arch_presets import (
    ArchPreset,
    ArchPresetMismatch,
    RopeConfig,
    registered_architectures,
    resolve_preset,
    resolve_preset_from_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ── Bonsai 1.7B — full topology, tied embeddings ──────────────────────
def test_bonsai_1_7b_resolves_to_qwen3_topology():
    preset = resolve_preset(_load("qwen3_bonsai_1.7b_config.json"))

    assert isinstance(preset, ArchPreset)
    assert preset.name == "qwen3"
    assert preset.model_type == "qwen3"
    assert preset.architectures == ("Qwen3ForCausalLM",)

    # Core dimensions
    assert preset.num_layers == 28
    assert preset.hidden_size == 2048
    assert preset.intermediate_size == 6144
    assert preset.num_attention_heads == 16
    assert preset.num_key_value_heads == 8
    assert preset.head_dim == 128
    assert preset.vocab_size == 151669
    assert preset.max_position_embeddings == 32768

    # GQA ratio (16 query heads / 8 KV heads)
    assert preset.gqa_groups == 2
    assert preset.is_gqa is True

    # Norm / MLP / activation scheme
    assert preset.norm_type == "rmsnorm"
    assert preset.mlp_type == "swiglu"
    assert preset.activation == "silu"
    assert preset.rms_norm_eps == pytest.approx(1e-6)

    # QK-Norm present (intrinsic to Qwen3)
    assert preset.has_qk_norm is True

    # Tied embeddings — TRUE on 1.7B
    assert preset.tie_word_embeddings is True

    # YaRN RoPE
    assert preset.rope.is_yarn is True
    assert preset.rope.rope_type == "yarn"
    assert preset.rope.theta == pytest.approx(1_000_000.0)
    assert preset.rope.factor == pytest.approx(4.0)
    assert preset.rope.original_max_position_embeddings == 8192


# ── Bonsai 8B — same shape, untied embeddings, wider YaRN base ────────
def test_bonsai_8b_resolves_to_qwen3_topology_untied():
    preset = resolve_preset(_load("qwen3_bonsai_8b_config.json"))

    assert preset.name == "qwen3"
    assert preset.architectures == ("Qwen3ForCausalLM",)

    # Core dimensions (scaled up from 1.7B)
    assert preset.num_layers == 36
    assert preset.hidden_size == 4096
    assert preset.intermediate_size == 12288
    assert preset.num_attention_heads == 32
    assert preset.num_key_value_heads == 8
    assert preset.head_dim == 128
    assert preset.vocab_size == 151669
    assert preset.max_position_embeddings == 65536

    # GQA ratio (32 / 8)
    assert preset.gqa_groups == 4

    # Same family scheme + QK-Norm
    assert preset.norm_type == "rmsnorm"
    assert preset.mlp_type == "swiglu"
    assert preset.has_qk_norm is True

    # Tied embeddings — FALSE on 8B (the structural delta from 1.7B)
    assert preset.tie_word_embeddings is False

    # YaRN with the 8B's wider original window
    assert preset.rope.is_yarn is True
    assert preset.rope.factor == pytest.approx(4.0)
    assert preset.rope.original_max_position_embeddings == 16384


def test_1_7b_and_8b_share_topology_shape_differ_on_tie_and_scale():
    """The two checkpoints are the same family; they differ on the
    weight-tying flag and on scale, exactly as Q1 flagged."""
    small = resolve_preset(_load("qwen3_bonsai_1.7b_config.json"))
    large = resolve_preset(_load("qwen3_bonsai_8b_config.json"))

    # Same family / scheme
    assert small.name == large.name == "qwen3"
    assert small.norm_type == large.norm_type
    assert small.mlp_type == large.mlp_type
    assert small.has_qk_norm == large.has_qk_norm is True
    assert small.num_key_value_heads == large.num_key_value_heads == 8
    assert small.head_dim == large.head_dim == 128

    # Differ on tying — the smaller model is the more entangled one
    assert small.tie_word_embeddings is True
    assert large.tie_word_embeddings is False


# ── Llama no-regression ───────────────────────────────────────────────
def test_llama_config_resolves_to_llama_preset_no_qk_norm():
    preset = resolve_preset(_load("llama32_1b_config.json"))

    assert preset.name == "llama"
    assert preset.model_type == "llama"
    assert "LlamaForCausalLM" in preset.architectures

    # Llama topology
    assert preset.num_layers == 16
    assert preset.hidden_size == 2048
    assert preset.num_attention_heads == 32
    assert preset.num_key_value_heads == 8
    assert preset.head_dim == 64

    # Llama family carries no QK-Norm
    assert preset.has_qk_norm is False
    assert preset.norm_type == "rmsnorm"
    assert preset.mlp_type == "swiglu"

    # Non-YaRN RoPE scaling (llama3) is not misread as YaRN
    assert preset.rope.is_yarn is False
    assert preset.rope.theta == pytest.approx(500_000.0)


def test_head_dim_falls_back_when_config_omits_it():
    """Older Llama configs omit head_dim; it derives from
    hidden_size // num_attention_heads."""
    cfg = _load("llama32_1b_config.json")
    cfg.pop("head_dim")
    preset = resolve_preset(cfg)
    assert preset.head_dim == cfg["hidden_size"] // cfg["num_attention_heads"]


# ── Dense Qwen3 stays distinct from the MoE adapter path ──────────────
def test_dense_qwen3_distinct_from_qwen3_moe():
    """``Qwen3MoeForCausalLM`` is a different topology (served by the
    qwen3_moe adapter) and does not resolve through the dense preset."""
    with pytest.raises(ArchPresetMismatch):
        resolve_preset(
            {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}
        )


# ── Dispatch / fallback / errors ──────────────────────────────────────
def test_model_type_fallback_when_architectures_absent():
    cfg = _load("qwen3_bonsai_1.7b_config.json")
    cfg.pop("architectures")
    preset = resolve_preset(cfg)
    assert preset.name == "qwen3"


def test_unknown_architecture_raises_with_registered_set():
    with pytest.raises(ArchPresetMismatch) as exc:
        resolve_preset({"architectures": ["NonesuchForCausalLM"]})
    msg = str(exc.value)
    assert "NonesuchForCausalLM" in msg
    assert "Qwen3ForCausalLM" in msg  # registered set is surfaced


def test_resolve_from_dir_reads_config_json(tmp_path):
    cfg = _load("qwen3_bonsai_8b_config.json")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    preset = resolve_preset_from_dir(tmp_path)
    assert preset.name == "qwen3"
    assert preset.tie_word_embeddings is False


def test_resolve_from_dir_missing_config_raises():
    with pytest.raises(ArchPresetMismatch) as exc:
        resolve_preset_from_dir("/nonexistent/model/dir")
    assert "config.json not found" in str(exc.value)


def test_registered_architectures_lists_qwen3_and_llama():
    archs = registered_architectures()
    assert "Qwen3ForCausalLM" in archs
    assert "LlamaForCausalLM" in archs


# ── Preset immutability (frozen dataclass) ────────────────────────────
def test_preset_is_frozen():
    preset = resolve_preset(_load("qwen3_bonsai_1.7b_config.json"))
    with pytest.raises(Exception):
        preset.hidden_size = 9999  # type: ignore[misc]


def test_rope_config_default_when_no_scaling():
    rope = RopeConfig.from_config({"rope_theta": 10000.0})
    assert rope.is_yarn is False
    assert rope.rope_type == "default"
    assert rope.factor is None
