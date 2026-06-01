"""
Numerical-fidelity tests for the Phase 2 Qwen3 runtime ops in the
CoreML builder: per-head QK-Norm and YaRN-scaled RoPE.

Each op is checked against the HuggingFace ``transformers.models.qwen3``
reference implementation, using the real Bonsai 8B ``config.json`` for
shape parameters. Weights are small seeded toy tensors — no real Bonsai
weights are pulled. Tolerances are FP16-grade: the builder tabulates
cos/sin in FP16, so the YaRN tables and the composed attention block are
compared within FP16 rounding.

Reference math sources (pinned during Phase 2 recon):
- YaRN: ``transformers.modeling_rope_utils._compute_yarn_parameters``
- RoPE apply / rotate_half: ``modeling_qwen3.apply_rotary_pos_emb``
- QK-Norm: ``modeling_qwen3.Qwen3RMSNorm`` (plain-weight, eps-in-rsqrt)

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
# terncore.coreml_export (imported below) pulls coremltools at module load;
# skip the whole module where coremltools is absent (e.g. CI installs
# .[dev,transformers] only) rather than failing collection.
pytest.importorskip("coremltools")

from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3Attention,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from terncore.arch_presets import RopeConfig, resolve_preset  # noqa: E402
from terncore.coreml_export import (  # noqa: E402
    _precompute_rope_freqs,
    _yarn_rope_freqs,
    rope_cos_sin,
)

FIXTURES = Path(__file__).parent / "fixtures"
# FP16 cos/sin tabulation tolerance.
FP16_ATOL = 2e-3


def _cfg8() -> dict:
    return json.loads((FIXTURES / "qwen3_bonsai_8b_config.json").read_text())


def _hf_config(cfg: dict, *, num_layers: int = 1, vocab: int = 256) -> Qwen3Config:
    """Build a single-block HF Qwen3Config from a Bonsai config dict."""
    qc = Qwen3Config(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        num_hidden_layers=num_layers,
        intermediate_size=cfg["intermediate_size"],
        vocab_size=vocab,
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_theta=cfg["rope_theta"],
        rope_scaling=cfg["rope_scaling"],
        rms_norm_eps=cfg["rms_norm_eps"],
        attention_bias=False,
    )
    qc._attn_implementation = "eager"
    return qc


# ── YaRN frequency tables — the core new op ───────────────────────────
def test_yarn_tables_match_hf_rotary_8b():
    """_yarn_rope_freqs reproduces Qwen3RotaryEmbedding's cos/sin
    (including the mscale attention factor) for the Bonsai 8B config."""
    cfg = _cfg8()
    head_dim = cfg["head_dim"]
    seq = 24
    qc = _hf_config(cfg)

    rotary = Qwen3RotaryEmbedding(config=qc)
    pos = torch.arange(seq).unsqueeze(0)
    cos_hf, sin_hf = rotary(torch.zeros(1, seq, qc.hidden_size), pos)
    # HF tables are full head_dim (cat([h, h])); compare the first half.
    cos_hf_half = cos_hf[0, :, : head_dim // 2].to(torch.float32).numpy()
    sin_hf_half = sin_hf[0, :, : head_dim // 2].to(torch.float32).numpy()

    cos_my, sin_my = _yarn_rope_freqs(
        seq, head_dim,
        theta=cfg["rope_theta"],
        factor=cfg["rope_scaling"]["factor"],
        original_max_position_embeddings=cfg["rope_scaling"][
            "original_max_position_embeddings"
        ],
    )
    assert cos_my.shape == (seq, head_dim // 2)
    assert np.allclose(cos_my.astype(np.float32), cos_hf_half, atol=FP16_ATOL)
    assert np.allclose(sin_my.astype(np.float32), sin_hf_half, atol=FP16_ATOL)


def test_yarn_differs_from_vanilla_rope():
    """YaRN scaling actually changes the frequencies — guards against a
    silent fall-through to the vanilla path."""
    cfg = _cfg8()
    seq, head_dim = 16, cfg["head_dim"]
    cos_yarn, _ = _yarn_rope_freqs(
        seq, head_dim, theta=cfg["rope_theta"], factor=4.0,
        original_max_position_embeddings=16384,
    )
    cos_van, _ = _precompute_rope_freqs(seq, head_dim=head_dim, theta=cfg["rope_theta"])
    assert not np.allclose(cos_yarn.astype(np.float32), cos_van.astype(np.float32),
                           atol=FP16_ATOL)


def test_yarn_attention_factor_folded_in():
    """At position 0 every cos entry equals the mscale attention factor
    (cos(0) * attention_factor), confirming the scaling is applied."""
    cfg = _cfg8()
    factor = cfg["rope_scaling"]["factor"]
    expected_af = 0.1 * math.log(factor) + 1.0  # get_mscale(factor)
    cos_my, _ = _yarn_rope_freqs(
        4, cfg["head_dim"], theta=cfg["rope_theta"], factor=factor,
        original_max_position_embeddings=16384,
    )
    assert np.allclose(cos_my[0].astype(np.float32), expected_af, atol=FP16_ATOL)


# ── RopeConfig dispatch / Llama no-regression ─────────────────────────
def test_rope_cos_sin_vanilla_equals_precompute_no_regression():
    """A non-YaRN RopeConfig produces tables bit-identical to the
    pre-Phase-2 vanilla function — the Llama path is untouched."""
    cos_v, sin_v = _precompute_rope_freqs(20, head_dim=128, theta=500000.0)
    cos_r, sin_r = rope_cos_sin(20, 128, RopeConfig(theta=500000.0))
    assert np.array_equal(cos_v, cos_r)
    assert np.array_equal(sin_v, sin_r)


def test_rope_cos_sin_routes_bonsai_preset_to_yarn():
    """The Phase-1 preset hook drives the op: resolve_preset(8B).rope
    is YaRN and rope_cos_sin emits the YaRN tables."""
    cfg = _cfg8()
    rope = resolve_preset(cfg).rope
    assert rope.is_yarn is True
    cos_r, sin_r = rope_cos_sin(12, cfg["head_dim"], rope)
    cos_y, sin_y = _yarn_rope_freqs(
        12, cfg["head_dim"], theta=cfg["rope_theta"], factor=4.0,
        original_max_position_embeddings=16384,
    )
    assert np.array_equal(cos_r, cos_y)
    assert np.array_equal(sin_r, sin_y)


def test_rope_cos_sin_yarn_missing_params_raises():
    bad = RopeConfig(theta=1e6, rope_type="yarn", factor=None,
                     original_max_position_embeddings=None)
    with pytest.raises(ValueError):
        rope_cos_sin(8, 128, bad)


# ── QK-Norm — per-head RMSNorm fidelity ───────────────────────────────
def _numpy_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Mirror of the builder's _rms_norm_cfg over the last axis."""
    x32 = x.astype(np.float32)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 / np.sqrt(var + eps)
    return normed * weight.astype(np.float32)


def test_qk_norm_matches_qwen3_rmsnorm():
    cfg = _cfg8()
    head_dim, eps = cfg["head_dim"], cfg["rms_norm_eps"]
    torch.manual_seed(0)
    weight = torch.randn(head_dim)
    x = torch.randn(1, cfg["num_attention_heads"], 8, head_dim)

    norm = Qwen3RMSNorm(head_dim, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(weight)
        ref = norm(x).numpy()

    mine = _numpy_rms_norm(x.numpy(), weight.numpy(), eps)
    assert np.allclose(mine, ref, atol=1e-4)


# ── Composed single attention block: QK-Norm + YaRN-RoPE end-to-end ───
def _repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """[b, kv_heads, s, d] -> [b, kv_heads*n_rep, s, d] (GQA tile)."""
    b, kv, s, d = x.shape
    return np.repeat(x, n_rep, axis=1)


def _rotate_half_np(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([-x2, x1], axis=-1)


def _numpy_attention_block(hidden, weights, cfg, cos_full, sin_full):
    """numpy mirror of the MIL block: proj -> per-head QK-Norm -> YaRN
    RoPE -> GQA -> full (non-causal) attention -> o_proj. Uses the same
    weights as the HF module so any delta is op math, not init."""
    h = cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    b, s, _ = hidden.shape

    def proj(x, w):  # mb.linear: x @ w.T
        return x @ w.T

    q = proj(hidden, weights["q_proj"]).reshape(b, s, h, hd)
    k = proj(hidden, weights["k_proj"]).reshape(b, s, kv, hd)
    v = proj(hidden, weights["v_proj"]).reshape(b, s, kv, hd)

    # per-head QK-Norm over head_dim (applied on the [.., heads, hd] view)
    q = _numpy_rms_norm(q, weights["q_norm"], eps)
    k = _numpy_rms_norm(k, weights["k_norm"], eps)

    q = q.transpose(0, 2, 1, 3)  # [b, h, s, hd]
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # RoPE with YaRN cos/sin (full head_dim tables), HF rotate_half form
    cos = cos_full[None, None, :, :]
    sin = sin_full[None, None, :, :]
    q = q * cos + _rotate_half_np(q) * sin
    k = k * cos + _rotate_half_np(k) * sin

    k = _repeat_kv(k, h // kv)
    v = _repeat_kv(v, h // kv)

    scale = 1.0 / math.sqrt(hd)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # full attention, no mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = probs @ v  # [b, h, s, hd]
    out = out.transpose(0, 2, 1, 3).reshape(b, s, h * hd)
    return out @ weights["o_proj"].T


def test_single_attention_block_matches_hf_qwen3():
    """One Qwen3 attention block (QK-Norm + YaRN RoPE composed) matches
    HF to FP16 tolerance for a single forward pass — the Phase 2
    end-to-end op assertion. Full (non-causal) attention on both sides,
    matching the builder's maskless export path."""
    cfg = _cfg8()
    hd, h, kv = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    seq = 6
    qc = _hf_config(cfg)

    torch.manual_seed(1)
    attn = Qwen3Attention(qc, layer_idx=0).eval()
    rotary = Qwen3RotaryEmbedding(config=qc)
    pos = torch.arange(seq).unsqueeze(0)
    hidden = torch.randn(1, seq, qc.hidden_size)
    cos_hf, sin_hf = rotary(hidden, pos)

    # HF reference: full attention via an all-allowed additive mask.
    full_mask = torch.zeros(1, 1, seq, seq)
    with torch.no_grad():
        ref_out, _ = attn(hidden, (cos_hf, sin_hf), full_mask)
    ref_out = ref_out.numpy()

    # Same weights from the HF module → numpy mirror with YaRN tables.
    sd = attn.state_dict()
    weights = {
        "q_proj": sd["q_proj.weight"].numpy(),
        "k_proj": sd["k_proj.weight"].numpy(),
        "v_proj": sd["v_proj.weight"].numpy(),
        "o_proj": sd["o_proj.weight"].numpy(),
        "q_norm": sd["q_norm.weight"].numpy(),
        "k_norm": sd["k_norm.weight"].numpy(),
    }
    cos_my, sin_my = _yarn_rope_freqs(
        seq, hd, theta=cfg["rope_theta"], factor=4.0,
        original_max_position_embeddings=16384,
    )
    # builder applies the [seq, hd/2] table to both halves → full table.
    cos_full = np.concatenate([cos_my, cos_my], axis=-1).astype(np.float32)
    sin_full = np.concatenate([sin_my, sin_my], axis=-1).astype(np.float32)

    mine = _numpy_attention_block(
        hidden.numpy(), weights, cfg, cos_full, sin_full
    )

    assert mine.shape == ref_out.shape
    assert np.allclose(mine, ref_out, atol=1e-2, rtol=1e-2)
