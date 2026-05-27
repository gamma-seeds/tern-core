"""
Tests for the ternary MoE expert bank + packed-MoE loader (Milestone 1
Stage 1).

Fast tests build a synthetic per-expert-sliced MoE ``.tern-model`` (2
layers × 4 experts, Qwen3-style names) and exercise the full routing,
addressing, and forward path with no archive dependency — these run under
the default ``pytest -m "not slow"`` gate.

The slow test loads the real Qwen3-30B-A3B artefact off the archive and
verifies 100% manifest coverage + reconstruction fidelity. Opt-in via
``pytest -m slow``; skips cleanly when the archive is not mounted.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from terncore.moe import PackedTernaryExpertBank, load_moe_packed
from terncore.tern_model import TernModelReader, TernModelWriter

# Synthetic dims (Qwen3-shaped, tiny).
H = 16      # hidden_size
I = 8       # moe_intermediate_size
NL = 2      # num_hidden_layers
NE = 4      # num_experts
VOCAB = 32


def _build_synthetic_moe(path: Path) -> None:
    """Write a tiny per-expert-sliced MoE .tern-model with Qwen3 names."""
    torch.manual_seed(1337)
    w = TernModelWriter({"source": "synthetic-moe", "adapter": "qwen3_moe"})

    for L in range(NL):
        for E in range(NE):
            # gate/up: [moe_intermediate, hidden]; down: [hidden, moe_intermediate]
            w.add_layer(f"model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
                        torch.randn(I, H), dtype="ternary2")
            w.add_layer(f"model.layers.{L}.mlp.experts.{E}.up_proj.weight",
                        torch.randn(I, H), dtype="ternary2")
            w.add_layer(f"model.layers.{L}.mlp.experts.{E}.down_proj.weight",
                        torch.randn(H, I), dtype="ternary2")
        # Attention (ternary)
        for proj in ("q", "k", "v", "o"):
            w.add_layer(f"model.layers.{L}.self_attn.{proj}_proj.weight",
                        torch.randn(H, H), dtype="ternary2")
        # Protected (FP16): norms, q/k norms, router
        for nm in ("input_layernorm", "post_attention_layernorm"):
            w.add_layer(f"model.layers.{L}.{nm}.weight",
                        torch.randn(H), dtype="float16")
        for nm in ("q_norm", "k_norm"):
            w.add_layer(f"model.layers.{L}.self_attn.{nm}.weight",
                        torch.randn(H), dtype="float16")
        w.add_layer(f"model.layers.{L}.mlp.gate.weight",
                    torch.randn(NE, H), dtype="float16")  # router

    # Global protected
    w.add_layer("model.embed_tokens.weight", torch.randn(VOCAB, H), dtype="float16")
    w.add_layer("model.norm.weight", torch.randn(H), dtype="float16")
    w.add_layer("lm_head.weight", torch.randn(VOCAB, H), dtype="float16")

    w.write(str(path))


@pytest.fixture()
def synthetic_reader(tmp_path) -> TernModelReader:
    path = tmp_path / "synthetic_moe.tern-model"
    _build_synthetic_moe(path)
    return TernModelReader(str(path))


# ── Fast unit tests (synthetic) ──────────────────────────────────────


def test_routing_counts(synthetic_reader):
    m = load_moe_packed(synthetic_reader, spot_check_n=3)
    c = m.coverage
    assert c["expert_entries"] == NL * NE * 3            # 24
    assert c["attention_entries"] == NL * 4             # 8
    assert c["protected_entries"] == NL * 5 + 3         # 13
    assert c["skipped_entries"] == 0
    assert c["routed_entries"] == c["total_entries"]    # 45, full coverage


def test_metadata_inferred(synthetic_reader):
    m = load_moe_packed(synthetic_reader)
    assert m.metadata["hidden"] == H
    assert m.metadata["moe_intermediate"] == I
    assert m.metadata["num_experts"] == NE
    assert m.metadata["num_layers"] == NL


def test_bank_addressing(synthetic_reader):
    m = load_moe_packed(synthetic_reader)
    bank = m.bank
    assert len(bank) == NL * NE * 3
    assert bank.layers() == [0, 1]
    assert bank.experts_in_layer(0) == [0, 1, 2, 3]
    assert bank.has(1, 3, "down")
    assert not bank.has(0, 99, "gate")   # absent expert


def test_expert_forward_is_packed_and_finite(synthetic_reader):
    m = load_moe_packed(synthetic_reader)
    gate = m.bank.get(0, 0, "gate")          # [I, H] linear
    out = gate(torch.randn(2, H))
    assert out.shape == (2, I)
    assert torch.isfinite(out).all()
    # Confirm it is packed (2-bit), not a dense nn.Linear.
    assert hasattr(gate, "packed_weights")


def test_attention_routed_as_packed(synthetic_reader):
    m = load_moe_packed(synthetic_reader)
    assert set(k[1] for k in m.attention) == {"q", "k", "v", "o"}
    qproj = m.attention[(0, "q")]
    assert hasattr(qproj, "packed_weights")
    assert torch.isfinite(qproj(torch.randn(1, H))).all()


def test_protected_are_dense_tensors(synthetic_reader):
    m = load_moe_packed(synthetic_reader)
    assert "model.embed_tokens.weight" in m.protected
    assert "lm_head.weight" in m.protected
    assert "model.layers.0.mlp.gate.weight" in m.protected   # router
    embed = m.protected["model.embed_tokens.weight"]
    assert tuple(embed.shape) == (VOCAB, H)
    assert torch.isfinite(embed).all()


def test_spot_check_fidelity(synthetic_reader):
    m = load_moe_packed(synthetic_reader, spot_check_n=4)
    checks = m.coverage["spot_checks"]
    assert len(checks) >= 1
    for ck in checks:
        assert ck["shape_ok"] and ck["finite"]
        assert ck["sparsity_within_tol"]


def test_limit_layers_bounds_load(synthetic_reader):
    m = load_moe_packed(synthetic_reader, limit_layers=1)
    assert m.bank.layers() == [0]
    assert m.coverage["expert_entries"] == NE * 3        # one layer
    assert m.coverage["skipped_entries"] > 0


def test_proj_key_guard():
    bank = PackedTernaryExpertBank()
    with pytest.raises(ValueError, match="projection must be"):
        bank.add(0, 0, "qkv", nn.Identity())


def test_load_packed_model_rejects_moe_manifest(synthetic_reader):
    """Dense loader must fail loud + actionable on a per-expert MoE manifest."""
    with pytest.raises(ValueError, match="load_moe_packed"):
        synthetic_reader.load_packed_model(nn.Linear(H, H))


def test_load_moe_packed_rejects_non_moe(tmp_path):
    """Loader must reject a dense (non per-expert) manifest."""
    w = TernModelWriter({"source": "dense"})
    w.add_layer("model.layers.0.self_attn.q_proj.weight",
                torch.randn(H, H), dtype="ternary2")
    p = tmp_path / "dense.tern-model"
    w.write(str(p))
    with pytest.raises(ValueError, match="does not look like"):
        load_moe_packed(TernModelReader(str(p)))


# ── Slow integration test (real Qwen3-30B-A3B artefact) ───────────────

_QWEN3_MANIFEST = (
    "/Volumes/Syn Archive/models/compressed/qwen3-30b-a3b/"
    "qwen3_30b_a3b_ternary_v0.1.0.tern-model/model.tern-model"
)


@pytest.mark.slow
def test_qwen3_30b_a3b_full_bank_load(capsys):
    """Load the real Qwen3-30B-A3B manifest into the bank; verify coverage.

    Expert weights stay ternary-resident (~7.5 GB); no 57 GB FP16 base.
    Skips when the archive is not mounted.
    """
    if not Path(_QWEN3_MANIFEST).exists():
        pytest.skip("Qwen3-30B-A3B manifest not on disk (archive not mounted).")

    reader = TernModelReader(_QWEN3_MANIFEST)
    m = load_moe_packed(reader, spot_check_n=6, verbose=True)

    c = m.coverage
    # 48 layers × 128 experts × 3 projections = 18,432 expert weights
    assert c["expert_entries"] == 48 * 128 * 3
    assert c["attention_entries"] == 48 * 4          # 192
    assert c["protected_entries"] == 243
    assert c["skipped_entries"] == 0
    assert c["routed_entries"] == c["total_entries"] == 18867

    assert len(m.bank) == 18432
    assert m.bank.layers() == list(range(48))
    assert m.metadata["hidden"] == 2048
    assert m.metadata["moe_intermediate"] == 768
    assert m.metadata["num_experts"] == 128

    for ck in c["spot_checks"]:
        assert ck["shape_ok"] and ck["finite"] and ck["sparsity_within_tol"]

    with capsys.disabled():
        print("\n" + m.summary_str())
