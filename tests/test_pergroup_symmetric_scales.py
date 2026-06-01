"""
Phase 3 tests — per-group symmetric scale support (``ternary_g128``).

Covers the six scope items: round-trip binary parity, reader loud-fail
on unknown dtype (regression), the MLX equivalence gate against a real
Bonsai 1.7B-unpacked slice (offline fixture, no full download), the
CoreML emitter producing valid MIL at block_size=128, torch forward
parity against the dequantised reference, and the array-valued scale
validator's loud raise on a degenerate group.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from terncore.coreml_export_helpers import _validate_group_scales
from terncore.mlx_ingest import (
    GROUP_SIZE_DEFAULT,
    IngestEquivalenceError,
    derive_per_group_symmetric,
    pack_group_symmetric,
    reconstruct_from_groups,
)
from terncore.packed_linear import PackedTernaryLinear
from terncore.tern_model import TernModelReader, TernModelWriter

FIXTURES = Path(__file__).parent / "fixtures"
GS = GROUP_SIZE_DEFAULT


def _synthetic_group_ternary(out=8, num_groups=3, group_size=GS, seed=0):
    """A genuinely per-group-ternary FP16 weight: trits × per-group scale."""
    rng = np.random.default_rng(seed)
    in_f = num_groups * group_size
    trits = rng.integers(-1, 2, size=(out, in_f)).astype(np.float32)
    # ensure each group has at least one +1 so scale == that group's magnitude
    trits[:, 0::group_size] = 1.0
    scales = (rng.random((out, num_groups)) * 0.04 + 0.01).astype(np.float16)
    w = (
        trits.reshape(out, num_groups, group_size)
        * scales[:, :, None].astype(np.float32)
    ).reshape(out, in_f)
    return w.astype(np.float16)


def _write_artifact(tmp_path, weight, name="model.layers.0.mlp.down_proj.weight"):
    pg = pack_group_symmetric(weight.astype(np.float32), name=name, group_size=GS)
    w = TernModelWriter()
    w.add_ternary_g128_layer(
        name=name,
        packed_weights=pg.packed_weights,
        scales=pg.scales,
        shape=pg.shape,
        scale_shape=pg.scale_shape,
        group_size=pg.group_size,
    )
    path = tmp_path / "g128.tern-model"
    w.write(path)
    return path, name, pg


# ── 1. Round-trip write/read (binary parity) ──────────────────────────
def test_roundtrip_write_read_binary_parity(tmp_path):
    weight = _synthetic_group_ternary()
    path, name, pg = _write_artifact(tmp_path, weight)

    reader = TernModelReader(path)
    entry = reader._get_manifest_entry(name)
    assert entry["dtype"] == "ternary_g128"
    assert entry["group_size"] == GS
    assert entry["scale_shape"] == [weight.shape[0], weight.shape[1] // GS]
    assert entry["scale_layout"] == "out_groups_row_major"

    recon = reader.reconstruct_layer(name)["weight"].numpy()
    # Reconstruction equals the original FP16 weight within FP16 tolerance.
    assert recon.shape == weight.shape
    assert np.allclose(recon, weight.astype(np.float32), atol=1e-3, rtol=0)


# ── 2. Reader loud-fail on unknown dtype (regression) ─────────────────
def test_reader_raises_on_unknown_dtype(tmp_path):
    weight = _synthetic_group_ternary()
    path, name, _ = _write_artifact(tmp_path, weight)
    reader = TernModelReader(path)
    # Corrupt the in-memory manifest entry's dtype → reconstruct must raise.
    reader._get_manifest_entry(name)["dtype"] = "garbage_dtype"
    with pytest.raises(ValueError, match="Unknown dtype"):
        reader.reconstruct_layer(name)


# ── 3. MLX equivalence gate on real Bonsai 1.7B-unpacked slice ────────
def test_mlx_equivalence_gate_on_bonsai_slice():
    fixture = FIXTURES / "bonsai_1.7b_downproj_slice.npy"
    weight = np.load(fixture)  # [4, 6144] FP16 from prism-ml/...-unpacked
    assert weight.shape[1] % GS == 0

    res = derive_per_group_symmetric(
        weight, name="bonsai.down_proj[slice]", group_size=GS
    )
    # Genuine ternary substrate → reconstruction is exact to FP16.
    assert res.trits.min() >= -1 and res.trits.max() <= 1
    assert res.max_abs_error < 1e-3
    recon = reconstruct_from_groups(res.trits, res.scales, GS)
    assert np.allclose(recon, weight.astype(np.float32), atol=1e-3, rtol=0)


def test_equivalence_gate_rejects_non_ternary():
    """A non-ternary tensor (Gaussian) must abort the ingest loudly."""
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((4, 256)).astype(np.float16)
    with pytest.raises(IngestEquivalenceError, match="group"):
        derive_per_group_symmetric(weight, name="dense.gaussian", group_size=GS)


# ── 4. CoreML emitter: ternary_g128 → valid MIL @ block_size=128 ──────
def test_coreml_emitter_block_size_128(tmp_path):
    pytest.importorskip("coremltools")  # skip where coremltools is absent (CI)
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb, types
    from terncore.coreml_export import _load_weight_for_coreml, _inject_weight

    weight = _synthetic_group_ternary(out=16, num_groups=2, group_size=GS)
    path, name, pg = _write_artifact(tmp_path, weight)
    reader = TernModelReader(path)

    kind, data, scales, padded_in = _load_weight_for_coreml(reader, name)
    assert kind == "int4"
    assert data.shape == (16, weight.shape[1])
    assert scales.shape == (16, weight.shape[1] // GS)  # [out, num_groups]
    assert padded_in == weight.shape[1]

    in_dim = weight.shape[1]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, in_dim), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        w = _inject_weight(reader, name)
        return mb.linear(x=x, weight=w, name="out")

    m = ct.convert(
        prog, source="milinternal", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )
    assert m is not None  # mlprogram built with the real per-group scales


# ── 5. torch forward parity vs dequantised reference ──────────────────
def test_torch_forward_parity(tmp_path):
    weight = _synthetic_group_ternary(out=12, num_groups=4, group_size=GS)
    path, name, pg = _write_artifact(tmp_path, weight)
    reader = TernModelReader(path)

    # Reader-reconstructed reference weight.
    w_recon = reader.reconstruct_layer(name)["weight"].to(torch.float32)

    layer = PackedTernaryLinear.from_packed_group_data(
        packed_weights=torch.frombuffer(bytearray(pg.packed_weights), dtype=torch.uint8).clone(),
        group_scales=torch.frombuffer(bytearray(pg.scales), dtype=torch.float16).clone()
        .to(torch.float32).reshape(pg.scale_shape),
        group_size=pg.group_size,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
    )

    x = torch.randn(5, weight.shape[1], dtype=torch.float32)
    got = layer(x)
    ref = torch.nn.functional.linear(x, w_recon)
    assert got.shape == (5, weight.shape[0])
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)


def test_scalar_alpha_path_unchanged():
    """group_size==0 keeps the scalar-alpha forward path intact."""
    layer = PackedTernaryLinear(in_features=8, out_features=4, bias=False, alpha=0.5)
    assert layer.group_size == 0
    assert layer.group_scales is None
    out = layer(torch.randn(2, 8))
    assert out.shape == (2, 4)


# ── 6. Array-valued validation: degenerate group raises loudly ────────
def test_validate_group_scales_clean_passes():
    scales = np.array([[0.04, 0.03], [0.02, 0.05]], dtype=np.float16)
    _validate_group_scales(scales, "ok.layer")  # must not raise


def test_validate_group_scales_zero_group_raises():
    scales = np.array([[0.04, 0.0], [0.02, 0.05]], dtype=np.float16)
    with pytest.raises(ValueError, match="group index"):
        _validate_group_scales(scales, "degenerate.layer")


def test_validate_group_scales_nonfinite_raises():
    scales = np.array([[0.04, 0.03], [np.inf, 0.05]], dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite"):
        _validate_group_scales(scales, "naninf.layer")


def test_validate_group_scales_subnormal_band_clamped():
    """FP16-subnormal scales clamp up to smallest-normal (ANE flush risk)."""
    min_normal = np.finfo(np.float16).tiny  # 6.104e-05
    # 5.746e-05 is the real Bonsai-8B offender (gate_proj, layer 1); sign-preserved.
    scales = np.array([[0.04, 5.746e-05], [-5.0e-05, 0.05]], dtype=np.float16)
    out = _validate_group_scales(scales, "bonsai.gate_proj")
    # Sub-normal entries pulled up to ±min_normal; normal entries untouched.
    assert out[0, 1] == np.float16(min_normal)
    assert out[1, 0] == np.float16(-min_normal)
    assert out[0, 0] == np.float16(0.04)
    assert out[1, 1] == np.float16(0.05)
    # No subnormal magnitudes remain below the FP16 normal floor.
    nz = np.abs(out)[np.abs(out) > 0]
    assert (nz >= np.float16(min_normal)).all()


def test_validate_group_scales_below_subnormal_raises():
    """A scale below the FP16 smallest-subnormal is unrepresentable → reject."""
    scales = np.array([[0.04, 1e-9], [0.02, 0.05]], dtype=np.float32)
    with pytest.raises(ValueError, match="group index"):
        _validate_group_scales(scales, "underflow.layer")
