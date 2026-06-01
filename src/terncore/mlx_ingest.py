"""
MLX → symmetric per-group ternary ingest.

Derives tern-core's ``ternary_g128`` representation — 2-bit trits plus a
``[out, num_groups]`` FP16 symmetric scale array — directly from a
full-precision (FP16) source weight, the form PrismML publishes in the
``Ternary-Bonsai-*-unpacked`` repos.

Sourcing from the FP16 unpacked weights (rather than the MLX-2bit affine
repo) keeps the derivation honest: a genuine QAT-ternary substrate has,
within every group of ``group_size`` weights along the input axis,
exactly the three values ``{-s, 0, +s}``. Recovering ``(trit, s)`` is
then exact — ``s = max(|w_group|)`` and ``trit = round(w / s)`` land on
``{-1, 0, +1}`` with no residual. A **hard equivalence gate** verifies
this per group and aborts on any group whose reconstruction drifts
beyond FP16 tolerance, naming the offending tensor and group index — so
a non-ternary tensor fed here fails loudly rather than packing lossily.

Symmetry is exploited: the scale alone reconstructs the weight, so no
bias/zero-point is stored (the MLX affine bias term is dropped).

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP_SIZE_DEFAULT = 128

# FP16 has a 10-bit mantissa: one ULP at magnitude m is ~ m * 2**-10.
# The equivalence gate allows two ULP at the group's scale magnitude
# (the largest weight in the group), plus a tiny absolute floor for the
# all-zero-group case.
_FP16_ULP = 2.0 ** -10
_ABS_FLOOR = 1e-8


class IngestEquivalenceError(Exception):
    """Raised when per-group reconstruction drifts beyond FP16 tolerance.

    Names the tensor and the first offending group so a non-ternary
    substrate (or a group-size mismatch) fails loudly at ingest rather
    than silently packing a lossy approximation.
    """


@dataclass(frozen=True)
class GroupQuantResult:
    """Output of :func:`derive_per_group_symmetric`.

    Attributes:
        trits: ``[out, in]`` int8 array with values in ``{-1, 0, +1}``.
        scales: ``[out, num_groups]`` float16 symmetric per-group scales,
            row-major, scale axis aligned to the input dim (MLX
            orientation; consumed directly by writer / CoreML / torch).
        group_size: Weights per group along the input axis.
        max_abs_error: Largest per-element reconstruction error observed
            across all groups (FP16-grade; for reporting).
    """

    trits: np.ndarray
    scales: np.ndarray
    group_size: int
    max_abs_error: float


def derive_per_group_symmetric(
    weight: np.ndarray,
    *,
    name: str = "<tensor>",
    group_size: int = GROUP_SIZE_DEFAULT,
) -> GroupQuantResult:
    """Derive symmetric per-group trits + scales from an FP16/FP32 weight.

    Args:
        weight: ``[out, in]`` source weight. ``in`` must be a multiple of
            ``group_size`` (true for Bonsai: 6144/2048/… are 128-aligned).
        name: Tensor name for diagnostics on gate failure.
        group_size: Weights per group along the input axis (default 128).

    Returns:
        :class:`GroupQuantResult` with trits, scales, and the observed
        max reconstruction error.

    Raises:
        ValueError: ``weight`` is not 2-D or ``in`` is not group-aligned.
        IngestEquivalenceError: any group's reconstruction drifts beyond
            FP16 tolerance (i.e. the source is not cleanly ternary).
    """
    if weight.ndim != 2:
        raise ValueError(
            f"{name}: expected a 2-D [out, in] weight, got shape {weight.shape}."
        )
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"{name}: input dim {in_f} is not a multiple of group_size "
            f"{group_size}. Per-group symmetric ingest requires an "
            f"aligned input dimension."
        )
    num_groups = in_f // group_size

    # Work in fp32; the source is fp16, the gate is fp16-grade.
    w = weight.astype(np.float32).reshape(out_f, num_groups, group_size)

    # Symmetric per-group scale = max(|w|) within the group.
    scale = np.max(np.abs(w), axis=-1)  # [out, num_groups]

    # Guard all-zero groups: a group of all zeros is exactly representable
    # (trit 0 × any scale = 0). Use a sentinel scale of 1.0 so the division
    # is well-defined and the downstream Inf-risk validator stays clean;
    # the trits are all zero so reconstruction is exactly 0 regardless.
    safe_scale = np.where(scale > 0.0, scale, 1.0)

    trits = np.rint(w / safe_scale[:, :, None]).astype(np.int8)  # {-1,0,+1}

    # Hard equivalence gate, per group.
    recon = trits.astype(np.float32) * safe_scale[:, :, None]
    err = np.abs(recon - w)  # [out, num_groups, group_size]
    per_group_max = err.max(axis=-1)  # [out, num_groups]
    # FP16 tolerance scaled to each group's magnitude (two ULP at scale).
    tol = 2.0 * _FP16_ULP * safe_scale + _ABS_FLOOR  # [out, num_groups]

    bad = np.argwhere(per_group_max > tol)
    if bad.size:
        o, g = int(bad[0][0]), int(bad[0][1])
        raise IngestEquivalenceError(
            f"{name}: per-group reconstruction exceeds FP16 tolerance at "
            f"output row {o}, group {g} (max_abs_error="
            f"{per_group_max[o, g]:.3e} > tol={tol[o, g]:.3e}). The source "
            f"weight is not cleanly ternary in this group; refusing to "
            f"pack a lossy approximation."
        )

    trits = trits.reshape(out_f, in_f)
    scales_fp16 = safe_scale.astype(np.float16)  # [out, num_groups] row-major
    return GroupQuantResult(
        trits=trits,
        scales=scales_fp16,
        group_size=group_size,
        max_abs_error=float(per_group_max.max()),
    )


@dataclass(frozen=True)
class PackedGroupLayer:
    """Writer-ready bytes for one ``ternary_g128`` layer.

    Feeds ``TernModelWriter.add_ternary_g128_layer`` directly.
    """

    packed_weights: bytes
    scales: bytes
    shape: list
    scale_shape: list
    group_size: int
    max_abs_error: float


def pack_group_symmetric(
    weight: np.ndarray,
    *,
    name: str = "<tensor>",
    group_size: int = GROUP_SIZE_DEFAULT,
) -> PackedGroupLayer:
    """Derive + 2-bit pack a weight into ``ternary_g128`` writer bytes.

    Runs :func:`derive_per_group_symmetric` (with its hard equivalence
    gate), 2-bit packs the trits through the shared
    ``pack_ternary_weights`` codec, and returns the byte payloads in the
    locked on-disk layout (scales ``[out, num_groups]`` row-major, FP16).
    """
    import torch
    from terncore.sparse import pack_ternary_weights

    res = derive_per_group_symmetric(weight, name=name, group_size=group_size)
    out_f, in_f = res.trits.shape
    num_groups = in_f // group_size

    packed, _bitmap = pack_ternary_weights(
        torch.from_numpy(res.trits.astype(np.int8))
    )
    return PackedGroupLayer(
        packed_weights=bytes(packed.numpy().tobytes()),
        scales=res.scales.astype(np.float16).tobytes(),  # row-major [out, ng]
        shape=[out_f, in_f],
        scale_shape=[out_f, num_groups],
        group_size=group_size,
        max_abs_error=res.max_abs_error,
    )


def reconstruct_from_groups(
    trits: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Reconstruct ``[out, in]`` FP32 weights from trits + per-group scales.

    The inverse of :func:`derive_per_group_symmetric`, shared by the
    reader and the torch/CoreML fidelity tests so all consumers apply the
    identical per-group scaling along the input axis.
    """
    out_f, in_f = trits.shape
    num_groups = in_f // group_size
    t = trits.astype(np.float32).reshape(out_f, num_groups, group_size)
    s = scales.astype(np.float32).reshape(out_f, num_groups, 1)
    return (t * s).reshape(out_f, in_f)
