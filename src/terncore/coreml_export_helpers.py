"""FP16 cast guards for the CoreML export path.

Extracted from coreml_export.py to avoid pulling coremltools at
import time when the guards are exercised in isolation (e.g.,
unit tests). The guards are pure numpy logic; the original
co-location with coreml_export.py was incidental.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

FP16_MAX = 65504.0


def _validate_ternary2_alpha(alpha: float, name: str) -> None:
    """Guard a ternary2 alpha against silent FP16 overflow.

    Non-finite alpha indicates a quantiser bug upstream; out-of-range
    alpha indicates a degenerate quantisation case. Either would cast
    to Inf in FP16 and break downstream palettisation. Both raise
    rather than silently clamp — clamping would mask the upstream bug.
    """
    if not np.isfinite(alpha):
        raise ValueError(
            f"Non-finite alpha {alpha} for ternary2 layer "
            f"{name}. This indicates a quantiser bug upstream "
            f"(likely in convert.py ternary2 path); the cast "
            f"to FP16 would silently produce Inf/NaN and break "
            f"downstream palettisation. Refusing to emit."
        )
    if abs(alpha) > FP16_MAX:
        raise ValueError(
            f"Alpha {alpha} for ternary2 layer {name} exceeds "
            f"FP16 range (±{int(FP16_MAX)}). This indicates a "
            f"degenerate quantisation case (e.g., sparse or "
            f"all-zero block); the cast to FP16 would silently "
            f"produce Inf. Refusing to emit."
        )


def _validate_group_scales(scales: np.ndarray, name: str) -> np.ndarray:
    """Array form of :func:`_validate_ternary2_alpha` for ``ternary_g128``.

    A per-group scale array carries one symmetric scale per group of
    weights. At group granularity the degenerate cases the scalar guard
    watches for are *more* likely than per-layer, so each is checked per
    element and the raise names the offending ``(row, group)`` index
    rather than failing anonymously:

    - **Non-finite** (NaN/Inf) → upstream ingest/quantiser bug.
    - **Zero / below the FP16 smallest-subnormal** → not representable in
      FP16 at all; an upstream ingest bug or a group the ingest should
      have collapsed to the all-zero sentinel.
    - **FP16-subnormal band** (``smallest_subnormal ≤ |s| < smallest_normal``)
      → representable, but the Apple Neural Engine flushes FP16 subnormals
      to zero, which would silently zero the group's contribution. These
      are **clamped up to the FP16 smallest-normal** (warn-level log naming
      the tensor + count) rather than rejected — empirically these are a
      handful of legitimate tiny-magnitude groups (e.g. early-layer
      SwiGLU ``gate_proj`` on Bonsai 8B), and clamping a ±5e-05 scale to
      ±6.1e-05 is a negligible perturbation of an already-tiny weight.
    - **FP16 overflow** → casts to Inf and breaks palettisation.

    Returns the (possibly clamped) scale array — callers must use the
    returned array so the clamp reaches the emitted ``constexpr`` weight.
    """
    scales = np.asarray(scales)

    nonfinite = np.argwhere(~np.isfinite(scales))
    if nonfinite.size:
        idx = tuple(int(i) for i in nonfinite[0])
        raise ValueError(
            f"Non-finite per-group scale {scales[idx]} for ternary_g128 "
            f"layer {name} at group index {idx}. This indicates an "
            f"upstream ingest/quantiser bug; the FP16 cast would produce "
            f"Inf/NaN and break downstream palettisation. Refusing to emit."
        )

    min_subnormal = np.finfo(np.float16).smallest_subnormal  # 5.96e-08
    min_normal = np.finfo(np.float16).tiny  # smallest positive FP16 normal
    abs_s = np.abs(scales)

    # Hard reject: zero / below the smallest representable FP16 subnormal.
    hard = np.argwhere(abs_s < min_subnormal)
    if hard.size:
        idx = tuple(int(i) for i in hard[0])
        raise ValueError(
            f"Zero / sub-subnormal per-group scale {scales[idx]} for "
            f"ternary_g128 layer {name} at group index {idx}. A scale "
            f"below the FP16 smallest-subnormal (±{min_subnormal:.2e}) is "
            f"not FP16-representable and signals a group the ingest should "
            f"have collapsed to the all-zero sentinel. Refusing to emit."
        )

    # Subnormal band: representable but an ANE subnormal-flush risk → clamp
    # up to the smallest FP16 normal (sign-preserving) and warn.
    band = (abs_s >= min_subnormal) & (abs_s < min_normal)
    n_band = int(band.sum())
    if n_band:
        scales = scales.copy()  # np.frombuffer views are read-only
        signs = np.where(scales < 0, -1.0, 1.0).astype(scales.dtype)
        scales = np.where(
            band, signs * scales.dtype.type(min_normal), scales
        ).astype(scales.dtype)
        logger.warning(
            "ternary_g128 %s: clamped %d sub-normal per-group scale(s) "
            "up to FP16 smallest-normal (±%.3e) — ANE subnormal-flush "
            "risk. Negligible perturbation of tiny-magnitude groups.",
            name, n_band, min_normal,
        )

    overflow = np.argwhere(np.abs(scales) > FP16_MAX)
    if overflow.size:
        idx = tuple(int(i) for i in overflow[0])
        raise ValueError(
            f"Per-group scale {scales[idx]} for ternary_g128 layer "
            f"{name} at group index {idx} exceeds FP16 range "
            f"(±{int(FP16_MAX)}); the cast would silently produce Inf. "
            f"Refusing to emit."
        )

    return scales


def _cast_fp16_retain_with_guards(
    weight_fp32: np.ndarray, name: str
) -> np.ndarray:
    """Cast a protected FP32 weight to FP16 with finite/range guards.

    Three-way handling distinguishes input-bug (raise) from
    representation-bug (clamp):
      - Source NaN/Inf: raise (source-model corruption upstream).
      - Finite but |value| > FP16_MAX: clamp to ±FP16_MAX with
        operator-visible WARNING. Preserves the run on legitimate
        outliers while surfacing them for investigation.
      - Finite within range: cast as-is.
    """
    if not np.all(np.isfinite(weight_fp32)):
        n_nan = int(np.isnan(weight_fp32).sum())
        n_inf = int(np.isinf(weight_fp32).sum())
        raise ValueError(
            f"Non-finite values in fp16_retain layer {name}: "
            f"{n_nan} NaN, {n_inf} Inf in source FP32 weight. "
            f"This indicates a source-model corruption or an "
            f"upstream bug; refusing to emit silently."
        )
    abs_max = float(np.abs(weight_fp32).max())
    if abs_max > FP16_MAX:
        n_clamped_high = int((weight_fp32 >  FP16_MAX).sum())
        n_clamped_low  = int((weight_fp32 < -FP16_MAX).sum())
        print(
            f"WARNING: fp16_retain layer {name} has values "
            f"outside FP16 range (abs_max={abs_max:.3e}); "
            f"clamping {n_clamped_high} values to +{int(FP16_MAX)} "
            f"and {n_clamped_low} values to -{int(FP16_MAX)}. "
            f"Source weight may have an outlier worth investigating.",
            flush=True,
        )
        weight_fp32 = np.clip(weight_fp32, -FP16_MAX, FP16_MAX)
    return weight_fp32.astype(np.float16)
