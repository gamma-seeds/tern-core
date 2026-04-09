# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.annotator — pure Python, no MLX deps.

Includes the second cross-repo trip-wire:
``test_threshold_constants_match_confidence_emitter`` imports
ConfidenceEmitter from tern-runtime and asserts the threshold class
constants are byte-identical. If a future contributor changes one
side without the other, that test fails before any training run can
proceed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tern-core/harness importable
HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# tern-runtime/src importable for the cross-repo trip-wire
TERN_RUNTIME_SRC = (
    Path(__file__).resolve().parents[3] / "tern-runtime" / "src"
)
if str(TERN_RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(TERN_RUNTIME_SRC))


from harness.annotator import EpistemicAnnotator, StepAnnotation
from harness.epistemic_state import Domain, EpistemicLabel, EpistemicState
from harness.projector import ProjectionResult


# ---------------------------------------------------------------------------
# Helpers — construct ProjectionResult and EpistemicLabel without MLX
# ---------------------------------------------------------------------------

def _result(sparsity: float, alpha: float = 0.1, threshold: float = 0.07) -> ProjectionResult:
    """Build a ProjectionResult with the fields the annotator reads.

    The annotator only consumes ``sparsity`` so the array fields can
    be left as ``None`` — Python's frozen-dataclass machinery does not
    introspect them.
    """
    return ProjectionResult(
        weights_ternary=None,    # type: ignore[arg-type]
        weights_dequant=None,    # type: ignore[arg-type]
        alpha=alpha,
        sparsity=sparsity,
        threshold=threshold,
    )


def _label(state: EpistemicState, score: float = 0.5) -> EpistemicLabel:
    return EpistemicLabel(
        epistemic_state=state,
        confidence_score=score,
        escalate=False,
        domain=Domain.FACTUAL,
        source_reliability=0.8,
    )


# ---------------------------------------------------------------------------
# Classification — three states by sparsity proxy
# ---------------------------------------------------------------------------

def test_confirmed_annotation_at_high_sparsity():
    """sparsity >= 0.85 → CONFIRMED."""
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.96), _label(EpistemicState.CONFIRMED, 0.9))
    assert sa.epistemic_state == EpistemicState.CONFIRMED
    assert sa.predicted_score == pytest.approx(0.96)
    assert sa.sparsity == pytest.approx(0.96)


def test_uncertain_annotation_at_mid_sparsity():
    """0.45 <= sparsity < 0.85 → UNCERTAIN."""
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.65), _label(EpistemicState.UNCERTAIN, 0.6))
    assert sa.epistemic_state == EpistemicState.UNCERTAIN


def test_disconfirmed_annotation_at_low_sparsity():
    """sparsity < 0.45 → DISCONFIRMED."""
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.20), _label(EpistemicState.DISCONFIRMED, 0.2))
    assert sa.epistemic_state == EpistemicState.DISCONFIRMED


def test_classification_boundary_confirmed():
    """sparsity == 0.85 → CONFIRMED (boundary maps upward)."""
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.85), _label(EpistemicState.CONFIRMED, 0.85))
    assert sa.epistemic_state == EpistemicState.CONFIRMED


def test_classification_boundary_uncertain():
    """sparsity == 0.45 → UNCERTAIN (boundary maps upward)."""
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.45), _label(EpistemicState.UNCERTAIN, 0.45))
    assert sa.epistemic_state == EpistemicState.UNCERTAIN


def test_just_below_confirmed_is_uncertain():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.8499), _label(EpistemicState.UNCERTAIN, 0.85))
    assert sa.epistemic_state == EpistemicState.UNCERTAIN


def test_just_below_uncertain_is_disconfirmed():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(sparsity=0.4499), _label(EpistemicState.DISCONFIRMED, 0.45))
    assert sa.epistemic_state == EpistemicState.DISCONFIRMED


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------

def test_calibration_error_is_absolute_difference():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(
        _result(sparsity=0.92),
        _label(EpistemicState.CONFIRMED, 0.80),
    )
    # |0.92 - 0.80| = 0.12
    assert sa.calibration_error == pytest.approx(0.12, abs=1e-9)


def test_calibration_error_is_zero_when_scores_match():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(
        _result(sparsity=0.50),
        _label(EpistemicState.UNCERTAIN, 0.50),
    )
    assert sa.calibration_error == 0.0


def test_calibration_error_never_negative():
    """The error is the absolute value, so swapping which side is
    higher must not produce a negative."""
    annotator = EpistemicAnnotator()
    sa1 = annotator.annotate(
        _result(sparsity=0.30),
        _label(EpistemicState.UNCERTAIN, 0.70),
    )
    sa2 = annotator.annotate(
        _result(sparsity=0.70),
        _label(EpistemicState.UNCERTAIN, 0.30),
    )
    assert sa1.calibration_error == sa2.calibration_error == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# is_correct flag
# ---------------------------------------------------------------------------

def test_is_correct_when_states_match():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(
        _result(sparsity=0.92),
        _label(EpistemicState.CONFIRMED, 0.85),
    )
    assert sa.is_correct is True


def test_is_correct_false_when_states_differ():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(
        _result(sparsity=0.92),                          # → CONFIRMED
        _label(EpistemicState.UNCERTAIN, 0.65),
    )
    assert sa.is_correct is False


# ---------------------------------------------------------------------------
# Batch annotate
# ---------------------------------------------------------------------------

def test_batch_annotate_length_matches():
    annotator = EpistemicAnnotator()
    results = [_result(0.95), _result(0.60), _result(0.30)]
    labels = [
        _label(EpistemicState.CONFIRMED, 0.9),
        _label(EpistemicState.UNCERTAIN, 0.6),
        _label(EpistemicState.DISCONFIRMED, 0.2),
    ]
    annotations = annotator.batch_annotate(results, labels)
    assert len(annotations) == 3
    assert annotations[0].epistemic_state == EpistemicState.CONFIRMED
    assert annotations[1].epistemic_state == EpistemicState.UNCERTAIN
    assert annotations[2].epistemic_state == EpistemicState.DISCONFIRMED
    assert all(a.is_correct for a in annotations)


def test_batch_annotate_rejects_mismatched_lengths():
    annotator = EpistemicAnnotator()
    with pytest.raises(ValueError, match="same length"):
        annotator.batch_annotate(
            [_result(0.9), _result(0.6)],
            [_label(EpistemicState.CONFIRMED, 0.9)],
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_correct_counts_and_mean():
    annotator = EpistemicAnnotator()
    results = [
        _result(0.95),  # CONFIRMED
        _result(0.92),  # CONFIRMED
        _result(0.65),  # UNCERTAIN
        _result(0.20),  # DISCONFIRMED
        _result(0.10),  # DISCONFIRMED
    ]
    labels = [
        _label(EpistemicState.CONFIRMED, 0.90),
        _label(EpistemicState.CONFIRMED, 0.88),
        _label(EpistemicState.UNCERTAIN, 0.60),
        _label(EpistemicState.DISCONFIRMED, 0.20),
        _label(EpistemicState.UNCERTAIN, 0.50),  # mismatch — predicted DISCONFIRMED
    ]
    annotations = annotator.batch_annotate(results, labels)
    summary = annotator.summary(annotations)

    assert summary["n"] == 5
    assert summary["confirmed_count"] == 2
    assert summary["uncertain_count"] == 1
    assert summary["disconfirmed_count"] == 2
    assert summary["accuracy"] == pytest.approx(4 / 5)  # 4 correct, 1 wrong
    expected_mean_sparsity = (0.95 + 0.92 + 0.65 + 0.20 + 0.10) / 5
    assert summary["mean_sparsity"] == pytest.approx(expected_mean_sparsity)
    expected_mean_cal = sum(a.calibration_error for a in annotations) / 5
    assert summary["mean_calibration_error"] == pytest.approx(expected_mean_cal)


def test_summary_empty_returns_zeros():
    annotator = EpistemicAnnotator()
    summary = annotator.summary([])
    assert summary["n"] == 0
    assert summary["mean_calibration_error"] == 0.0
    assert summary["accuracy"] == 0.0
    assert summary["confirmed_count"] == 0
    assert summary["uncertain_count"] == 0
    assert summary["disconfirmed_count"] == 0
    assert summary["mean_sparsity"] == 0.0


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

def test_step_annotation_is_frozen():
    annotator = EpistemicAnnotator()
    sa = annotator.annotate(_result(0.9), _label(EpistemicState.CONFIRMED, 0.9))
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        sa.predicted_score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CROSS-REPO THRESHOLD TRIP-WIRE — second day-one trip-wire
# ---------------------------------------------------------------------------

def test_threshold_constants_match_confidence_emitter():
    """The TFH EpistemicAnnotator and the LIS ConfidenceEmitter must
    declare byte-identical CONFIRMED and UNCERTAIN threshold values.

    This is the second cross-repo trip-wire (the first lives in
    test_epistemic_state.test_cross_repo_string_match). String values
    and threshold cut points are the two halves of the vocabulary
    contract — both must hold or training-time labels and
    inference-time confidence drift apart.

    If this test fails, change BOTH files in the SAME commit. Do not
    weaken the test by aliasing or remapping.
    """
    try:
        from tern_runtime.inspector.confidence_emitter import (
            ConfidenceEmitter,
        )
    except ImportError as e:
        pytest.skip(
            f"tern-runtime not importable from this venv: {e}. "
            f"The trip-wire requires both repos in PYTHONPATH."
        )

    assert (
        EpistemicAnnotator.CONFIRMED_THRESHOLD
        == ConfidenceEmitter.CONFIRMED_THRESHOLD
        == 0.85
    ), (
        f"CONFIRMED threshold drift: TFH="
        f"{EpistemicAnnotator.CONFIRMED_THRESHOLD} "
        f"LIS={ConfidenceEmitter.CONFIRMED_THRESHOLD}"
    )
    assert (
        EpistemicAnnotator.UNCERTAIN_THRESHOLD
        == ConfidenceEmitter.UNCERTAIN_THRESHOLD
        == 0.45
    ), (
        f"UNCERTAIN threshold drift: TFH="
        f"{EpistemicAnnotator.UNCERTAIN_THRESHOLD} "
        f"LIS={ConfidenceEmitter.UNCERTAIN_THRESHOLD}"
    )
