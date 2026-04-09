# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.objective — pure Python, no MLX deps.

The objective takes ProjectionResult instances as input but only
reads their .sparsity field, so test fixtures construct them with
None for the array fields. This keeps the test pure Python.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


from harness.epistemic_state import Domain, EpistemicLabel, EpistemicState
from harness.objective import ConfidenceObjective, ObjectiveResult
from harness.projector import ProjectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(sparsity: float) -> ProjectionResult:
    return ProjectionResult(
        weights_ternary=None,    # type: ignore[arg-type]
        weights_dequant=None,    # type: ignore[arg-type]
        alpha=0.1,
        sparsity=sparsity,
        threshold=0.07,
    )


def _label(state: EpistemicState, score: float = 0.9) -> EpistemicLabel:
    return EpistemicLabel(
        epistemic_state=state,
        confidence_score=score,
        escalate=False,
        domain=Domain.FACTUAL,
        source_reliability=0.8,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_default_sparsity_target():
    obj = ConfidenceObjective()
    assert obj.sparsity_target == 0.90


def test_construction_validates_sparsity_target():
    with pytest.raises(ValueError, match="sparsity_target"):
        ConfidenceObjective(sparsity_target=0.0)
    with pytest.raises(ValueError, match="sparsity_target"):
        ConfidenceObjective(sparsity_target=1.5)
    with pytest.raises(ValueError, match="sparsity_target"):
        ConfidenceObjective(sparsity_target=-0.1)


# ---------------------------------------------------------------------------
# Compute — basic shape
# ---------------------------------------------------------------------------

def test_compute_returns_objective_result():
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=2.5,
        projection_results=[_result(0.92), _result(0.95)],
        labels=[_label(EpistemicState.CONFIRMED, 0.92)],
        alpha=1.0,
    )
    assert isinstance(result, ObjectiveResult)
    assert result.task_loss == 2.5
    assert result.alpha_used == 1.0
    assert result.mean_predicted_sparsity == pytest.approx(0.935)


def test_compute_validates_inputs():
    obj = ConfidenceObjective()
    with pytest.raises(ValueError, match="alpha"):
        obj.compute(2.5, [_result(0.9)], [_label(EpistemicState.CONFIRMED)], alpha=-0.1)
    with pytest.raises(ValueError, match="projection_results"):
        obj.compute(2.5, [], [_label(EpistemicState.CONFIRMED)], alpha=1.0)
    with pytest.raises(ValueError, match="labels"):
        obj.compute(2.5, [_result(0.9)], [], alpha=1.0)


# ---------------------------------------------------------------------------
# Alpha gating — confidence terms vanish at alpha=0
# ---------------------------------------------------------------------------

def test_total_loss_equals_task_loss_at_alpha_zero():
    """During the warmup window the scheduler returns alpha=0; the
    composite loss should reduce to the task loss alone."""
    obj = ConfidenceObjective()
    # Pick a config where calibration AND sparsity penalty are both > 0
    result = obj.compute(
        task_loss=3.0,
        projection_results=[_result(0.20)],   # well below target 0.90
        labels=[_label(EpistemicState.CONFIRMED, 0.95)],
        alpha=0.0,
    )
    assert result.calibration_penalty > 0.0
    assert result.sparsity_penalty > 0.0
    # But the total ignores them at alpha=0
    assert result.total_loss == pytest.approx(3.0)


def test_total_loss_includes_penalties_at_alpha_one():
    """At alpha=1.0 the composite picks up calibration + 0.1*sparsity."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=3.0,
        projection_results=[_result(0.20)],
        labels=[_label(EpistemicState.CONFIRMED, 0.95)],
        alpha=1.0,
    )
    expected = 3.0 + (
        result.calibration_penalty
        + ConfidenceObjective.SPARSITY_PENALTY_WEIGHT * result.sparsity_penalty
    )
    assert result.total_loss == pytest.approx(expected)


def test_total_loss_scales_linearly_with_alpha():
    """Doubling alpha must double the contribution from the
    confidence terms (the task loss is unaffected)."""
    obj = ConfidenceObjective()
    args = dict(
        task_loss=2.0,
        projection_results=[_result(0.30)],
        labels=[_label(EpistemicState.CONFIRMED, 0.90)],
    )
    a05 = obj.compute(alpha=0.5, **args)
    a10 = obj.compute(alpha=1.0, **args)

    contrib_05 = a05.total_loss - a05.task_loss
    contrib_10 = a10.total_loss - a10.task_loss
    assert contrib_10 == pytest.approx(2.0 * contrib_05)


# ---------------------------------------------------------------------------
# Sparsity penalty
# ---------------------------------------------------------------------------

def test_sparsity_penalty_zero_when_at_or_above_target():
    """We never penalise excess sparsity — only the shortfall counts."""
    obj = ConfidenceObjective(sparsity_target=0.90)
    result_at = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.90)],
        labels=[_label(EpistemicState.CONFIRMED, 0.90)],
        alpha=1.0,
    )
    result_above = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.96)],
        labels=[_label(EpistemicState.CONFIRMED, 0.96)],
        alpha=1.0,
    )
    assert result_at.sparsity_penalty == 0.0
    assert result_above.sparsity_penalty == 0.0


def test_sparsity_penalty_squared_shortfall():
    """At sparsity 0.40 with target 0.90, shortfall=0.50, penalty=0.25."""
    obj = ConfidenceObjective(sparsity_target=0.90)
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.40)],
        labels=[_label(EpistemicState.CONFIRMED, 0.90)],
        alpha=0.0,  # disable composite to inspect penalty in isolation
    )
    assert result.sparsity_penalty == pytest.approx(0.25)


def test_sparsity_penalty_increases_monotonically_as_sparsity_falls():
    obj = ConfidenceObjective(sparsity_target=0.90)
    penalties = []
    for s in [0.85, 0.70, 0.50, 0.30, 0.10]:
        r = obj.compute(
            task_loss=1.0,
            projection_results=[_result(s)],
            labels=[_label(EpistemicState.CONFIRMED, 0.90)],
            alpha=0.0,
        )
        penalties.append(r.sparsity_penalty)
    for i in range(1, len(penalties)):
        assert penalties[i] > penalties[i - 1]


# ---------------------------------------------------------------------------
# Calibration penalty — KL divergence properties
# ---------------------------------------------------------------------------

def test_calibration_penalty_is_zero_when_distributions_match():
    """If predicted state matches label state AND the sparsity
    proxy equals the label confidence_score, the predicted and target
    distributions are identical → KL divergence is 0."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[_label(EpistemicState.CONFIRMED, 0.92)],
        alpha=1.0,
    )
    assert result.calibration_penalty == pytest.approx(0.0, abs=1e-9)


def test_calibration_penalty_positive_when_states_disagree():
    """Predicted CONFIRMED vs labelled DISCONFIRMED → KL divergence > 0."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],   # predicts CONFIRMED
        labels=[_label(EpistemicState.DISCONFIRMED, 0.92)],
        alpha=1.0,
    )
    assert result.calibration_penalty > 0.0


def test_calibration_penalty_grows_with_disagreement():
    """A label that puts more mass on a state the predictor missed
    must produce a larger KL than a label that puts less mass there."""
    obj = ConfidenceObjective()
    soft_disagreement = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[_label(EpistemicState.UNCERTAIN, 0.50)],
        alpha=1.0,
    )
    hard_disagreement = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[_label(EpistemicState.UNCERTAIN, 0.95)],
        alpha=1.0,
    )
    assert hard_disagreement.calibration_penalty > soft_disagreement.calibration_penalty


def test_calibration_penalty_handles_degenerate_distributions():
    """A label with confidence_score=1.0 puts all mass on one state.
    The predicted distribution always has at least some mass on every
    state (because we split (1-score)/2). So KL is finite and positive
    when the predicted state differs."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=0.0,
        projection_results=[_result(0.20)],   # predicts DISCONFIRMED
        labels=[_label(EpistemicState.CONFIRMED, 1.0)],
        alpha=1.0,
    )
    assert math.isfinite(result.calibration_penalty)
    assert result.calibration_penalty > 0.0


# ---------------------------------------------------------------------------
# Distribution helpers — directly tested at the class level
# ---------------------------------------------------------------------------

def test_state_to_distribution_sums_to_one():
    for state in EpistemicState:
        for score in [0.1, 0.5, 0.85, 1.0]:
            dist = ConfidenceObjective._state_to_distribution(state, score)
            assert sum(dist) == pytest.approx(1.0)
            assert all(0.0 <= p <= 1.0 for p in dist)


def test_state_to_distribution_places_score_at_named_state():
    confirmed = ConfidenceObjective._state_to_distribution(
        EpistemicState.CONFIRMED, 0.80
    )
    assert confirmed == pytest.approx((0.80, 0.10, 0.10))
    uncertain = ConfidenceObjective._state_to_distribution(
        EpistemicState.UNCERTAIN, 0.60
    )
    assert uncertain == pytest.approx((0.20, 0.60, 0.20))
    disconfirmed = ConfidenceObjective._state_to_distribution(
        EpistemicState.DISCONFIRMED, 0.40
    )
    assert disconfirmed == pytest.approx((0.30, 0.30, 0.40))


def test_kl_divergence_zero_for_identical_distributions():
    p = (0.7, 0.2, 0.1)
    assert ConfidenceObjective._kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_positive_for_different_distributions():
    p = (0.7, 0.2, 0.1)
    q = (0.1, 0.2, 0.7)
    kl = ConfidenceObjective._kl_divergence(p, q)
    assert kl > 0.0


def test_kl_divergence_handles_zero_target_mass():
    """If a target component is zero, that term contributes nothing
    to the sum (limit of p log(p/q) as p → 0 is 0)."""
    p = (1.0, 0.0, 0.0)
    q = (0.5, 0.3, 0.2)
    kl = ConfidenceObjective._kl_divergence(p, q)
    # Only the first term contributes: 1.0 * log(1.0 / 0.5) = log(2)
    assert kl == pytest.approx(math.log(2.0))


# ---------------------------------------------------------------------------
# Multi-layer projection aggregation
# ---------------------------------------------------------------------------

def test_multiple_projection_results_averaged():
    """The objective takes the mean sparsity across all layer projections."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.80), _result(0.90), _result(1.00)],
        labels=[_label(EpistemicState.CONFIRMED, 0.90)],
        alpha=1.0,
    )
    assert result.mean_predicted_sparsity == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Multi-example label aggregation
# ---------------------------------------------------------------------------

def test_multiple_labels_averaged_into_target_distribution():
    """The target distribution is the element-wise mean across the batch."""
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[
            _label(EpistemicState.CONFIRMED, 0.90),
            _label(EpistemicState.CONFIRMED, 0.90),
        ],
        alpha=1.0,
    )
    # Two identical labels → target dist same as the single-label case
    single_result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[_label(EpistemicState.CONFIRMED, 0.90)],
        alpha=1.0,
    )
    assert result.calibration_penalty == pytest.approx(single_result.calibration_penalty)


# ---------------------------------------------------------------------------
# Frozen result
# ---------------------------------------------------------------------------

def test_objective_result_is_frozen():
    obj = ConfidenceObjective()
    result = obj.compute(
        task_loss=1.0,
        projection_results=[_result(0.92)],
        labels=[_label(EpistemicState.CONFIRMED, 0.92)],
        alpha=1.0,
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.total_loss = 0.0  # type: ignore[misc]
