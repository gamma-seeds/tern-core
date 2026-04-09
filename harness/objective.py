# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — TFH composite loss
# function with epistemic confidence triples as a training objective
# (flag to Rod).
"""
ConfidenceObjective — composite training loss for the TFH.

Composite formula (SPEC-TFH-001 § 3.4):

    total_loss = task_loss
                 + alpha * (calibration_penalty + 0.1 * sparsity_penalty)

Where:
    task_loss            — standard cross-entropy or task-appropriate
                           loss, supplied by the trainer
    calibration_penalty  — KL divergence between predicted and
                           ground-truth epistemic distributions over
                           {confirmed, uncertain, disconfirmed}
    sparsity_penalty     — squared L1 distance between actual and
                           target ternary sparsity ratio
    alpha                — confidence-loss weight from the
                           AdaptationScheduler (0.0 → 1.0 over the
                           training schedule)
    sparsity_target      — fixed target ratio (default 0.90 per
                           harness.yaml ``ternary_projector.sparsity_target``)

Phase 1 gradient note
=====================
The task_loss term is the gradient source for backprop through the
model weights. The calibration and sparsity penalties at this layer
are computed from sparsity scalars that the projector materialises
via ``.item()`` (Python floats), so they do not currently propagate
gradients through MLX. They are diagnostic / scheduling signals
that shape the LOGGED total loss but not yet the WEIGHT updates.

Phase 2 will lift the sparsity penalty into a continuous MLX
expression (e.g. sum of sigmoids around the threshold) so it
contributes to the gradient. The mathematical contract for the
total loss value stays the same — only the gradient flow changes.
This is documented as a known limitation in the trainer's planning
notes; for Phase 1 the harness loop is correct as a forward
computation and as a logging stream, even if the calibration term
is not yet a true gradient signal.

Distribution conversion
=======================
Both the predicted and target distributions are 3-element vectors
over {confirmed, uncertain, disconfirmed}, summing to 1. We convert
a (state, score) pair into a distribution by placing ``score`` at
the named state and splitting ``(1 - score)`` evenly across the
other two states. KL divergence is then computed in the standard
direction: ``KL(target || predicted)``.

This is intentionally simple. SPEC-TFH-001 § 9 lists ``calibration_loss``
as one of {``kl_divergence``, ``focal``, ``brier``} — KL is the default
and what this Phase 1 implementation provides. Swapping in focal or
brier is a single-method override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harness.annotator import EpistemicAnnotator
from harness.epistemic_state import EpistemicLabel, EpistemicState
from harness.projector import ProjectionResult


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveResult:
    """Frozen output of a single ``ConfidenceObjective.compute`` call.

    All four numbers are Python floats so they serialise straight into
    ConfidenceEventLog³ entries and harness dashboards. The
    ``total_loss`` is the same scalar that the trainer would log; the
    Phase 1 gradient note in the module docstring explains why
    backprop currently flows through ``task_loss`` only.
    """

    total_loss: float
    task_loss: float
    calibration_penalty: float
    sparsity_penalty: float
    alpha_used: float
    mean_predicted_sparsity: float


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

class ConfidenceObjective:
    """Composite loss for the TFH training step.

    Stateless except for two configuration constants:
        sparsity_target          — default 0.90 (harness.yaml)
        SPARSITY_PENALTY_WEIGHT  — default 0.10 (SPEC-TFH-001 § 3.4)

    The 0.10 weight on the sparsity penalty is locked at the class
    level because it is part of the published loss formula in the
    spec. The sparsity TARGET is per-instance because it is the
    documented config knob in harness.yaml.
    """

    SPARSITY_PENALTY_WEIGHT = 0.10
    DEFAULT_SPARSITY_TARGET = 0.90

    # Small floor used inside log() to avoid log(0). The KL divergence
    # is unbounded above when q has zero mass on a state where p has
    # mass — clamping prevents NaNs without distorting the loss
    # signal at non-degenerate distributions.
    _PROB_FLOOR = 1e-12

    def __init__(self, sparsity_target: float = DEFAULT_SPARSITY_TARGET) -> None:
        if not 0.0 < sparsity_target <= 1.0:
            raise ValueError(
                f"sparsity_target must be in (0.0, 1.0], got {sparsity_target}"
            )
        self.sparsity_target = sparsity_target

    # ----------------------------------------------------------- compute

    def compute(
        self,
        task_loss: float,
        projection_results: list[ProjectionResult],
        labels: list[EpistemicLabel],
        alpha: float,
    ) -> ObjectiveResult:
        """Compute the composite loss for one training step.

        Args:
            task_loss: Scalar task-loss value (cross-entropy etc.) for
                the batch. Python float — the trainer extracts it from
                the MLX scalar before calling. Phase 2 will accept an
                mx.array directly to keep the gradient path clean.
            projection_results: One ProjectionResult per layer the
                projector ran on for this batch. Aggregated by mean
                sparsity to produce a single per-step model-confidence
                proxy.
            labels: Ground-truth epistemic labels — one per example
                in the batch. The target distribution is the mean of
                each example's per-state distribution.
            alpha: Current scheduler value for the confidence-loss
                weight. Multiplied into both the calibration penalty
                and the sparsity penalty.

        Returns:
            ``ObjectiveResult`` with task_loss, calibration_penalty,
            sparsity_penalty, alpha_used, total_loss, and the mean
            predicted sparsity (for logging).

        Raises:
            ValueError: empty inputs, alpha < 0, or shape mismatches.
        """
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not projection_results:
            raise ValueError("projection_results must be non-empty")
        if not labels:
            raise ValueError("labels must be non-empty")

        # Aggregate projection results: one mean sparsity per step
        mean_sparsity = sum(r.sparsity for r in projection_results) / len(projection_results)

        # Calibration penalty: KL(target || predicted)
        target_dist = self._mean_label_distribution(labels)
        predicted_dist = self._predicted_distribution(mean_sparsity)
        calibration_penalty = self._kl_divergence(target_dist, predicted_dist)

        # Sparsity penalty: squared shortfall from target. Zero when
        # actual sparsity >= target — we never penalise excess sparsity.
        shortfall = max(0.0, self.sparsity_target - mean_sparsity)
        sparsity_penalty = shortfall ** 2

        # Composite
        total_loss = task_loss + alpha * (
            calibration_penalty + self.SPARSITY_PENALTY_WEIGHT * sparsity_penalty
        )

        return ObjectiveResult(
            total_loss=float(total_loss),
            task_loss=float(task_loss),
            calibration_penalty=float(calibration_penalty),
            sparsity_penalty=float(sparsity_penalty),
            alpha_used=float(alpha),
            mean_predicted_sparsity=float(mean_sparsity),
        )

    # ----------------------------------------------------------- distributions

    @classmethod
    def _predicted_distribution(cls, sparsity: float) -> tuple[float, float, float]:
        """Convert the per-step mean sparsity into a 3-element distribution.

        The annotator's classification gives us the predicted state;
        the sparsity scalar gives us the confidence at that state.
        We place ``sparsity`` at the predicted state and split
        ``(1 - sparsity)`` evenly across the other two states.

        Returns:
            ``(p_confirmed, p_uncertain, p_disconfirmed)`` summing to 1.0.
        """
        predicted_state = EpistemicAnnotator._classify(sparsity)
        return cls._state_to_distribution(predicted_state, sparsity)

    @classmethod
    def _mean_label_distribution(
        cls, labels: list[EpistemicLabel]
    ) -> tuple[float, float, float]:
        """Mean per-example distribution across the batch.

        Each label is converted to its own 3-element distribution
        (confidence_score at the named state, the rest split evenly),
        then averaged across the batch element-wise.
        """
        n = len(labels)
        sums = [0.0, 0.0, 0.0]
        for label in labels:
            dist = cls._state_to_distribution(label.epistemic_state, label.confidence_score)
            sums[0] += dist[0]
            sums[1] += dist[1]
            sums[2] += dist[2]
        return (sums[0] / n, sums[1] / n, sums[2] / n)

    @staticmethod
    def _state_to_distribution(
        state: EpistemicState, score: float
    ) -> tuple[float, float, float]:
        """One (state, score) pair → ``(p_conf, p_unc, p_disc)``.

        Places ``score`` at the named state. Splits ``(1 - score)``
        evenly across the other two states. The result is always a
        valid probability distribution: every entry in [0, 1], total
        sums to exactly 1.0 (modulo floating-point noise that the
        callers tolerate).
        """
        score = max(0.0, min(1.0, score))
        rest = (1.0 - score) / 2.0
        if state == EpistemicState.CONFIRMED:
            return (score, rest, rest)
        if state == EpistemicState.UNCERTAIN:
            return (rest, score, rest)
        return (rest, rest, score)

    # ----------------------------------------------------------- KL

    @classmethod
    def _kl_divergence(
        cls,
        target: tuple[float, float, float],
        predicted: tuple[float, float, float],
    ) -> float:
        """KL(target || predicted) in nats.

        Standard formula: ``sum(p * log(p / q))`` where ``p`` is target
        and ``q`` is predicted. Both inputs are clamped above
        ``_PROB_FLOOR`` before division and log to avoid NaNs at
        degenerate distributions; the floor is small enough that the
        result is unaffected for non-degenerate inputs.

        Returns 0 if every target component is below the floor (the
        sum collapses to zero), and grows monotonically as the
        distributions diverge.
        """
        kl = 0.0
        for p, q in zip(target, predicted):
            if p <= cls._PROB_FLOOR:
                continue
            q_safe = max(q, cls._PROB_FLOOR)
            kl += p * math.log(p / q_safe)
        return kl
