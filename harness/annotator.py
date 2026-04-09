# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — EpistemicAnnotator³
# weight annotation mechanism (flag to Rod).
"""
EpistemicAnnotator — per-step training analogue of ConfidenceEmitter.

The training-time counterpart to
``tern-runtime/inspector/confidence_emitter.ConfidenceEmitter``.
ConfidenceEmitter classifies a token by its top-1 logit probability;
EpistemicAnnotator classifies a training step by the projection's
sparsity ratio. Both produce values from the same vocabulary
(``EpistemicState`` — ``confirmed`` / ``uncertain`` / ``disconfirmed``)
with byte-identical string values, so a TFH-trained checkpoint's
training-time annotations propagate into LIS inference-time
confidence without conversion.

Why sparsity is the confidence proxy
====================================
Sparsity = (count of zero ternary weights) / (total weights). A
higher sparsity ratio means more weight magnitudes have collapsed
into the deadband — the model has settled on a smaller, more
decisive support set for its current representation. A lower
sparsity ratio means many weights are still active and the model
is hedging across more directions in weight space. The classifier
maps that scalar onto the same three-state vocabulary the
inference-time emitter uses, with the same threshold cut points.

Threshold constants — DO NOT TUNE PER INSTANCE
==============================================
The CONFIRMED and UNCERTAIN thresholds are CLASS CONSTANTS, declared
identically here and in
``tern_runtime.inspector.confidence_emitter.ConfidenceEmitter``:

    CONFIRMED_THRESHOLD = 0.85
    UNCERTAIN_THRESHOLD = 0.45

If these change, change BOTH files in the SAME commit. The
cross-repo trip-wire test ``test_threshold_constants_match_
confidence_emitter`` imports both classes and asserts equality —
if a future contributor changes one without the other, that test
fails before any training run can proceed. Two callers, one
vocabulary, one set of cut points, forever.

Reference: SPEC-TFH-001 § 3.4 + ARCH-LIS-001 § 4.3
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.epistemic_state import EpistemicLabel, EpistemicState
from harness.projector import ProjectionResult


# ---------------------------------------------------------------------------
# Per-step annotation record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepAnnotation:
    """Frozen output of a single ``EpistemicAnnotator.annotate`` call.

    Travels through the training loop into ConfidenceEventLog³ as the
    audit-trail entry for one training step. Frozen so the loop cannot
    mutate an annotation after the calibration loss has read it.
    """

    epistemic_state: EpistemicState   # predicted from the projection
    predicted_score: float            # the sparsity scalar that drove the prediction
    label_state: EpistemicState       # ground truth
    label_score: float                # ground truth confidence
    calibration_error: float          # |predicted - label_score|
    sparsity: float                   # passthrough from result.sparsity
    is_correct: bool                  # predicted_state == label_state


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class EpistemicAnnotator:
    """Maps a ProjectionResult to a StepAnnotation using fixed thresholds.

    Stateless — every ``annotate()`` call is independent. The class
    exists only to namespace the threshold constants and to mirror the
    ``ConfidenceEmitter`` interface on the LIS side. Two annotators in
    different processes are guaranteed to agree because the cut points
    are class constants, not instance state.
    """

    # LOCKED — see module docstring. Same values as
    # tern_runtime.inspector.confidence_emitter.ConfidenceEmitter.
    CONFIRMED_THRESHOLD = 0.85
    UNCERTAIN_THRESHOLD = 0.45

    def annotate(
        self,
        result: ProjectionResult,
        label: EpistemicLabel,
    ) -> StepAnnotation:
        """Classify a single projection step against its ground-truth label.

        Args:
            result: The projector output for the current training step.
                The ``sparsity`` field drives the classification.
            label: The ground-truth epistemic label for the example.
                Provides the target state and confidence score for the
                calibration error computation.

        Returns:
            A frozen ``StepAnnotation`` carrying the predicted state,
            the label state, the calibration error, and the
            agreement flag.
        """
        predicted_score = float(result.sparsity)
        epistemic_state = self._classify(predicted_score)
        label_score = float(label.confidence_score)
        calibration_error = abs(predicted_score - label_score)
        is_correct = epistemic_state == label.epistemic_state

        return StepAnnotation(
            epistemic_state=epistemic_state,
            predicted_score=predicted_score,
            label_state=label.epistemic_state,
            label_score=label_score,
            calibration_error=calibration_error,
            sparsity=float(result.sparsity),
            is_correct=is_correct,
        )

    def batch_annotate(
        self,
        results: list[ProjectionResult],
        labels: list[EpistemicLabel],
    ) -> list[StepAnnotation]:
        """Annotate a batch of (projection, label) pairs.

        Raises ``ValueError`` if the two lists have different lengths —
        a mismatch always indicates a bug upstream and should fail
        loudly rather than silently truncate via ``zip``.
        """
        if len(results) != len(labels):
            raise ValueError(
                f"results and labels must have the same length, "
                f"got {len(results)} results and {len(labels)} labels"
            )
        return [self.annotate(r, l) for r, l in zip(results, labels)]

    def summary(self, annotations: list[StepAnnotation]) -> dict:
        """Aggregate counts and means across a batch of annotations.

        Suitable for ConfidenceEventLog³ entries and harness dashboards.
        Empty input returns zeros throughout (never raises) so logging
        is unconditional.
        """
        if not annotations:
            return {
                "n": 0,
                "mean_calibration_error": 0.0,
                "accuracy": 0.0,
                "confirmed_count": 0,
                "uncertain_count": 0,
                "disconfirmed_count": 0,
                "mean_sparsity": 0.0,
            }
        n = len(annotations)
        confirmed = sum(
            1 for a in annotations if a.epistemic_state == EpistemicState.CONFIRMED
        )
        uncertain = sum(
            1 for a in annotations if a.epistemic_state == EpistemicState.UNCERTAIN
        )
        disconfirmed = sum(
            1 for a in annotations if a.epistemic_state == EpistemicState.DISCONFIRMED
        )
        return {
            "n": n,
            "mean_calibration_error": sum(a.calibration_error for a in annotations) / n,
            "accuracy": sum(1 for a in annotations if a.is_correct) / n,
            "confirmed_count": confirmed,
            "uncertain_count": uncertain,
            "disconfirmed_count": disconfirmed,
            "mean_sparsity": sum(a.sparsity for a in annotations) / n,
        }

    # ----------------------------------------------------------- helpers

    @classmethod
    def _classify(cls, score: float) -> EpistemicState:
        """Map a [0, 1] confidence proxy to an EpistemicState.

        Same boundary semantics as ``ConfidenceEmitter._classify``:
        the comparators are ``>=`` so an exact threshold value maps
        to the upper bucket.
        """
        if score >= cls.CONFIRMED_THRESHOLD:
            return EpistemicState.CONFIRMED
        if score >= cls.UNCERTAIN_THRESHOLD:
            return EpistemicState.UNCERTAIN
        return EpistemicState.DISCONFIRMED
