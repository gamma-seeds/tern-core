"""base.py — IntegratedProvider3Base canonical Provider³ Protocol satisfaction.

Concrete template-method base class implementing the canonical
disposition wiring (EpistemicState mapping + Provenance³ stamp +
AtomIdentifier³ identity) shared across every tern-core integration³
attachment. Concrete subclasses (``KokoroProvider3`` at this attachment;
future MLXProvider3 / direct-on-device adapters; future
MockProvider³-for-tests) inherit the canonical wiring exactly once.

Architectural posture per Phase 0 brief ratification 2026-05-19:

  - Wraps a tern-core adapter via dependency injection — the adapter
    stays clean of integration³ knowledge (one-way relationship;
    pattern_cross_module_composition_v1 outside-package Protocol-rebind
    posture). The base wraps; the adapter is wrapped.
  - Implements integration³ ``provider3.Provider3`` ``@runtime_checkable``
    Protocol structurally (config / health_check / is_available /
    infer). Subclasses satisfy the same Protocol by inheritance.
  - Stamps ``Provenance3`` with ``device_signature = f"tern-core@{endpoint}"``
    soft-stamp pattern (Cluster D ratification; P096 §130 vocabulary —
    future MLX-native / Jetson adapters with actual SPE access
    override).
  - Computes ``AtomIdentifier3`` via SHA-256-of-fields (Cluster D
    ratification; P096 §140 — matches Wave 4 cluster E
    ``decision_atom_ref`` scheme; one canonical identity scheme across
    integration³).
  - Maps inference outcomes to three-cell ``EpistemicState`` per
    P099 §107 (Cluster C ratification):
        non-empty result + no error           → CONFIRMED
        empty / fallback / partial-evidence   → UNCERTAIN
        exception / counter-evidence          → DISCONFIRMED
  - Subscriber³ Protocol satisfaction **DECLINED** (Cluster A
    ratification — adapters do not satisfy Subscriber³; F-5 falsifier
    enforces no ``on_event`` method, no Subscriber³ import).
  - Ledger³ writeback **DECLINED** (Cluster B ratification — caller-
    controlled; F-6 falsifier enforces no decision3 / Ledger3 import
    across this module tree).

Template-method shape:

  ``IntegratedProvider3Base.infer(...)`` runs the canonical pipeline:

      1. Validate input (optional context atoms)
      2. Propagate Confidence³ weakest-link from context atoms
      3. Dispatch to subclass hook ``_run_inference(prompt, context_str)``
         → returns ``(output_content, outcome)`` where ``outcome`` is
         one of ``"confirmed"`` / ``"uncertain"`` / ``"disconfirmed"``
      4. Stamp Provenance³ + compute AtomIdentifier³ + map
         EpistemicState
      5. Build the output Thought³ + InferenceResult³

  Subclasses override ``_run_inference`` (load model, run forward pass,
  serialise output). Everything else stays canonical.

Inventor: Robert Lakelin
Gamma Seeds Pte Ltd / Synapticode
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

# integration³ canonical surfaces — Wave 2 + Wave 5 canonical types per
# Type Index §5.1 + §5.13. The import path follows the agents/integration3/
# layout convention (top-level ``integration3`` package). Resolution
# discipline: prefer the installed package; fall back to the canonical
# agents/ checkout layout. The dependency stays one-way (tern-core
# consumes integration³; integration³ never imports tern-core).
try:
    from integration3 import (
        AtomIdentifier3,
        Confidence3,
        EpistemicState,
        InferenceResult3,
        Provenance3,
        Provider3,
        ProviderConfig3,
        Thought3,
    )
    from integration3.predication3 import weakest
except ImportError:  # pragma: no cover — fallback path for agents/ checkout
    _AGENTS_PATH = Path("/Users/syn/synapticode/agents/integration3")
    if _AGENTS_PATH.exists() and str(_AGENTS_PATH) not in sys.path:
        sys.path.insert(0, str(_AGENTS_PATH))
    from integration3 import (
        AtomIdentifier3,
        Confidence3,
        EpistemicState,
        InferenceResult3,
        Provenance3,
        Provider3,
        ProviderConfig3,
        Thought3,
    )
    from integration3.predication3 import weakest


# Outcome literal returned by subclass ``_run_inference`` hook.
# Three-cell mapping per Cluster C ratification (P099 §107):
INFERENCE_OUTCOME_CONFIRMED = "confirmed"
INFERENCE_OUTCOME_UNCERTAIN = "uncertain"
INFERENCE_OUTCOME_DISCONFIRMED = "disconfirmed"

_OUTCOME_TO_EPISTEMIC: dict[str, EpistemicState] = {
    INFERENCE_OUTCOME_CONFIRMED: EpistemicState.CONFIRMED,
    INFERENCE_OUTCOME_UNCERTAIN: EpistemicState.UNCERTAIN,
    INFERENCE_OUTCOME_DISCONFIRMED: EpistemicState.DISCONFIRMED,
}


@runtime_checkable
class _AdapterRunnerProtocol(Protocol):
    """Structural shape for a tern-core inference runner that the base
    class wraps via dependency injection.

    Concrete subclasses bind a runner (typically a closure or a class
    method that loads the compressed artefact and runs forward pass).
    The runner is invoked from ``_run_inference``.

    The Protocol is module-local — concrete adapters do not need to
    declare it explicitly; duck-typing via the ``infer_audio`` /
    ``run`` callable surface suffices.
    """

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        ...


class IntegratedProvider3Base:
    """Concrete template-method base for tern-core Provider³ adapters.

    Subclasses bind a tern-core adapter via dependency injection (the
    ``adapter`` constructor parameter) and override ``_run_inference``
    to implement the model-specific forward pass. The base class
    handles all canonical disposition wiring exactly once.

    Construction:

        provider = IntegratedProvider3Base(
            config=ProviderConfig3(endpoint="...", model="..."),
            adapter=<tern-core adapter instance>,
        )

    Provider³ Protocol satisfaction:

        isinstance(provider, integration3.Provider3) is True

    Subclass contract:

        Override ``_run_inference(self, prompt, context_str)`` to
        return a tuple ``(output_content, outcome_token)`` where
        ``outcome_token`` is one of the module-level
        ``INFERENCE_OUTCOME_*`` constants. The base class handles
        EpistemicState mapping, Provenance³ stamping,
        AtomIdentifier³ computation, and InferenceResult³ assembly.

    Cluster A + B architectural rules enforced:

      - No ``on_event`` method (Subscriber³ Protocol NOT satisfied;
        F-5).
      - No decision3 / Ledger3 import anywhere in this module tree
        (F-6 verifies via grep).
    """

    #: Author identity token surfaced on stamped Provenance³ records.
    #: Subclasses override to surface their specific identity.
    AUTHOR_IDENTITY_TOKEN: str = "terncore.integration3.IntegratedProvider3Base"

    def __init__(
        self,
        config: ProviderConfig3,
        adapter: Any = None,
    ) -> None:
        """Construct a Provider³-satisfying tern-core attachment.

        Args:
            config:  ``integration3.ProviderConfig3`` carrying endpoint /
                     model / timeout / output_half_life / faculty_id
                     fields. Concrete subclasses may re-interpret the
                     ``endpoint`` field (e.g. as a filesystem path
                     to a compressed artefact rather than an HTTP URL).
            adapter: Optional tern-core adapter dependency. Subclasses
                     bind their own adapter at construction; the base
                     stores it for subclass access via ``self._adapter``.
        """
        self._config: ProviderConfig3 = config
        self._adapter: Any = adapter
        self._available: Optional[bool] = None

    # ── Provider³ Protocol surface ────────────────────────────────────────────

    @property
    def config(self) -> ProviderConfig3:
        """The ``ProviderConfig3`` bound at construction time."""
        return self._config

    def health_check(self) -> bool:
        """Default lazy probe — subclasses override to perform real
        checks (e.g. confirm compressed artefact exists; verify voice
        embeddings load).

        Default implementation returns the subclass-set ``_available``
        flag if present; ``True`` otherwise. Subclasses override for
        real readiness probes.
        """
        if self._available is None:
            # No subclass-specific probe ran yet — assume available
            # (lazy posture per Phase 0 OQ-5 default disposition).
            self._available = True
        return self._available

    @property
    def is_available(self) -> bool:
        """Cached availability flag from the most recent ``health_check``
        invocation. Triggers a probe if no cached state."""
        if self._available is None:
            return self.health_check()
        return self._available

    def infer(
        self,
        prompt: str,
        context_atoms: Optional[list[Thought3]] = None,
        source: str = "tern-core-integration3",
    ) -> InferenceResult3:
        """Run inference with ternary-confidence-weighted context.

        Template method orchestrating the canonical pipeline:

          1. Confidence³ weakest-link propagation from context atoms.
          2. Subclass hook ``_run_inference`` dispatched.
          3. EpistemicState three-cell mapping from outcome token.
          4. Provenance³ soft-stamp at completion timestamp.
          5. AtomIdentifier³ SHA-256-of-fields deterministic identity.
          6. Output Thought³ + InferenceResult³ assembly.

        Concrete subclasses (KokoroProvider3 + future siblings) only
        override ``_run_inference`` — every disposition cluster (C / D)
        lands here exactly once.
        """
        context_atoms = context_atoms or []
        now = time.time()

        # ── Step 1: weakest-link Confidence³ propagation ─────────────────
        states = [a.confidence(now) for a in context_atoms]
        input_conf = weakest(*states) if states else Confidence3.SURE

        context_str = self._build_context(context_atoms, now)

        # ── Step 2: subclass hook dispatch with exception capture ────────
        t0 = time.time()
        try:
            output_content, outcome_token = self._run_inference(
                prompt=prompt,
                context_str=context_str,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Counter-evidence path → DISCONFIRMED per Cluster C.
            output_content = f"[inference error: {exc}]"
            outcome_token = INFERENCE_OUTCOME_DISCONFIRMED
            input_conf = Confidence3.UNKNOWN

        latency_ms = (time.time() - t0) * 1000.0
        completion_ts = time.time()

        # ── Step 3: EpistemicState three-cell mapping ────────────────────
        epistemic_state = _OUTCOME_TO_EPISTEMIC.get(
            outcome_token,
            EpistemicState.UNCERTAIN,
        )

        # ── Step 4: Provenance³ soft-stamp at completion ─────────────────
        provenance = self._stamp_provenance(
            output_content=output_content,
            completion_ts=completion_ts,
        )

        # ── Step 5: Output Thought³ with weakest-link confidence ─────────
        output_atom = Thought3(
            content=output_content,
            source=f"{source}/{self._config.model}",
            atom_id=str(uuid.uuid4()),
            timestamp=completion_ts,
            sure_half_life=self._config.output_half_life,
            unsure_half_life=self._config.output_half_life * 2.0,
            base_confidence=input_conf,
            provenance=provenance,
        )

        # ── Step 6: AtomIdentifier³ SHA-256-of-fields identity ───────────
        atom_identifier = AtomIdentifier3.compute(
            value=output_content,
            confidence=input_conf,
            provenance=provenance,
        )

        return InferenceResult3(
            text=self._render_text(output_content),
            model=self._config.model,
            output_atom=output_atom,
            input_confidence=input_conf,
            latency_ms=latency_ms,
            endpoint=self._config.endpoint,
            epistemic_state=epistemic_state,
            provenance=provenance,
            atom_identifier=atom_identifier,
        )

    # ── Subclass hook (override required) ─────────────────────────────────────

    def _run_inference(
        self,
        prompt: str,
        context_str: str,
    ) -> tuple[Any, str]:
        """Subclass hook — run the model-specific forward pass.

        Concrete subclasses (KokoroProvider3 + future siblings)
        override this to dispatch to their underlying tern-core
        adapter + compressed artefact.

        Returns:
            ``(output_content, outcome_token)`` where ``outcome_token``
            is one of:
              - ``INFERENCE_OUTCOME_CONFIRMED``    high-confidence result
              - ``INFERENCE_OUTCOME_UNCERTAIN``    partial / fallback
              - ``INFERENCE_OUTCOME_DISCONFIRMED`` counter-evidence

            ``output_content`` may be any type (str for LLM; numpy
            ndarray for TTS audio; dict for structured outputs). The
            base class renders it to a text representation only for
            the ``InferenceResult3.text`` field via ``_render_text``.

        Default raises NotImplementedError — base class is template;
        subclass binding is required for actual use.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _run_inference"
        )

    # ── Optional subclass hook ────────────────────────────────────────────────

    def _render_text(self, output_content: Any) -> str:
        """Render the output content to its ``InferenceResult3.text``
        representation. Default: ``str(output_content)``.

        TTS subclasses (e.g. KokoroProvider3) override to surface a
        descriptive summary (sample rate / duration / voice) rather
        than the raw audio bytes.
        """
        return str(output_content)

    # ── Provenance³ stamp helper (Cluster D ratification) ─────────────────────

    def _stamp_provenance(
        self,
        output_content: Any,
        completion_ts: float,
    ) -> Provenance3:
        """Stamp the canonical soft-stamp ``Provenance3`` on the output.

        Soft-stamp pattern per P096 §130 + Cluster D ratification:
        ``device_signature = f"tern-core@{endpoint}"`` (future MLX-
        native or Jetson SPE-equipped adapters with actual hardware
        attestation override this).
        """
        return Provenance3(
            device_signature=f"tern-core@{self._config.endpoint}",
            author_identity_token=self.AUTHOR_IDENTITY_TOKEN,
            hardware_clock_timestamp=completion_ts,
            content_hash=Provenance3.compute_content_hash(output_content),
            parent_atom_hash=None,
        )

    # ── Context building helper ───────────────────────────────────────────────

    def _build_context(
        self,
        atoms: list[Thought3],
        now: float,
    ) -> str:
        """Render context atoms with their ternary markers.

        Mirrors the ``provider3.TernCoreProvider3._build_context``
        rendering convention so concrete subclasses can produce
        equivalent context strings if they expose a text-conditioned
        model.
        """
        lines: list[str] = []
        for atom in atoms:
            if atom.is_tombstone:
                continue
            _, conf = atom.value(now)
            marker = conf.symbol()
            content_str = str(atom.content)[:512]
            lines.append(f"{marker} [{atom.source}] {content_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"endpoint={self._config.endpoint!r} "
            f"model={self._config.model!r})"
        )


__all__ = [
    "IntegratedProvider3Base",
    "INFERENCE_OUTCOME_CONFIRMED",
    "INFERENCE_OUTCOME_UNCERTAIN",
    "INFERENCE_OUTCOME_DISCONFIRMED",
]
