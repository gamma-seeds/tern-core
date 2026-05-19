"""terncore.integration3 — Integration³ Provider³ Protocol satisfaction layer.

Outer-adapter module per integration³ Wave 5 cluster (A) + (B) ratifications
2026-05-19 (Option B scaffolding ratification 2026-05-19): tern-core's
canonical integration³ attachment point lives here as a clean bridge that
does NOT invert dependencies; the existing tern-core adapter cohort
(``src/terncore/adapters/``) stays untouched apart from the additive
``vocoder_pattern`` + ``acoustic_stack_pattern`` ``AdapterInfo`` fields
landing alongside the Kokoro 82M adapter.

Architectural posture per Wave 5 cluster (A) + (B) + Phase 0 brief 2026-05-19:

  - provider3.Provider³ Protocol satisfaction lives at this outer layer
  - Subscriber³ Protocol satisfaction DECLINED (Cluster A — provider3
    is an adapter, not a faculty consumer; SEED §93 "What Integration³
    is NOT" framing)
  - Ledger³ writeback DECLINED (Cluster B — caller-controlled per Wave 5
    cluster B; no decision3 / Ledger3 import across this module tree)
  - EpistemicState three-cell mapping per P099 §107:
      high-confidence    → CONFIRMED
      partial-evidence   → UNCERTAIN
      counter-evidence   → DISCONFIRMED
  - Provenance³ soft-stamp via ``device_signature = f"tern-core@{endpoint}"``
  - AtomIdentifier³ SHA-256-of-fields per Wave 2 OQ-1 canonical identity
    scheme

Module layout:
  - ``base.py``    : ``IntegratedProvider3Base`` — template-method base
                     wrapping a tern-core adapter via DI; concrete
                     subclasses override ``_run_inference`` and inherit
                     all canonical disposition wiring.
  - ``kokoro.py``  : ``KokoroProvider3`` — first canonical Provider3
                     concrete; Kokoro 82M TTS; demo-subset 6 voices.

Public surface (Phase 0 implementation 2026-05-19):

  >>> from terncore.integration3 import (
  ...     IntegratedProvider3Base,
  ...     KokoroProvider3,
  ... )

Inventor: Robert Lakelin
Gamma Seeds Pte Ltd / Synapticode
"""

from terncore.integration3.base import (
    INFERENCE_OUTCOME_CONFIRMED,
    INFERENCE_OUTCOME_DISCONFIRMED,
    INFERENCE_OUTCOME_UNCERTAIN,
    IntegratedProvider3Base,
)
from terncore.integration3.kokoro import (
    DEFAULT_COMPRESSED_ARTEFACT,
    DEFAULT_MODEL_PATH,
    DEFAULT_VOICES_DIR,
    DEMO_VOICES,
    KOKORO_SAMPLE_RATE_HZ,
    KokoroProvider3,
)

__all__ = [
    "IntegratedProvider3Base",
    "KokoroProvider3",
    "DEMO_VOICES",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_VOICES_DIR",
    "DEFAULT_COMPRESSED_ARTEFACT",
    "KOKORO_SAMPLE_RATE_HZ",
    "INFERENCE_OUTCOME_CONFIRMED",
    "INFERENCE_OUTCOME_UNCERTAIN",
    "INFERENCE_OUTCOME_DISCONFIRMED",
]
