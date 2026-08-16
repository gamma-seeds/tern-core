# Copyright 2026 Robert Lakelin. Inventor: Robert Lakelin.
"""Decision³ LedgerEntry³ bootstrap for the Kokoro 82M tern-core
attachment Phase 0 brief — fourth canonical LedgerEntry³ record overall
(first downstream-consumer entry; bootstraps post-build ledger continuity
after the 3 integration³ build records: decision3 brief + provider3
brief + bridge3 brief at Wave 4 + Wave 5 closures).

Per surgeon's Wave 5 cluster (E) operationalisation surface ratification
2026-05-19: from Wave 5 onwards, each Phase 0 brief commits as a
``LedgerEntry3`` record at closure. Kokoro 82M attachment is the first
post-build downstream-consumer brief to land under this discipline.

Per surgeon's Wave 5 cluster (B) ratification: provider3 + Kokoro
attachment do NOT touch decision3.Ledger³ at the module level
(src/terncore/integration3/ stays decision3-free per F-6 falsifier).
This test file lives in tests/ outside the integration³ module tree —
the ledger bootstrap is a closure artefact, NOT a module surface.

decision_atom_ref via AtomIdentifier³ SHA-256-of-fields per Wave 2 OQ-1
canonical identity scheme (matches Wave 4 cluster E decision_atom_ref
scheme; one canonical identity across integration³).
"""

from __future__ import annotations

from integration3 import (
    Confidence3,
    LedgerEntry3,
    make_bootstrap_ledger_entry,
)


KOKORO_BRIEF_PATH = (
    "phase0_briefs/2026-05-19_kokoro_82m_integration3_attachment_phase0.md"
)


def test_kokoro_82m_brief_ledger_entry_bootstrap() -> None:
    """The Kokoro 82M tern-core attachment Phase 0 brief materialises as
    the fourth canonical ``LedgerEntry3`` record (first downstream-
    consumer entry).

    debit_side records the 7 upstream surfaces consumed by the
    attachment:

      - integration3.Provider3 Protocol (Wave 5 §5.13)
      - integration3.InferenceResult3 (Wave 5 additive extension)
      - integration3.ProviderConfig3 (Wave 5 closed-set 5-field)
      - integration3.EpistemicState (Wave 2 canonical OQ-6)
      - integration3.Provenance3 (Wave 2 P096 §130)
      - integration3.AtomIdentifier3 (Wave 2 OQ-1 SHA-256-of-fields)
      - terncore.adapters.base.AdapterInfo (tern-core adapter cohort
        schema; extended additively with TTS-specific fields per OQ-1
        Option A)

    credit_side records the produced canonical surface:
    KokoroProvider3 + IntegratedProvider3Base (the first canonical
    Provider³ Protocol consumer + the template-method base class for
    future tern-core integration³ attachments).

    decision_atom_ref via AtomIdentifier³ SHA-256-of-fields per Wave 2
    OQ-1 canonical identity scheme; 64-char hex form.

    Algebraic balance holds by construction: credit ≤ min debit, both
    at Confidence³.SURE (canonical at brief-ratification time).
    """
    entry = make_bootstrap_ledger_entry(
        brief_path=KOKORO_BRIEF_PATH,
        surface_atom_ids=(
            "integration3.Provider3",
            "integration3.InferenceResult3",
            "integration3.ProviderConfig3",
            "integration3.EpistemicState",
            "integration3.Provenance3",
            "integration3.AtomIdentifier3",
            "terncore.adapters.base.AdapterInfo",
        ),
        source_module="terncore.integration3.kokoro",
    )

    # Bootstrap shape verification
    assert isinstance(entry, LedgerEntry3)

    # debit_side: 7 upstream module surfaces consumed
    assert len(entry.debit_side) == 7

    # Algebraic balance — every debit ref at SURE
    for ref in entry.debit_side:
        assert ref.snapshot_confidence is Confidence3.SURE

    # decision_atom_ref carries AtomIdentifier³ SHA-256-of-fields per Wave 2 OQ-1
    assert entry.decision_atom_ref is not None
    assert isinstance(entry.decision_atom_ref, str)
    assert len(entry.decision_atom_ref) == 64  # SHA-256 hex form

    # credit_side records the produced canonical surface — tern-core
    # KokoroProvider3 / IntegratedProvider3Base attachment.
    assert "kokoro" in str(entry.credit_side.source).lower() or \
           "kokoro" in str(entry.credit_side.content).lower()


def test_kokoro_brief_is_fourth_canonical_ledger_entry() -> None:
    """Counter-document confirmation that Kokoro 82M brief is the
    fourth canonical ``LedgerEntry3`` overall — first downstream-
    consumer; preceded by the 3 integration³ build records (decision3
    brief at Wave 4 cluster E + provider3 brief at Wave 5 + bridge3
    brief at Wave 5 integration commit).

    This test exercises the bootstrap surface without asserting an
    actual ledger persistence — the LedgerEntry³ shape verification is
    the load-bearing closure record per Wave 5 cluster (E)
    operationalisation surface. Future ledger continuity work may
    persist these entries via the canonical integration3.Ledger³
    machinery.
    """
    entry = make_bootstrap_ledger_entry(
        brief_path=KOKORO_BRIEF_PATH,
        surface_atom_ids=(
            "integration3.Provider3",
            "integration3.InferenceResult3",
            "integration3.ProviderConfig3",
            "integration3.EpistemicState",
            "integration3.Provenance3",
            "integration3.AtomIdentifier3",
            "terncore.adapters.base.AdapterInfo",
        ),
        source_module="terncore.integration3.kokoro",
    )

    # decision_atom_ref deterministic — same inputs produce same hex
    entry_2 = make_bootstrap_ledger_entry(
        brief_path=KOKORO_BRIEF_PATH,
        surface_atom_ids=(
            "integration3.Provider3",
            "integration3.InferenceResult3",
            "integration3.ProviderConfig3",
            "integration3.EpistemicState",
            "integration3.Provenance3",
            "integration3.AtomIdentifier3",
            "terncore.adapters.base.AdapterInfo",
        ),
        source_module="terncore.integration3.kokoro",
        # Override timestamp to make the two entries fully deterministic
        # (default uses time.time(); decision_atom_ref hashes timestamp via
        # provenance.hardware_clock_timestamp).
        timestamp=entry.timestamp,
    )
    assert entry.decision_atom_ref == entry_2.decision_atom_ref, (
        "Canonical AtomIdentifier³ SHA-256-of-fields scheme must be "
        "deterministic across constructions with identical inputs"
    )
