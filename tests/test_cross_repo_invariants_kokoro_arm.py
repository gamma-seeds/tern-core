"""
Cross-repo trip-wire activation arm for tern-core KokoroProvider3
integration³ attachment (Phase 0 OQ-7 ratification 2026-05-19).

Wave 4 + Wave 5 trip-wire arms transitioning SKIPPED → LIVE at this
attachment:

  TW-2 — EpistemicState string-value parity across integration³ + the
         downstream tern-core consumer. integration³ canonical surface
         lives at ``integration3.thought3.EpistemicState`` (post-Wave 2
         OQ-6 ratification); tern-core consumes it via the
         KokoroProvider3 InferenceResult³ surface.

  TW-9 — Confidence³ three-state enum parity (φ_can three-state enum)
         extends to tern-core consumption at this attachment. Member
         NAMES + string VALUES must match canonical.

This test file is LOCAL to tern-core per Phase 0 brief §5.5 +
Addendum 2 §9 halt-at-gate posture — integration³ at HEAD 67c7f74
is at rest per build-closure stand-down; the upstream
``test_cross_repo_invariants.py`` is NOT edited at this attachment.
A future integration³ housekeeping commit may activate the upstream
TW-N stubs' cross-repo arms post-attachment.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# sys.path routing (worktree src/ + agents/integration3) handled in
# tests/conftest.py — module-identity preservation across the test
# session.

import integration3  # noqa: E402
from integration3 import Confidence3, EpistemicState  # noqa: E402

import terncore.integration3  # noqa: E402
from terncore.integration3 import KokoroProvider3  # noqa: E402


# ── TW-2 — EpistemicState canonical-surface parity ────────────────────────────

def test_tw2_epistemic_state_canonical_member_names_match():
    """TW-2: EpistemicState canonical member NAMES match the patent
    vocabulary mapping per P101 §1.5 (verified / uncertain / invalid
    → CONFIRMED / UNCERTAIN / DISCONFIRMED)."""
    expected = {"CONFIRMED", "UNCERTAIN", "DISCONFIRMED"}
    actual = {m.name for m in EpistemicState}
    assert actual == expected


def test_tw2_epistemic_state_canonical_string_values_match():
    """TW-2: EpistemicState string VALUES match canonical exactly.
    String parity protects cross-repo IPC serialisation surface."""
    assert EpistemicState.CONFIRMED.value == "CONFIRMED"
    assert EpistemicState.UNCERTAIN.value == "UNCERTAIN"
    assert EpistemicState.DISCONFIRMED.value == "DISCONFIRMED"


def test_tw2_epistemic_state_canonical_surface_used_by_kokoro_provider3():
    """TW-2: KokoroProvider3.infer() produces an InferenceResult³
    whose epistemic_state is the integration³ canonical
    EpistemicState type (not a tern-core re-declaration or local
    Protocol stub)."""
    provider = KokoroProvider3()
    result = provider.infer("hello")
    assert isinstance(result.epistemic_state, EpistemicState)
    # The class identity matches the canonical surface (same class
    # object, not a re-declared sibling).
    assert type(result.epistemic_state) is EpistemicState


def test_tw2_epistemic_state_three_cell_count_invariant():
    """TW-2: Exactly three EpistemicState members (closed-set per
    Wave 2 OQ-6 ratification)."""
    assert len(list(EpistemicState)) == 3


# ── TW-9 — Confidence³ three-state enum parity ────────────────────────────────

def test_tw9_confidence3_canonical_member_names_match():
    """TW-9: Confidence³ canonical member NAMES — SURE / UNSURE /
    UNKNOWN per P140 §32-§33 + SEED canonical vocabulary."""
    expected = {"SURE", "UNSURE", "UNKNOWN"}
    actual = {m.name for m in Confidence3}
    assert actual == expected


def test_tw9_confidence3_canonical_string_values_match():
    """TW-9: Confidence³ string VALUES match canonical Unicode glyphs
    exactly (◆ / ◇ / ○)."""
    assert Confidence3.SURE.value == "◆"
    assert Confidence3.UNSURE.value == "◇"
    assert Confidence3.UNKNOWN.value == "○"


def test_tw9_confidence3_canonical_surface_used_by_kokoro_provider3():
    """TW-9: KokoroProvider3.infer() produces an InferenceResult³
    whose input_confidence is the integration³ canonical
    Confidence³ type."""
    provider = KokoroProvider3()
    result = provider.infer("hello")
    assert isinstance(result.input_confidence, Confidence3)
    assert type(result.input_confidence) is Confidence3


def test_tw9_confidence3_three_state_count_invariant():
    """TW-9: Exactly three Confidence³ members (closed-set per
    canonical SEED vocabulary)."""
    assert len(list(Confidence3)) == 3


# ── Cross-tier discipline — two-tier lattice preserved ────────────────────────

def test_two_tier_lattice_confidence_and_epistemic_disjoint():
    """Two-tier discipline per SEED §Core item 2 — Confidence³ +
    EpistemicState are distinct types; never aliased; never merged."""
    assert Confidence3 is not EpistemicState
    confidence_names = {m.name for m in Confidence3}
    epistemic_names = {m.name for m in EpistemicState}
    # Member names from the two tiers do not overlap
    assert confidence_names.isdisjoint(epistemic_names)


def test_two_tier_lattice_carried_simultaneously_on_inference_result3():
    """InferenceResult³ carries both tiers per Wave 5 closure —
    input_confidence (routing-tier) + epistemic_state (inference-tier)
    coexist on every result."""
    provider = KokoroProvider3()
    result = provider.infer("hello")
    assert isinstance(result.input_confidence, Confidence3)
    assert isinstance(result.epistemic_state, EpistemicState)


# ── Integration³ canonical types reachable from tern-core ─────────────────────

def test_integration3_canonical_types_consumable_at_tern_core():
    """All canonical types consumed at this attachment are reachable
    via the integration3 public API surface."""
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
    for cls in (
        AtomIdentifier3, Confidence3, EpistemicState,
        InferenceResult3, Provenance3, Provider3,
        ProviderConfig3, Thought3,
    ):
        assert cls is not None


def test_integration3_canonical_no_tern_core_back_reference():
    """integration³ does not import tern-core — one-way dependency.

    Static scan: no integration³ module references ``terncore`` in
    its imports. Cluster H pattern boundary preserved."""
    integration3_root = Path(integration3.__file__).parent
    for py in integration3_root.rglob("*.py"):
        text = py.read_text()
        # An ``import terncore`` or ``from terncore`` line would
        # invert the dependency — verify absent.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import terncore" not in stripped, (
                f"integration³ imports terncore in {py}: {line!r} — "
                f"dependency inverted (cluster H pattern boundary)"
            )
            assert "from terncore" not in stripped, (
                f"integration³ imports from terncore in {py}: {line!r} — "
                f"dependency inverted (cluster H pattern boundary)"
            )
