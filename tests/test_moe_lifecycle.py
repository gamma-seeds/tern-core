"""
Tests for the Expert Lifecycle Component (Milestone 1 Stage 4, SURE-only).

Pure unit tests over a hand-built bank — no model assembly, no archive.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from terncore.moe import (
    ConfidenceState,
    ExpertLifecycle,
    PackedTernaryExpertBank,
)


def _bank_with(layers_experts):
    bank = PackedTernaryExpertBank()
    for layer, expert in layers_experts:
        for proj in ("gate", "up", "down"):
            bank.add(layer, expert, proj, nn.Linear(2, 2))
    return bank


def test_confidence_state_vocabulary():
    assert {s.value for s in ConfidenceState} == {"SURE", "UNSURE", "UNKNOWN"}


def test_resident_experts_are_sure():
    lc = ExpertLifecycle(_bank_with([(0, 0), (0, 1), (1, 0)]))
    assert lc.state(0, 0) is ConfidenceState.SURE
    assert lc.state(1, 0) is ConfidenceState.SURE
    assert lc.is_resident(0, 1)


def test_absent_expert_is_unknown():
    lc = ExpertLifecycle(_bank_with([(0, 0)]))
    assert lc.state(0, 7) is ConfidenceState.UNKNOWN
    assert not lc.is_resident(0, 7)


def test_prepare_is_noop_for_resident():
    lc = ExpertLifecycle(_bank_with([(0, 0)]))
    assert lc.prepare(0, 0) is ConfidenceState.SURE


def test_prepare_refuses_non_resident_in_m1():
    """M1 does not page; preparing an absent expert is a loud KeyError
    (demand-paging of UNSURE/UNKNOWN is Milestone 2)."""
    lc = ExpertLifecycle(_bank_with([(0, 0)]))
    with pytest.raises(KeyError, match="Milestone 2"):
        lc.prepare(0, 9)


def test_summary_all_sure_in_m1():
    lc = ExpertLifecycle(_bank_with([(0, 0), (0, 1), (1, 0), (1, 1)]))
    s = lc.summary()
    assert s["resident_experts"] == 4
    assert s["states"] == {"SURE": 4, "UNSURE": 0, "UNKNOWN": 0}
