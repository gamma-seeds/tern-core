"""
Expert lifecycle — Milestone 1 Stage 4 (SURE-only).

The :class:`ExpertLifecycle` sits *beside* the
:class:`~terncore.moe.expert_bank.PackedTernaryExpertBank` (not inside the
scheduler or the model registry) and is the attach point for P145's
confidence-stratified controlled-operation pattern, mapped onto expert
memory residency:

    SURE     resident in unified memory — fire immediately
    UNSURE   manifest-known, pageable on selection      (Milestone 2)
    UNKNOWN  cold on disk, not yet touched              (Milestone 2)

In Milestone 1 every expert the bank holds is SURE; :meth:`prepare` (the
P146 "prepare" phase) is therefore a no-op that simply asserts residency.
This class deliberately implements **neither** the UNSURE/UNKNOWN
transitions nor a demand-pager — those are Milestone 2, where a subclass
overrides :meth:`prepare` to page an expert in (UNKNOWN/UNSURE → SURE) and
adds cache/eviction transitions (SURE → UNSURE). The interface is shaped now
so M2 extends rather than replaces it.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard import cycle / heavy import at module load
    from terncore.moe.expert_bank import PackedTernaryExpertBank


class ConfidenceState(enum.Enum):
    """Canonical Synapticode confidence states applied to expert residency."""

    SURE = "SURE"
    UNSURE = "UNSURE"
    UNKNOWN = "UNKNOWN"


class ExpertLifecycle:
    """Per-(layer, expert) confidence-state view over an expert bank.

    Addressable by ``(layer, expert)`` — the residency unit is the expert
    (its gate/up/down projections move together), not the individual
    projection.
    """

    def __init__(self, bank: "PackedTernaryExpertBank") -> None:
        self._bank = bank

    def state(self, layer: int, expert: int) -> ConfidenceState:
        """Confidence state of an expert.

        Milestone 1: SURE if the bank holds the expert (resident), else
        UNKNOWN (informational — there is no pager to make it SURE yet).
        """
        resident = self._bank.has(layer, expert, "gate")
        return ConfidenceState.SURE if resident else ConfidenceState.UNKNOWN

    def is_resident(self, layer: int, expert: int) -> bool:
        return self.state(layer, expert) is ConfidenceState.SURE

    def prepare(self, layer: int, expert: int) -> ConfidenceState:
        """P146 "prepare" phase for an expert about to fire.

        Milestone 1: experts are resident (SURE), so this asserts residency
        and returns SURE — a no-op. Milestone 2 overrides this to transition
        UNKNOWN/UNSURE → SURE by paging the expert in from the fast store.
        """
        st = self.state(layer, expert)
        if st is not ConfidenceState.SURE:
            raise KeyError(
                f"expert (layer={layer}, expert={expert}) is {st.value}, not "
                f"resident. Milestone 1 holds every expert SURE; demand-paging "
                f"of UNSURE/UNKNOWN experts is Milestone 2 (override prepare)."
            )
        return st

    def summary(self) -> dict:
        """Counts by state. Milestone 1: all resident experts are SURE."""
        n_experts = len(self._bank) // 3 if len(self._bank) else 0
        return {
            "resident_experts": n_experts,
            "states": {"SURE": n_experts, "UNSURE": 0, "UNKNOWN": 0},
        }
