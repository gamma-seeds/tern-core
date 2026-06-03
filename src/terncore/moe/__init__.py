"""
Ternary Mixture-of-Experts inference primitives.

Milestone 1 (resident sparse-MoE inference): the experts of an MoE model
are held as addressable, ternary-resident ``PackedTernaryLinear`` modules
in a :class:`PackedTernaryExpertBank` rather than materialised dense into
the transformers fused-experts Parameter. This keeps Qwen3-30B-A3B's
18,432 expert weights at ~7.5 GB packed instead of ~57 GB FP16, fitting
the M4 Pro 64 GB unified-memory envelope, and provides the substrate the
Stage 3 custom MoE block routes through (P145 indexing-vector-conditional
firing; P146 prepare-and-launch dispatch).

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from terncore.moe.expert_bank import (
    MoEPackedModel,
    PackedTernaryExpertBank,
    load_moe_packed,
)
from terncore.moe.runnable import TernaryMoEBlock, build_runnable_qwen3_moe

__all__ = [
    "PackedTernaryExpertBank",
    "MoEPackedModel",
    "load_moe_packed",
    "TernaryMoEBlock",
    "build_runnable_qwen3_moe",
]
