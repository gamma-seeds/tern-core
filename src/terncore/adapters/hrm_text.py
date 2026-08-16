"""
HRM-Text dual-timescale recurrent architecture adapter.

Maps Sapient Intelligence's HRM-Text (``HrmTextForCausalLM``,
``model_type="hrm_text"``) HuggingFace weight names to tern-core's
internal conversion schema. HRM-Text is the first **recurrent**
architecture tern-core ingests — it is not a stock transformer.

Architecture (probe 2026-06-13, ``sapientinc/HRM-Text-1B``):
- **Dual-timescale recurrence.** Two independently-parameterised stacks
  share the same block design but own separate weights:
  ``model.H_module.layers.N.*`` (slow / strategic, ``z_H``) and
  ``model.L_module.layers.N.*`` (fast / execution, ``z_L``). Per forward
  pass the **L-stack executes 6×** (2 H-cycles × 3 L-steps) and the
  **H-stack 2×** over the same recurrent state — so the two stacks carry
  identical footprint but different per-forward reuse leverage. The
  conversion is symmetric; :meth:`stack_of` tags each weight H/L for
  per-stack reporting (a reporting aid, not a correctness requirement —
  classification is stack-agnostic).
- **Fused projections in the safetensors.** The on-disk checkpoint stores
  attention as a single ``attn.gqkv_proj`` ``[4*hidden, hidden]`` (gated
  Q/K/V) and the MLP as a single ``mlp.gate_up_proj`` ``[2*intermediate,
  hidden]``; the HF modeling code splits these into ``q/k/v/gate`` and
  ``gate/up`` on load. **tern-core reads the safetensors directly**, so it
  ternarises the *fused* tensors — the conversion sees 128 ternary tensors
  for HRM-Text-1B (64 H + 64 L), not the 256 of the split state_dict. Both
  views cover the same parameters; the fused layout is the ground truth
  for conversion and the source of the layer count in reports.
- **Parameterless MagicNorm.** Block norms carry no weights, so the only
  protected tensors are the embeddings, the (untied) LM head, and the 1-D
  recurrent initialiser ``model.z_L_init`` (named explicitly below for
  clarity; the generic 1-D rule would also retain it).

This is a dense, text-only adapter — no MoE expert stacking
(:meth:`expand_stacked` stays at the base ``None``) and no multimodal
components. Weight classification reuses the standard substring +
dimensionality policy: embeddings / LM head / norms / ``z_L_init`` →
FP16-retain; remaining 2-D block weights → ternary-eligible.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from terncore.adapters import register
from terncore.adapters.base import (
    AdapterInfo,
    ArchitectureAdapter,
    WeightClassification,
)

# Matches the per-block index in both stacks
# (``model.H_module.layers.7.attn.gqkv_proj.weight`` -> 7). The H/L stack
# is *not* captured here — both stacks share ``.layers.N.`` numbering; use
# :meth:`stack_of` to disambiguate the stack.
_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")

# Fused HRM projection names, ordered by ternary tolerance (most-tolerant
# first), for reporting / priority. ``gqkv_proj`` = fused gated Q/K/V,
# ``gate_up_proj`` = fused MLP gate+up.
_PROJ_PRIORITY = [
    "gqkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]

# Protection patterns. ``z_l_init`` is named explicitly (the 1-D recurrent
# state initialiser) even though the 1-D rule would retain it anyway; the
# remaining patterns cover embeddings, the untied LM head, and any norm
# weights a future HRM variant might carry.
_ALWAYS_PROTECTED = (
    "embed_tokens",
    "lm_head",
    "z_l_init",
    "norm",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "classifier",
)


@register("hrm_text")
class HrmTextAdapter(ArchitectureAdapter):
    """Architecture adapter for the recurrent ``HrmTextForCausalLM`` family.

    Weight classification:
    1. Embeddings, LM head, ``z_L_init``, norms → FP16-retain.
    2. 1-D weights (scalars / norms / recurrent init) → FP16-retain.
    3. All 2-D weights in the H/L block stacks (fused ``gqkv_proj`` /
       ``gate_up_proj``, ``o_proj``, ``down_proj``) → ternary-eligible.
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="hrm_text",
            architectures=["HrmTextForCausalLM"],
            model_type="hrm_text",
            description=(
                "HRM-Text dual-timescale recurrent adapter — "
                "HrmTextForCausalLM (Sapient HRM-Text). Two separately-"
                "parameterised H/L block stacks (slow z_H / fast z_L), "
                "fused gqkv_proj + gate_up_proj in the safetensors, "
                "parameterless MagicNorm. Dense, text-only."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=False,
        )

    def normalize_name(self, name: str) -> str:
        return name

    def stack_of(self, name: str) -> Literal["H", "L", "shared"]:
        """Return the recurrence stack a weight belongs to.

        ``"H"`` for the slow/strategic stack (``H_module``), ``"L"`` for
        the fast/execution stack (``L_module``), ``"shared"`` for tensors
        outside both stacks (embeddings, LM head, ``z_L_init``). This is a
        reporting aid — it lets ``--verbose`` / conversion reports tag each
        layer by stack so the asymmetric per-forward reuse (L 6× vs H 2×)
        is visible. Classification itself does not depend on the stack.
        """
        if "H_module" in name:
            return "H"
        if "L_module" in name:
            return "L"
        return "shared"

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        canonical = self.normalize_name(name)
        name_lower = canonical.lower()

        for pattern in _ALWAYS_PROTECTED:
            if pattern in name_lower:
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="fp16_retain",
                    reason=f"Protected pattern: '{pattern}'",
                    component="language",
                )

        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (norm, scalar, or recurrent init)",
                component="language",
            )

        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in H/L block stack",
            component="language",
        )
