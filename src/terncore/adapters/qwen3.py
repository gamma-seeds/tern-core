"""
Dense Qwen3 architecture adapter for tern-core conversion pipeline.

Maps Alibaba Qwen3 dense (``Qwen3ForCausalLM``) HuggingFace weight
names to tern-core's internal weight schema. Targets the dense Qwen3
family including the PrismML Ternary Bonsai checkpoints (1.7B / 8B),
which are ``Qwen3ForCausalLM`` (Q1 recon 2026-05-31).

Qwen3 dense vs the Llama family it otherwise mirrors:
- **Per-head QK-Norm.** Each attention block carries
  ``self_attn.q_norm.weight`` and ``self_attn.k_norm.weight`` — per-head
  RMSNorm on the query and key projections (shape ``[head_dim]``). These
  are RMSNorm weights and stay FP16: they match the protected ``"norm"``
  substring *and* are 1-D, so they classify as ``fp16_retain`` by two
  independent rules. Llama has no analog.
- **Tied embeddings on the smaller checkpoint.** Bonsai 1.7B sets
  ``tie_word_embeddings: true`` (the embedding doubles as the LM head, so
  the safetensors carries no separate ``lm_head.weight``); 8B is untied.
  Classification needs no special case — ``embed_tokens`` is protected
  whether or not a separate ``lm_head`` is present, and an absent tensor
  is simply never classified. Topology-level tie handling lives in
  ``terncore.arch_presets`` (the ``ArchPreset.tie_word_embeddings`` flag).
- **No attention/MLP bias.** Qwen3 sets ``attention_bias: false`` /
  ``no_bias: true``; there are no bias tensors to classify.

This is the *dense* adapter. The sparse ``Qwen3MoeForCausalLM`` variant
(per-expert FFN, router as ``mlp.gate.weight``) is served by
``terncore.adapters.qwen3_moe`` and must not route here.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import re
from typing import Optional

from terncore.adapters import register
from terncore.adapters.base import (
    AdapterInfo,
    ArchitectureAdapter,
    WeightClassification,
)

_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")

_PROJ_PRIORITY = [
    "v_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Protection patterns. ``"norm"`` covers the standard block norms
# (input_layernorm, post_attention_layernorm, model.norm) and the
# Qwen3 per-head QK-Norm weights (self_attn.q_norm / self_attn.k_norm)
# in one stroke — both carry the ``norm`` substring. The QK-Norm
# tensors are additionally 1-D, so they are doubly protected.
_ALWAYS_PROTECTED = (
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "classifier",
)


@register("qwen3")
class Qwen3Adapter(ArchitectureAdapter):
    """Architecture adapter for the dense ``Qwen3ForCausalLM`` family.

    Weight classification:
    1. Embeddings, norms (including per-head ``q_norm`` / ``k_norm``),
       LM head → FP16-retain.
    2. 1-D weights (norms, scalars) → FP16-retain.
    3. All 2-D weights in transformer blocks (Q/K/V/O, gate/up/down)
       → ternary-eligible.
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3",
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            description=(
                "Dense Qwen3 adapter — Qwen3ForCausalLM family including "
                "PrismML Ternary Bonsai 1.7B/8B. GQA + SwiGLU + RMSNorm "
                "with per-head QK-Norm (q_norm/k_norm protected FP16). "
                "Distinct from the sparse qwen3_moe adapter. Text-only."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=False,
        )

    def normalize_name(self, name: str) -> str:
        return name

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
                reason="1-D tensor (norm or scalar parameter)",
                component="language",
            )

        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block",
            component="language",
        )
