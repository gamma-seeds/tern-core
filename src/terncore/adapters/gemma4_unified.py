"""
Gemma 4 Unified (encoder-free) architecture adapter.

Maps Google Gemma 4 *Unified* (``Gemma4UnifiedForConditionalGeneration``,
``model_type="gemma4_unified"``) HuggingFace weight names to tern-core's
internal conversion schema. This is the dense 12B "Unified" topology —
distinct from the encoder-based E4B / MoE variants handled by the base
:class:`~terncore.adapters.gemma4.Gemma4Adapter`.

Key differences from the encoder-based Gemma 4 (E4B / 26B MoE):

- **Encoder-free multimodal.** No ``vision_tower`` / ``audio_tower``
  stacks and no ``multi_modal_projector``. Instead a lightweight inline
  embedder projects raw patches / waveforms directly:
    * ``model.embed_vision.embedding_projection.weight``
    * ``model.embed_audio.embedding_projection.weight``
    * ``model.vision_embedder.{patch_dense,patch_ln*,pos_embedding,pos_norm}``
  All of these are FP16-retained (modality projection is precision-
  sensitive and a tiny fraction of the checkpoint; ~35M params).
- **Dense, not MoE.** The 12B has no expert tensors — ``expert_pattern``
  is ``None``.
- **Interleaved attention, value-free globals.** 48 layers = 40
  ``sliding_attention`` + 8 ``full_attention`` (global) at indices
  5, 11, 17, 23, 29, 35, 41, 47. The 8 global layers carry q/k/o +
  q_norm/k_norm but **no discrete ``v_proj``** (``num_kv_shared_layers``
  is 0 — this is value-free global attention, not KV-sharing). Weight
  classification is per-tensor, so layers with fewer attention tensors
  flow through unchanged; the asymmetry matters to the Axis-B KV pass,
  not to weight conversion.

Text-tower projection names match Llama / the base Gemma 4 adapter
(q/k/v/o + gate/up/down + the gemma4 norms + ``layer_scalar``), so the
base classification and stacked-expert (no-op here) logic are inherited.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from terncore.adapters import register
from terncore.adapters.base import AdapterInfo
from terncore.adapters.gemma4 import (
    _ALWAYS_PROTECTED,
    _BLOCK_PATTERN,
    _PROJ_PRIORITY,
    Gemma4Adapter,
)

# Dense projection priority — drop the MoE expert entries the base adapter
# carries (``gemma4_unified`` 12B is dense; keeping them would be inert but
# misleading). Order preserved by empirical ternary tolerance.
_DENSE_PROJ_PRIORITY = [
    p for p in _PROJ_PRIORITY if p not in ("gate_up_proj",)
]


@register("gemma4_unified")
class Gemma4UnifiedAdapter(Gemma4Adapter):
    """Architecture adapter for Gemma 4 Unified (encoder-free 12B dense).

    Inherits the base Gemma 4 weight-classification rules and extends the
    multimodal component patterns to the inline encoder-free embedder so
    every modality-projection tensor is FP16-retained (never ternarised).
    """

    # Encoder-free embedder. ``embed_vision`` / ``embed_audio`` already
    # match the base patterns; ``vision_embedder.*`` (patch_dense,
    # pos_embedding, ...) is unique to the Unified topology and must be
    # added or its 2-D ``patch_dense.weight`` would fall through to
    # ternary-eligible.
    _VISION_PATTERNS = Gemma4Adapter._VISION_PATTERNS + ["vision_embedder"]
    _AUDIO_PATTERNS = list(Gemma4Adapter._AUDIO_PATTERNS)
    _PROJECTOR_PATTERNS = list(Gemma4Adapter._PROJECTOR_PATTERNS)

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="gemma4_unified",
            architectures=["Gemma4UnifiedForConditionalGeneration"],
            model_type="gemma4_unified",
            description=(
                "Google Gemma 4 Unified adapter — dense 12B, encoder-free "
                "multimodal. FP16-retains the inline vision/audio embedder "
                "(embed_vision / embed_audio / vision_embedder); ternarises "
                "the language_model decoder tower. No MoE; 8 global layers "
                "carry no v_proj (value-free global attention)."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_DENSE_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=True,
            multimodal_components=["vision", "audio"],
            expert_pattern=None,  # dense — no experts
        )
