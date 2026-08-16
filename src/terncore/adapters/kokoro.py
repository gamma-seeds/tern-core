"""
Kokoro 82M architecture adapter for tern-core conversion pipeline.

Kokoro 82M is a StyleTTS 2-derived multi-speaker text-to-speech model
shipped by hexgrad on HuggingFace as ``hexgrad/Kokoro-82M`` (Apache
2.0). The model is the first downstream consumer integration of
integration³'s Provider³ Protocol attaching tern-core (per surgeon's
dispatch 2026-05-19 following integration³ Wave 5 closure).

Architecture facts (per ``config.json`` at
``~/synapticode/model-library/Kokoro-82M/``):
  - 82M parameters; FP32 form 312 MB (kokoro-v1_0.pth dict-of-dicts)
  - plbert text encoder: 768 hidden / 12 heads / 12 layers
  - main acoustic stack: hidden_dim 512, n_layer 3, dropout 0.2
  - istftnet vocoder: 3 ResBlocks (dilation [1,3,5];
    kernel [3,7,11]); upsample rates [10,6]; FFT 20; hop 5
  - multispeaker: 54 voice embeddings (style_dim 128, .pt files)
  - n_token: 178 (IPA phoneme vocabulary)

Weight tree shape (dict-of-dicts top-level keys):
  - ``bert``         — plbert text encoder (ALBERT-style, 25 tensors)
  - ``bert_encoder`` — bert-to-acoustic projection (2 tensors)
  - ``predictor``    — duration / F0 / N predictor stack (122 tensors)
  - ``decoder``      — ISTFTNet vocoder + AdaIN modulators (375 tensors)
  - ``text_encoder`` — phoneme→features text encoder (24 tensors)

Compression tier policy per Phase 0 OQ ratifications 2026-05-19:

  - **FP16 retained** (sensitive / low-savings) — embeddings + structural
    norms (LayerNorm / gamma / beta) + biases + weight-norm gains
    (weight_g) + 1-D tensors
  - **INT4** (audio-output sensitive) — F0_proj / N_proj small
    projector heads (auxiliary; total <0.01 MB at INT4)
  - **Ternary** (compression-eligible) — all 2-D and 3-D weight tensors
    that don't match the protect / INT4 patterns, including:
      - plbert attention + FFN weights (LLM-style)
      - predictor LSTM weight_ih / weight_hh
      - decoder ISTFTNet ResBlock weight_v (3-D conv tensors)
      - text_encoder cnn weight_v + lstm weights
      - AdaIN .norm1./.norm2. modulator FC weights (functionally FC,
        despite their .norm naming — refined patterns avoid spurious
        FP16 retention here)

Estimated compressed footprint:
  - Ternary: ~19 MB (81.3M params × 2 bits / 8 = 20.3 MB conservative)
  - INT4:    ~0.01 MB
  - FP16:    ~0.81 MB (425K params × 2 bytes; embeddings + 1-D + norms)
  - Total:   ~20.2 MB main-model + ~1.5 MB 6 demo voices FP16 = ~21.7 MB
  - F-8 ceiling: ≤22 MB demo artefact (OQ-6 ratification) — within band

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

# Kokoro 82M ships a torch.save() dict-of-dicts (NOT safetensors / NOT a
# HuggingFace transformers checkpoint). The standard ``layers.N`` block
# pattern does not apply; we use a synthesised pattern that the
# ``classify_weight`` path uses for component detection only.
_BLOCK_PATTERN = re.compile(r"(?:albert_layers|decode|lstms?|cnn)\.(\d+)")

# Projection priority list — informational only for Kokoro since
# routing is by sub-stack pattern rather than projection name.
_PROJ_PRIORITY = [
    "weight_v",       # weight-norm 'main' tensor for ISTFTNet convs
    "weight",          # standard Linear / LSTM weights
    "weight_ih_l0",    # LSTM input-hidden
    "weight_hh_l0",    # LSTM hidden-hidden
    "fc.weight",       # AdaIN modulator FC weights
]

# Always-protected pattern globs (substring match on the FULL composite
# key form ``"<section>/<weight_name>"``).
_ALWAYS_PROTECTED: tuple[str, ...] = (
    # Structural norms (LayerNorm, AdaIN gamma/beta, etc.)
    ".LayerNorm",
    "layer_norm",
    "layernorm",
    # Embeddings — phoneme + position + token-type + style embeddings
    "embedding",
    # Weight-norm gain scalar (per-channel; tiny; sensitive)
    "weight_g",
    # InstanceNorm gain / beta
    ".gamma",
    ".beta",
)

# INT4-routed patterns (audio-output-adjacent sensitive heads).
# Kept small per the v0.6.0+ layer-sensitivity routing precedent.
_INT4_PATTERNS: tuple[str, ...] = (
    "F0_proj",
    "N_proj",
)

# Acoustic-stack pattern (StyleTTS 2 plbert + predictor + text encoder).
# Matches the bert / bert_encoder / predictor / text_encoder roots in
# Kokoro's composite key naming.
_ACOUSTIC_STACK_PATTERN = re.compile(
    r"^(?:bert|bert_encoder|predictor|text_encoder)/"
)

# Vocoder pattern (ISTFTNet decoder).
_VOCODER_PATTERN = re.compile(r"^decoder/")


@register("kokoro")
class KokoroAdapter(ArchitectureAdapter):
    """Architecture adapter for Kokoro 82M (StyleTTS 2 + ISTFTNet).

    Distinctive vs the LLM cohort (llama / gemma3 / gemma4 / phi3 /
    qwen3_moe):

    - **Weight source format** — Kokoro ships ``kokoro-v1_0.pth``
      (torch.save dict-of-dicts) rather than safetensors. The Phase 0
      brief OQ-1 Option A disposition routes weight loading through a
      Kokoro-aware caller (the
      ``terncore.integration3.kokoro.KokoroProvider3`` consumer) that
      reads the .pth and feeds tensors keyed as ``"<section>/<key>"``
      composite names into the per-weight ``classify_weight``.
    - **HF architecture identifier** — Kokoro does not surface an HF
      ``architectures`` field. The adapter declares the synthetic
      identifier ``"KokoroForTTS"`` and the consumer passes that
      identifier explicitly at ``validate_architecture`` time. This
      matches the established conservative-allow-list discipline.
    - **TTS-specific sub-stacks** — ``vocoder_pattern`` matches the
      decoder ISTFTNet sub-stack; ``acoustic_stack_pattern`` matches
      the plbert + predictor + text encoder sub-stacks. Both fields
      are additive Optional on :class:`AdapterInfo` per the Phase 0
      attachment.

    Weight classification rules (per Phase 0 OQ-2 + OQ-3 dispositions):

    1. Embeddings, LayerNorm, gamma/beta, weight_g, biases → FP16-retain.
    2. 1-D tensors (biases / scalars) → FP16-retain.
    3. F0_proj / N_proj small projectors → INT4 (audio-output sensitive).
    4. All other 2-D and 3-D weights (including AdaIN .norm1./.norm2.
       modulator FCs and ISTFTNet ResBlock conv weight_v tensors) →
       ternary-eligible.

    Component tagging (4-bucket vocab on
    :attr:`WeightClassification.component`):
      - ``"audio"``      — decoder / vocoder sub-stack
      - ``"language"``   — plbert / bert_encoder / text_encoder / predictor
      - (vision / projector buckets unused at this attachment)

    The ``"audio"`` tagging surfaces the vocoder weights for downstream
    sensitivity analysis (e.g. ResBlock-vs-upsample routing refinement
    in future fork sessions). Kokoro 82M is text + audio only; no
    vision sub-stack lands at this attachment.
    """

    _AUDIO_PATTERNS: list[str] = [r"^decoder/"]

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="kokoro",
            architectures=["KokoroForTTS"],
            model_type="kokoro_tts",
            description=(
                "Kokoro 82M adapter — StyleTTS 2 acoustic stack + "
                "ISTFTNet vocoder + plbert text encoder; 8 languages / "
                "54 voices; phoneme-in / mel-out (24 kHz); Apache 2.0. "
                "First TTS adapter in the cohort; first downstream "
                "consumer of integration³ Provider³ Protocol."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=True,
            multimodal_components=["audio"],
            vocoder_pattern=_VOCODER_PATTERN,
            acoustic_stack_pattern=_ACOUSTIC_STACK_PATTERN,
        )

    def normalize_name(self, name: str) -> str:
        """Kokoro names arrive as composite ``"<section>/<weight_key>"``
        per the dict-of-dicts loader convention. No normalisation
        applied at this attachment — names propagate through the
        compression pipeline as-given."""
        return name

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        canonical = self.normalize_name(name)
        name_lower = canonical.lower()
        component = self._detect_component(canonical)

        # Embeddings + structural norms + weight_g + 1-D → FP16-retain.
        for pattern in _ALWAYS_PROTECTED:
            if pattern.lower() in name_lower:
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="fp16_retain",
                    reason=f"Protected pattern: '{pattern}'",
                    component=component,
                )

        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component=component,
            )

        # Small auxiliary projector heads (F0_proj, N_proj) — audio-
        # output adjacent. Phase 0 OQ-3 Option B layer-sensitivity
        # routing nominates INT4 for these; tern-core's INT4 path
        # expects 2-D Linear tensors (block-wise quantisation). When
        # the shape is 3-D (conv kernel) we fall back to ternary — the
        # absolute parameter count is negligible (<0.01 MB either way)
        # and ternary stays within the spectral-fidelity envelope per
        # the v0.6.0+ small-projector precedent.
        for pattern in _INT4_PATTERNS:
            if pattern in canonical:
                if shape is not None and len(shape) == 2:
                    return WeightClassification(
                        name=name,
                        canonical_name=canonical,
                        category="int4_eligible",
                        reason=(
                            f"Audio-output-adjacent projector head "
                            f"('{pattern}') — INT4 per OQ-3 layer-"
                            f"sensitivity routing"
                        ),
                        component=component,
                    )
                # 3-D conv kernel form for F0_proj / N_proj — route to
                # ternary (INT4 path requires 2-D).
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="ternary_eligible",
                    reason=(
                        f"Audio-output-adjacent projector head "
                        f"('{pattern}') — 3-D conv kernel routes "
                        f"to ternary (INT4 reserved for 2-D)"
                    ),
                    component=component,
                )

        # Everything else 2-D/3-D → ternary-eligible. The .norm1./.norm2.
        # AdaIN modulators are FC weights despite the ``.norm`` substring
        # — refined patterns above already excluded them from the protect
        # set, so they land here as intended.
        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason=(
                "2-D/3-D weight in acoustic stack or vocoder — "
                "ternary-eligible per Kokoro 82M layer policy"
            ),
            component=component,
        )

    def is_vocoder_weight(self, name: str) -> bool:
        """Return True iff ``name`` matches the ISTFTNet vocoder
        sub-stack pattern.

        Helper for downstream layer-sensitivity routing refinement
        (e.g. ResBlock ternary vs upsample INT4 within the vocoder
        sub-stack per Phase 0 OQ-3 Option B). At this attachment all
        vocoder ternary-eligible tensors share the same ternary policy;
        finer-grained INT4 routing within the vocoder remains a
        downstream refinement.
        """
        pattern = self.info().vocoder_pattern
        if pattern is None:
            return False
        return pattern.search(name) is not None

    def is_acoustic_stack_weight(self, name: str) -> bool:
        """Return True iff ``name`` matches the StyleTTS 2 acoustic
        sub-stack pattern (plbert + bert_encoder + predictor +
        text_encoder)."""
        pattern = self.info().acoustic_stack_pattern
        if pattern is None:
            return False
        return pattern.search(name) is not None
