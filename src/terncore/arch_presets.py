"""
Declarative architecture presets resolved from a HuggingFace ``config.json``.

This module is the topology layer that sits between two existing
tern-core surfaces:

* ``terncore.coreml_export.ARCH_PRESETS`` — declarative topology dicts
  keyed by *model name* and hand-selected via ``--arch-preset``. Rich
  in topology, yet selected by operator choice rather than read from a
  model's own config.
* ``terncore.adapters`` — resolved by *architecture string* and rich in
  weight-name classification, yet carrying no topology (head counts,
  norm scheme, RoPE).

An :class:`ArchPreset` unifies both axes: it is a declarative topology
record (mirroring the ``ARCH_PRESETS`` field shape) that resolves
directly from a model's ``config.json`` by reading ``architectures`` and
``model_type``. A Qwen3 ``config.json`` resolves to the qwen3 preset; a
Llama ``config.json`` continues to resolve to the llama preset.

Scope (Phase 1): the preset declares topology — including QK-Norm
presence and the YaRN RoPE fields — as data. The runtime operators for
QK-Norm and YaRN land in Phase 2; the preset's boolean / config fields
are the declarative hooks those ops will read, exactly as
``coreml_export`` already gates ``has_qk_norm`` / ``activation`` off its
preset dicts today.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union


class ArchPresetMismatch(Exception):
    """Raised when a ``config.json`` declares an architecture that no
    registered preset builder covers, or when the architecture cannot
    be read from the config.

    The message names the offending architecture and lists the
    registered set so the operator can either route to a different
    preset or register the architecture if it is genuinely
    topology-compatible.
    """


# ---------------------------------------------------------------------------
# RoPE configuration — declarative, covers default and YaRN-scaled forms.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RopeConfig:
    """RoPE configuration carried declaratively on a preset.

    ``rope_type`` is ``"default"`` for plain RoPE and ``"yarn"`` for
    YaRN context extension. When YaRN is active, ``factor`` and
    ``original_max_position_embeddings`` carry the scaling parameters
    the Phase 2 runtime will apply. ``theta`` is the RoPE base
    frequency in every case.
    """

    theta: float
    rope_type: str = "default"
    factor: Optional[float] = None
    original_max_position_embeddings: Optional[int] = None

    @property
    def is_yarn(self) -> bool:
        """True when YaRN context extension is configured."""
        return self.rope_type == "yarn"

    @classmethod
    def from_config(cls, config: dict) -> "RopeConfig":
        """Build from a HF config dict.

        Reads ``rope_theta`` and the optional ``rope_scaling`` block.
        A ``rope_scaling`` of ``{"rope_type": "yarn", "factor": ...,
        "original_max_position_embeddings": ...}`` yields a YaRN
        config; its absence yields a default RoPE config.
        """
        theta = float(config.get("rope_theta", 10000.0))
        scaling = config.get("rope_scaling")
        if not scaling:
            return cls(theta=theta)
        # HF uses "rope_type" (newer) or "type" (older) inside the block.
        rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
        factor = scaling.get("factor")
        original = scaling.get("original_max_position_embeddings")
        return cls(
            theta=theta,
            rope_type=str(rope_type),
            factor=float(factor) if factor is not None else None,
            original_max_position_embeddings=(
                int(original) if original is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# ArchPreset — declarative topology record.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchPreset:
    """Declarative topology for one decoder-only transformer family.

    Mirrors the field shape of ``coreml_export.ARCH_PRESETS`` and
    extends it with the structured :class:`RopeConfig` (covering YaRN)
    and explicit norm / MLP scheme tags. Every field is data — no
    runtime operator lives here. Phase 2 reads :attr:`has_qk_norm` and
    :attr:`rope` to wire the QK-Norm and YaRN ops.
    """

    name: str
    model_type: str
    architectures: tuple[str, ...]

    # Core dimensions
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int

    # Norm + MLP + activation scheme
    rms_norm_eps: float
    norm_type: str  # "rmsnorm"
    mlp_type: str  # "swiglu"
    activation: str  # "silu" | "gelu"

    # Architectural feature flags (declarative; ops land Phase 2)
    has_qk_norm: bool
    tie_word_embeddings: bool

    # RoPE (default or YaRN)
    rope: RopeConfig

    @property
    def gqa_groups(self) -> int:
        """Grouped-query-attention ratio = query heads per KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        """True when KV heads are fewer than query heads (GQA active)."""
        return self.num_key_value_heads < self.num_attention_heads


# ---------------------------------------------------------------------------
# Shared field extraction.
# ---------------------------------------------------------------------------
def _common_dims(config: dict) -> dict:
    """Extract the dimension fields shared across decoder families.

    Uses HF canonical key names. ``head_dim`` falls back to
    ``hidden_size // num_attention_heads`` when the config omits it
    (older Llama configs), matching the convention in
    ``coreml_export``.
    """
    hidden = int(config["hidden_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv = int(config.get("num_key_value_heads", n_heads))
    head_dim = int(config.get("head_dim") or (hidden // n_heads))
    return {
        "num_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden,
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "vocab_size": int(config["vocab_size"]),
        "max_position_embeddings": int(
            config.get("max_position_embeddings", 0)
        ),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
        "rope": RopeConfig.from_config(config),
    }


# ---------------------------------------------------------------------------
# Per-family preset builders.
# ---------------------------------------------------------------------------
def _build_llama(config: dict) -> ArchPreset:
    """Llama / Mistral family: RMSNorm + SwiGLU, no QK-Norm.

    Covers the ``LlamaForCausalLM`` / ``MistralForCausalLM`` shapes
    that the existing llama adapter and ``ARCH_PRESETS`` Llama entries
    describe. QK-Norm is absent in this family.
    """
    return ArchPreset(
        name="llama",
        model_type="llama",
        architectures=("LlamaForCausalLM", "MistralForCausalLM"),
        norm_type="rmsnorm",
        mlp_type="swiglu",
        activation=str(config.get("hidden_act", "silu")),
        has_qk_norm=False,
        **_common_dims(config),
    )


def _build_qwen3(config: dict) -> ArchPreset:
    """Dense Qwen3 family (``Qwen3ForCausalLM``): RMSNorm + SwiGLU + QK-Norm.

    Qwen3 carries per-head RMSNorm on the query and key projections
    (``self_attn.q_norm`` / ``self_attn.k_norm``) — an intrinsic
    architectural feature, so ``has_qk_norm`` is declared True for the
    family. RoPE is read from the config and is YaRN-scaled for the
    long-context Bonsai checkpoints. Tied embeddings vary per
    checkpoint (1.7B ties, 8B does not) and are read from the config.

    This is the *dense* Qwen3 preset; the sparse ``Qwen3MoeForCausalLM``
    variant is served by ``terncore.adapters.qwen3_moe`` and resolves
    through its own path.
    """
    return ArchPreset(
        name="qwen3",
        model_type="qwen3",
        architectures=("Qwen3ForCausalLM",),
        norm_type="rmsnorm",
        mlp_type="swiglu",
        activation=str(config.get("hidden_act", "silu")),
        has_qk_norm=True,
        **_common_dims(config),
    )


# Registry: HF architecture string -> builder. Architecture-string match
# is the primary key (it is the authoritative discriminator in a HF
# config); ``model_type`` provides a secondary fallback below.
_ARCH_BUILDERS: dict[str, Callable[[dict], ArchPreset]] = {
    "LlamaForCausalLM": _build_llama,
    "MistralForCausalLM": _build_llama,
    "Qwen3ForCausalLM": _build_qwen3,
}

# Secondary fallback: model_type -> builder, for configs whose
# ``architectures`` list is absent or unrecognised but whose
# ``model_type`` is known.
_MODEL_TYPE_BUILDERS: dict[str, Callable[[dict], ArchPreset]] = {
    "llama": _build_llama,
    "mistral": _build_llama,
    "qwen3": _build_qwen3,
}


def registered_architectures() -> list[str]:
    """Return the HF architecture strings with a registered builder."""
    return sorted(_ARCH_BUILDERS)


# ---------------------------------------------------------------------------
# Resolution entry points.
# ---------------------------------------------------------------------------
def resolve_preset(config: dict) -> ArchPreset:
    """Resolve a HuggingFace ``config.json`` dict to an :class:`ArchPreset`.

    Dispatch reads ``architectures[0]`` first (the authoritative
    discriminator), then falls back to ``model_type``. A config that
    matches neither raises :class:`ArchPresetMismatch` naming the
    architecture and the registered set.

    Args:
        config: Parsed ``config.json`` as a dict.

    Returns:
        The declarative :class:`ArchPreset` for the model's family,
        populated from the config's own fields.
    """
    architectures = config.get("architectures") or []
    for arch in architectures:
        builder = _ARCH_BUILDERS.get(arch)
        if builder is not None:
            return builder(config)

    model_type = config.get("model_type")
    if model_type is not None:
        builder = _MODEL_TYPE_BUILDERS.get(str(model_type).lower())
        if builder is not None:
            return builder(config)

    raise ArchPresetMismatch(
        f"No registered preset for architectures={architectures!r} "
        f"model_type={config.get('model_type')!r}. Registered "
        f"architectures: {registered_architectures()}. Either route to "
        f"a different preset or register this architecture if it is "
        f"genuinely topology-compatible."
    )


def resolve_preset_from_dir(model_dir: Union[str, Path]) -> ArchPreset:
    """Read ``config.json`` from a model directory and resolve its preset.

    Mirrors the ``config.json`` reading convention in
    ``terncore.convert`` and raises :class:`ArchPresetMismatch` when
    the config is missing, unreadable, or unrecognised — so a single
    exception type covers the whole resolution path.

    Args:
        model_dir: Directory containing the model's ``config.json``.

    Returns:
        The resolved :class:`ArchPreset`.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise ArchPresetMismatch(
            f"config.json not found at {config_path}. Cannot resolve "
            f"an architecture preset for {model_dir}."
        )
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as e:
        raise ArchPresetMismatch(
            f"Failed to read config.json at {config_path}: {e}. "
            f"Cannot resolve an architecture preset."
        ) from e
    return resolve_preset(config)


__all__ = [
    "ArchPreset",
    "RopeConfig",
    "ArchPresetMismatch",
    "resolve_preset",
    "resolve_preset_from_dir",
    "registered_architectures",
]
