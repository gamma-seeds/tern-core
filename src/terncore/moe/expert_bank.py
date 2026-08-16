"""
Ternary MoE expert bank and packed-MoE loader — Milestone 1 Stage 1.

Per-expert-sliced ``.tern-model`` manifests (Qwen3-30B-A3B,
Gemma-4-26B-A4B) store experts as separate ternary tensors
(``model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight``). The
transformers 5.5+ runtime, however, exposes MoE experts as *fused stacked
Parameters* (``Qwen3MoeExperts.gate_up_proj`` shape
``[num_experts, 2*moe_intermediate, hidden]`` and ``down_proj`` shape
``[num_experts, hidden, moe_intermediate]``) with no per-expert submodule.
Loading the manifest into that fused Parameter would require dense
materialisation (~57 GB FP16 for Qwen3-30B-A3B) — busting the M4 Pro
64 GB ceiling and discarding the ternary memory win.

:func:`load_moe_packed` instead routes per-expert weights into a
:class:`PackedTernaryExpertBank`, addressable by ``(layer, expert, proj)``,
keeping experts ternary-resident (~7.5 GB). Attention projections load as
``PackedTernaryLinear`` (ternary); norms, router, embeddings and LM head
load as protected dense tensors. The bank is the substrate the Stage 3
custom MoE block routes through:

    router top-k  →  Index³ indexing vector  →  bank.get(layer, expert, proj)
                  →  ternary matmul (Metal kernel, PR #9)  →  gate-weighted combine

mapping P145's multi-controlled-operation / indexing-vector-conditional
firing pattern onto MoE inference, dispatched across the P146
prepare-and-launch boundary.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Manifest entry name patterns (Qwen3 / Gemma-4 MoE convention).
_EXPERT_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.weight$"
)
_ATTENTION_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.self_attn\.(?P<qkvo>q|k|v|o)_proj\.weight$"
)
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

_PROJECTIONS = ("gate", "up", "down")


def _layer_of(name: str) -> Optional[int]:
    """Return the transformer block index in ``name``, or None if global."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _module_bytes(module: nn.Module) -> int:
    """Total bytes held by a module's parameters + buffers."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


class PackedTernaryExpertBank(nn.Module):
    """Addressable, ternary-resident store of MoE expert weights.

    Experts are keyed by ``(layer_idx, expert_idx, projection)`` where
    ``projection`` is one of ``"gate"``, ``"up"``, ``"down"``. Each entry
    is a :class:`~terncore.packed_linear.PackedTernaryLinear` holding 2-bit
    packed weights (no dense materialisation). Backed by an
    ``nn.ModuleDict`` so the whole bank moves with ``.to(device)`` and is
    visible to ``state_dict`` / parameter introspection.

    The Stage 3 custom MoE block consumes :meth:`get` per selected expert;
    Milestone 2 attaches the SURE/UNSURE/UNKNOWN lifecycle on top.
    """

    def __init__(
        self,
        num_experts: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        # nn.ModuleDict keys may not contain '.', so encode the tuple key.
        self._experts = nn.ModuleDict()
        self.num_experts = num_experts
        self.num_layers = num_layers

    @staticmethod
    def _key(layer: int, expert: int, proj: str) -> str:
        if proj not in _PROJECTIONS:
            raise ValueError(
                f"projection must be one of {_PROJECTIONS}, got {proj!r}"
            )
        return f"L{layer}_E{expert}_{proj}"

    def add(self, layer: int, expert: int, proj: str, module: nn.Module) -> None:
        key = self._key(layer, expert, proj)
        if key in self._experts:
            raise ValueError(
                f"Duplicate expert entry for (layer={layer}, expert={expert}, "
                f"proj={proj!r}) — manifest contains the same expert twice."
            )
        self._experts[key] = module

    def get(self, layer: int, expert: int, proj: str) -> nn.Module:
        return self._experts[self._key(layer, expert, proj)]

    def has(self, layer: int, expert: int, proj: str) -> bool:
        return self._key(layer, expert, proj) in self._experts

    def __len__(self) -> int:
        return len(self._experts)

    def layers(self) -> List[int]:
        seen = set()
        for key in self._experts:
            seen.add(int(key.split("_", 1)[0][1:]))  # "L{n}"
        return sorted(seen)

    def experts_in_layer(self, layer: int) -> List[int]:
        seen = set()
        prefix = f"L{layer}_E"
        for key in self._experts:
            if key.startswith(prefix):
                # key = L{layer}_E{expert}_{proj}
                seen.add(int(key.split("_E", 1)[1].split("_", 1)[0]))
        return sorted(seen)

    def nbytes(self) -> int:
        return sum(_module_bytes(m) for m in self._experts.values())


@dataclass
class MoEPackedModel:
    """Components of a packed MoE model, loaded from a ``.tern-model``.

    Stage 1 deliverable — the addressable building blocks. Stage 3 assembles
    these into a runnable model: the protected tensors and attention modules
    populate a Qwen3 skeleton, and a custom ``Qwen3MoeSparseMoeBlock``
    replacement routes through :attr:`bank`.

    Attributes:
        bank:       per-expert ternary weights, addressable by
                    ``(layer, expert, proj)``.
        attention:  per-(layer, "q"|"k"|"v"|"o") ternary projections.
        protected:  dense tensors keyed by manifest name (norms, router
                    ``mlp.gate.weight``, ``embed_tokens``, ``lm_head``,
                    q/k norms). Reconstructed from FP16/INT4 entries.
        metadata:   inferred shapes/counts (hidden, moe_intermediate, …).
        coverage:   per-category routing counts + verification results.
    """

    bank: PackedTernaryExpertBank
    attention: Dict[Tuple[int, str], nn.Module]
    protected: Dict[str, torch.Tensor]
    metadata: Dict[str, object] = field(default_factory=dict)
    coverage: Dict[str, object] = field(default_factory=dict)

    def nbytes(self) -> int:
        b = self.bank.nbytes()
        b += sum(_module_bytes(m) for m in self.attention.values())
        b += sum(t.numel() * t.element_size() for t in self.protected.values())
        return b

    def summary_str(self) -> str:
        c = self.coverage
        gb = self.nbytes() / (1024 ** 3)
        lines = [
            "MoEPackedModel:",
            f"  experts (bank):   {c.get('expert_entries', 0):>6}  "
            f"({len(self.bank.layers())} layers)",
            f"  attention:        {c.get('attention_entries', 0):>6}",
            f"  protected (dense):{c.get('protected_entries', 0):>6}",
            f"  skipped (limit):  {c.get('skipped_entries', 0):>6}",
            f"  total routed:     {c.get('routed_entries', 0):>6} / "
            f"{c.get('total_entries', 0)}",
            f"  resident size:    {gb:.2f} GB",
            f"  hidden={self.metadata.get('hidden')}  "
            f"moe_intermediate={self.metadata.get('moe_intermediate')}  "
            f"num_experts={self.metadata.get('num_experts')}",
        ]
        return "\n".join(lines)


def load_moe_packed(
    reader,
    *,
    limit_layers: Optional[int] = None,
    spot_check_n: int = 4,
    verbose: bool = False,
):
    """Load a per-expert-sliced MoE ``.tern-model`` into a packed structure.

    Routes manifest entries by name/dtype:

    - ``...mlp.experts.{E}.{gate,up,down}_proj.weight`` (ternary2) → bank
    - ``...self_attn.{q,k,v,o}_proj.weight`` (ternary2) → attention
    - everything else (FP16 norms/router/embeddings/LM head, INT4) →
      protected dense tensors

    Experts and attention stay packed (``PackedTernaryLinear``); only the
    protected entries materialise dense. The 57 GB FP16 base is never
    loaded — the manifest covers 100% of parameters.

    Args:
        reader:        a ``TernModelReader`` over the artefact.
        limit_layers:  if set, only load transformer blocks with index
                       ``< limit_layers`` (global entries always load).
                       Bounds memory/time for smoke runs.
        spot_check_n:  number of expert entries to verify by independent
                       dense reconstruction (shape + finite alpha + sparsity
                       within tolerance of the manifest).
        verbose:       print a progress heartbeat every 2000 entries.

    Returns:
        :class:`MoEPackedModel`.

    Raises:
        ValueError: if a ternary entry matches neither the expert nor the
            attention pattern (unexpected for a known MoE architecture), or
            if no per-expert entries are present (not an MoE manifest).
    """
    layers = reader.manifest["layers"]
    if not any(_EXPERT_RE.search(e["name"]) for e in layers):
        raise ValueError(
            "No per-expert entries ('...mlp.experts.N....') found in manifest "
            "— this does not look like a per-expert-sliced MoE artefact. "
            "Use TernModelReader.load_packed_model for dense models."
        )

    bank = PackedTernaryExpertBank()
    attention: Dict[Tuple[int, str], nn.Module] = {}
    protected: Dict[str, torch.Tensor] = {}

    expert_n = attn_n = protected_n = skipped_n = 0
    max_layer = -1
    max_expert = -1
    hidden = moe_intermediate = None
    t0 = time.perf_counter()

    for i, entry in enumerate(layers):
        name = entry["name"]
        dtype = entry["dtype"]
        layer = _layer_of(name)
        if limit_layers is not None and layer is not None and layer >= limit_layers:
            skipped_n += 1
            continue

        m = _EXPERT_RE.search(name)
        if m:
            if dtype != "ternary2":
                raise ValueError(
                    f"Expert weight {name!r} expected ternary2, got {dtype!r}."
                )
            L = int(m.group("layer"))
            E = int(m.group("expert"))
            proj = m.group("proj")
            bank.add(L, E, proj, reader.build_packed_linear(name))
            expert_n += 1
            max_layer = max(max_layer, L)
            max_expert = max(max_expert, E)
            # Infer shapes from a gate/up (out=moe_intermediate, in=hidden).
            if proj in ("gate", "up") and hidden is None:
                moe_intermediate, hidden = entry["shape"][0], entry["shape"][1]
            continue

        a = _ATTENTION_RE.search(name)
        if a and dtype == "ternary2":
            L = int(a.group("layer"))
            attention[(L, a.group("qkvo"))] = reader.build_packed_linear(name)
            attn_n += 1
            max_layer = max(max_layer, L)
            continue

        if dtype == "ternary2":
            raise ValueError(
                f"Ternary entry {name!r} matched neither the expert nor the "
                f"attention pattern. Unexpected for a known MoE architecture "
                f"— extend load_moe_packed's routing if this is intended."
            )

        # Protected: FP16 (norms, router, embeddings, LM head) or INT4.
        recon = reader.reconstruct_layer(name)
        protected[name] = recon["weight"]
        if "bias" in recon:
            protected[f"{name}::bias"] = recon["bias"]
        protected_n += 1

        if verbose and (i % 2000 == 0) and i:
            print(
                f"  …{i}/{len(layers)} entries "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

    bank.num_experts = (max_expert + 1) if max_expert >= 0 else None
    bank.num_layers = (max_layer + 1) if max_layer >= 0 else None

    # Independent fidelity spot-check on a spread of expert entries.
    spot_checks: List[dict] = []
    if spot_check_n > 0:
        expert_names = [
            e["name"] for e in layers
            if _EXPERT_RE.search(e["name"])
            and not (
                limit_layers is not None
                and (_layer_of(e["name"]) or 0) >= limit_layers
            )
        ]
        if expert_names:
            step = max(1, len(expert_names) // spot_check_n)
            for name in expert_names[:: step][:spot_check_n]:
                entry = reader._get_manifest_entry(name)
                dense = reader.reconstruct_layer(name)["weight"]
                sparsity = float((dense == 0).float().mean())
                spot_checks.append(
                    {
                        "name": name,
                        "shape_ok": list(dense.shape) == list(entry["shape"]),
                        "finite": bool(torch.isfinite(dense).all()),
                        "alpha": entry.get("alpha"),
                        "sparsity_observed": round(sparsity, 4),
                        "sparsity_manifest": round(entry.get("sparsity", -1), 4),
                        "sparsity_within_tol": abs(
                            sparsity - entry.get("sparsity", -1)
                        ) < 0.02,
                    }
                )

    coverage = {
        "total_entries": len(layers),
        "expert_entries": expert_n,
        "attention_entries": attn_n,
        "protected_entries": protected_n,
        "skipped_entries": skipped_n,
        "routed_entries": expert_n + attn_n + protected_n,
        "load_seconds": round(time.perf_counter() - t0, 1),
        "spot_checks": spot_checks,
    }
    metadata = {
        "hidden": hidden,
        "moe_intermediate": moe_intermediate,
        "num_experts": bank.num_experts,
        "num_layers": bank.num_layers,
        "source": reader.manifest.get("model_metadata", {}).get("source"),
        "adapter": reader.manifest.get("model_metadata", {}).get("adapter"),
    }

    return MoEPackedModel(
        bank=bank,
        attention=attention,
        protected=protected,
        metadata=metadata,
        coverage=coverage,
    )
