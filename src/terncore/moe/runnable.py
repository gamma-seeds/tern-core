"""
Runnable ternary MoE — Milestone 1 Stage 2+3 (merged custom block).

:class:`TernaryMoEBlock` is a drop-in replacement for transformers'
``Qwen3MoeSparseMoeBlock`` that routes through a
:class:`~terncore.moe.expert_bank.PackedTernaryExpertBank` instead of a
fused dense experts Parameter. It is the reduction-to-practice of P145 for
MoE inference:

    router (mlp.gate.weight)  →  softmax → top-k  →  Index³ indexing vector
        →  bank.get(layer, expert, proj)   [P146 "prepare": resident SURE
                                            lookup in Milestone 1]
        →  ternary packed matmul           [P146 "launch"]
        →  gate-weighted scatter-combine

Only the top-k experts named by the indexing vector fire per token —
P145's indexing-vector-conditional firing / multi-controlled-operation
pattern. The router and per-expert FFN math mirror
``Qwen3MoeTopKRouter`` / ``Qwen3MoeExperts`` exactly (softmax-then-top-k,
optional top-k renorm; ``down(act(gate(x)) * up(x))``), so a packed model
is numerically equivalent to the dense-ternary reference.

:func:`build_runnable_qwen3_moe` assembles a runnable ``Qwen3MoeForCausalLM``
from a :class:`~terncore.moe.expert_bank.MoEPackedModel`: a meta-device
skeleton (no 57 GB expert allocation) with attention as
``PackedTernaryLinear``, norms/router/embeddings/LM head materialised from
the protected tensors, and each MoE block swapped for a
:class:`TernaryMoEBlock` over the shared bank.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from terncore.moe.expert_bank import MoEPackedModel, PackedTernaryExpertBank
from terncore.moe.lifecycle import ExpertLifecycle

_ATTN_PROJ = ("q", "k", "v", "o")


class TernaryMoEBlock(nn.Module):
    """Ternary, bank-backed replacement for ``Qwen3MoeSparseMoeBlock``.

    Holds its own router weight (``mlp.gate.weight``) and a *reference* to
    the shared :class:`PackedTernaryExpertBank` (wrapped in a tuple so it is
    not re-registered as a submodule per layer — the bank is registered
    once on the top-level model and moves with ``.to(device)``).
    """

    def __init__(
        self,
        bank: PackedTernaryExpertBank,
        layer_idx: int,
        router_weight: torch.Tensor,
        *,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        hidden_act: str,
        lifecycle: "ExpertLifecycle | None" = None,
    ) -> None:
        super().__init__()
        # Tuple wrapper: shared bank reference, deliberately unregistered.
        self._bank_ref = (bank,)
        # Lifecycle (plain object) is not an nn.Module → not registered.
        self._lifecycle = lifecycle
        self.layer_idx = int(layer_idx)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)

        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[hidden_act]
        # Router weight [num_experts, hidden]; inference-only.
        self.gate_weight = nn.Parameter(
            router_weight.detach().clone(), requires_grad=False
        )

    @property
    def bank(self) -> PackedTernaryExpertBank:
        return self._bank_ref[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq, hidden = hidden_states.shape
        x = hidden_states.reshape(-1, hidden)  # [N, H]

        # ── Router (P145 indexing vector): softmax → top-k → renorm ──
        router_logits = F.linear(x, self.gate_weight)  # [N, E]
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)  # [N, k]
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        top_val = top_val.to(x.dtype)

        # ── Conditional firing: only experts named by the indexing vector ──
        out = torch.zeros_like(x)
        bank = self.bank
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_idx, num_classes=self.num_experts
            ).permute(2, 1, 0)  # [E, k, N]
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for e in expert_hit:
            e = int(e[0])
            slot, token_idx = torch.where(expert_mask[e])
            cur = x[token_idx]  # [M, H]
            # P146 "prepare": ensure the expert is resident. In Milestone 1
            # this is a SURE no-op; Milestone 2's lifecycle pages it in.
            if self._lifecycle is not None:
                self._lifecycle.prepare(self.layer_idx, e)
            # P146 "launch": resident bank lookup + ternary packed matmul.
            gate = bank.get(self.layer_idx, e, "gate")
            up = bank.get(self.layer_idx, e, "up")
            down = bank.get(self.layer_idx, e, "down")
            # P146 "launch": ternary packed matmul; FFN mirrors Qwen3MoeExperts.
            h = self.act_fn(gate(cur)) * up(cur)
            o = down(h)  # [M, H]
            o = o * top_val[token_idx, slot, None]
            out.index_add_(0, token_idx, o.to(out.dtype))

        return out.reshape(bsz, seq, hidden)


def _assign_param(module: nn.Module, attr: str, tensor: torch.Tensor) -> None:
    """Replace a (possibly meta) parameter with a real, frozen tensor."""
    setattr(
        module,
        attr,
        nn.Parameter(tensor.detach().clone(), requires_grad=False),
    )


def build_runnable_qwen3_moe(
    packed: MoEPackedModel,
    config,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Assemble a runnable ``Qwen3MoeForCausalLM`` from a packed MoE model.

    Builds the skeleton on the ``meta`` device (the fused experts Parameter
    is never allocated → no ~57 GB cost), then:

    - swaps each MoE ``mlp`` for a :class:`TernaryMoEBlock` over ``packed.bank``,
    - installs attention ``q/k/v/o_proj`` as ``PackedTernaryLinear``,
    - materialises norms, router, embeddings and LM head from ``packed.protected``,
    - re-instantiates the rotary embedding with real ``inv_freq``.

    Raises ``RuntimeError`` if any parameter/buffer is left on ``meta`` after
    assembly (loud over a silent garbage forward).
    """
    from transformers import Qwen3MoeForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeRotaryEmbedding,
        Qwen3MoeSparseMoeBlock,
    )

    bank = packed.bank
    protected = packed.protected
    attention = packed.attention
    # Stage 4: lifecycle beside the bank. M1 holds every expert SURE; its
    # prepare-phase is a no-op the blocks call before dispatch.
    lifecycle = ExpertLifecycle(bank)

    def prot(name: str) -> torch.Tensor:
        return protected[name].to(dtype)

    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)

    inner = model.model
    num_layers = config.num_hidden_layers

    for i in range(num_layers):
        layer = inner.layers[i]

        # MoE block → TernaryMoEBlock (drops the meta fused experts).
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            layer.mlp = TernaryMoEBlock(
                bank,
                i,
                prot(f"model.layers.{i}.mlp.gate.weight"),
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                hidden_act=config.hidden_act,
                lifecycle=lifecycle,
            )
        else:  # dense Qwen3MoeMLP layer (none in Qwen3-30B-A3B, but be safe)
            for proj in ("gate", "up", "down"):
                key = f"model.layers.{i}.mlp.{proj}_proj.weight"
                if key in protected:
                    _assign_param(getattr(layer.mlp, f"{proj}_proj"), "weight", prot(key))

        # Attention projections → packed ternary.
        for proj in _ATTN_PROJ:
            setattr(layer.self_attn, f"{proj}_proj", attention[(i, proj)])
        # Attention sub-norms + decoder norms (RMSNorm weights).
        _assign_param(layer.self_attn.q_norm, "weight",
                      prot(f"model.layers.{i}.self_attn.q_norm.weight"))
        _assign_param(layer.self_attn.k_norm, "weight",
                      prot(f"model.layers.{i}.self_attn.k_norm.weight"))
        _assign_param(layer.input_layernorm, "weight",
                      prot(f"model.layers.{i}.input_layernorm.weight"))
        _assign_param(layer.post_attention_layernorm, "weight",
                      prot(f"model.layers.{i}.post_attention_layernorm.weight"))

    # Embeddings, final norm, LM head.
    _assign_param(inner.embed_tokens, "weight", prot("model.embed_tokens.weight"))
    _assign_param(inner.norm, "weight", prot("model.norm.weight"))
    _assign_param(model.lm_head, "weight", prot("lm_head.weight"))

    # Rotary embedding: real inv_freq (the meta one carries a meta buffer).
    inner.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    # Register the shared bank once so model.to(device) moves it.
    model._ternary_expert_bank = bank
    # Lifecycle is a plain object (not an nn.Module) → attached for queryability
    # (model._expert_lifecycle.state(layer, expert)) without re-registering bank.
    model._expert_lifecycle = lifecycle

    # Move non-meta state to the target device/dtype. Attention packed
    # modules carry uint8 buffers (dtype-agnostic); protected/router are
    # cast to `dtype`.
    model = model.to(device=device)
    model.eval()

    # Loud check: nothing left on meta.
    stranded: List[str] = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_meta
    ]
    if stranded:
        raise RuntimeError(
            f"build_runnable_qwen3_moe left {len(stranded)} tensor(s) on the "
            f"meta device — assembly is incomplete. First few: {stranded[:8]}"
        )
    return model
