"""CoreML MIL builder for Gemma 4 E4B (text path) — Route B hand-built graph.

The Llama scaffold (`build_llama_coreml`) cannot express E4B: it needs per-type
head_dim (256 sliding / 512 full), per-type RoPE (θ + partial-rotary 0.25 on full),
cross-layer KV-sharing (≥L24 reuse L22 sliding / L23 full), a per-layer-input
subsystem, q/k/**v**-norm (v_norm with_scale=False), scaling=1.0, the embedding
×√hidden normalizer, and the final logit softcap. This module builds that graph.

The graph code here is the validated Route-B builder (parity-gated against HF
`output_hidden_states` to cos=0.99999996 over the full 42-layer stack). The weight
*source* is swappable — a dict of fp32 arrays (verification) or a TernModelReader
(production ternary injection) — so promotion preserves the gated graph exactly.

E4B is dense (enable_moe_block=False) — MoE is the 26B delta and is not built here.

Copyright (c) 2026 Robert Lakelin. All rights reserved. Inventor: Robert Lakelin.
"""
from __future__ import annotations

import numpy as np
from coremltools.converters.mil.mil import Builder as mb


# ---------------------------------------------------------------------------
# E4B text topology (from config.text_config). Full-attention layer indices and
# the KV-share map are derived, not hand-listed.
# ---------------------------------------------------------------------------
GEMMA4_E4B = {
    "num_layers": 42,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim_sliding": 256,
    "head_dim_full": 512,
    "hidden_size_per_layer_input": 256,
    "vocab_size": 262144,
    "num_kv_shared_layers": 18,
    "sliding_window": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta_sliding": 1.0e4,
    "rope_theta_full": 1.0e6,
    "partial_rotary_full": 0.25,
    "final_logit_softcap": 30.0,
    "embed_scale": 2560 ** 0.5,
    "embed_scale_per_layer": 256 ** 0.5,
    "per_layer_model_projection_scale": 2560 ** -0.5,
    "per_layer_input_scale": 2.0 ** -0.5,
    "tie_word_embeddings": True,
    # layer_types pattern [s×5, f] × 7
    "full_layers": (5, 11, 17, 23, 29, 35, 41),
}


def full_layer_set(cfg):
    return set(cfg["full_layers"])


def kv_shared_index(cfg, i):
    """Shared layer i (≥ first_shared) reuses the last non-shared layer of its
    type: full layers reuse the last non-shared full, sliding the last sliding."""
    full = full_layer_set(cfg)
    first_shared = cfg["num_layers"] - cfg["num_kv_shared_layers"]
    prev = list(range(first_shared))
    want_full = i in full
    cands = [j for j in prev if (j in full) == want_full]
    return cands[-1]


# ---------------------------------------------------------------------------
# MIL building blocks (Gemma forms).
# ---------------------------------------------------------------------------
def rms_norm(x, weight_var=None, eps=1e-6):
    """Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) [* weight]. No +1 shift."""
    sq = mb.mul(x=x, y=x)
    mean_sq = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
    inv = mb.rsqrt(x=mb.add(x=mean_sq, y=np.float32(eps)))
    y = mb.mul(x=x, y=inv)
    if weight_var is not None:
        y = mb.mul(x=y, y=weight_var)
    return y


def rope_tables(S, head_dim, theta, partial=1.0):
    """GPT-NeoX cos/sin, len head_dim/2. partial<1 → zero-freq NOPE tail
    (partial-rotary falls out for free — those dims get cos=1/sin=0)."""
    rope_angles = int(partial * head_dim // 2)
    inv = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        inv = np.concatenate([inv, np.zeros(nope, np.float32)])
    t = np.arange(S, dtype=np.float32)
    f = np.outer(t, inv)
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)


def causal_mask(S, sentinel=-1.0e4):
    """Finite large-negative additive causal mask (fp16-safe; avoids -inf→NaN
    on a fully-masked row, and zeroes attention to right-pad positions)."""
    m = np.zeros((S, S), np.float32)
    m[np.triu_indices(S, 1)] = sentinel
    return m


def _rope(x, cos_v, sin_v, d):
    x1 = mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[0, 0, 0, d // 2],
                           end_mask=[True, True, True, False])
    x2 = mb.slice_by_index(x=x, begin=[0, 0, 0, d // 2], end=[0, 0, 0, 0],
                           end_mask=[True, True, True, True])
    r1 = mb.sub(x=mb.mul(x=x1, y=cos_v), y=mb.mul(x=x2, y=sin_v))
    r2 = mb.add(x=mb.mul(x=x2, y=cos_v), y=mb.mul(x=x1, y=sin_v))
    return mb.concat(values=[r1, r2], axis=-1)


def _repeat_kv(x, n_kv, n_rep, d, S):
    """Interleaving repeat_kv (each KV head repeated consecutively)."""
    x = mb.reshape(x=x, shape=[1, n_kv, 1, S, d])
    x = mb.tile(x=x, reps=[1, 1, n_rep, 1, 1])
    return mb.reshape(x=x, shape=[1, n_kv * n_rep, S, d])


def build_kv(hidden, *, w, eps, n_kv, d, S, cos_np, sin_np):
    """Post-norm/RoPE/transpose (k,v) [1,n_kv,S,d] each — the stashable KV a
    KV-shared downstream layer reuses (positions identical across same type)."""
    cos_v = mb.const(val=cos_np.reshape(1, 1, S, d // 2))
    sin_v = mb.const(val=sin_np.reshape(1, 1, S, d // 2))
    k = mb.linear(x=hidden, weight=mb.const(val=w["k_proj"].astype(np.float32)))
    k = mb.reshape(x=k, shape=[1, S, n_kv, d])
    k = rms_norm(k, mb.const(val=w["k_norm"].astype(np.float32)), eps=eps)
    k = _rope(mb.transpose(x=k, perm=[0, 2, 1, 3]), cos_v, sin_v, d)
    v = mb.linear(x=hidden, weight=mb.const(val=w["v_proj"].astype(np.float32)))
    v = mb.reshape(x=v, shape=[1, S, n_kv, d])
    v = rms_norm(v, None, eps=eps)                       # v_norm with_scale=False
    v = mb.transpose(x=v, perm=[0, 2, 1, 3])             # NO RoPE on v
    return k, v


def build_attention(hidden, *, w, eps, n_q, n_kv, d, S, cos_np, sin_np, mask_np,
                    shared_kv=None):
    """scaling=1.0 (NOT 1/√d); v_norm with_scale=False; GQA interleave;
    shared_kv=(k,v) → KV-shared layer reuses upstream KV (skip own k/v)."""
    gqa = n_q // n_kv
    cos_v = mb.const(val=cos_np.reshape(1, 1, S, d // 2))
    sin_v = mb.const(val=sin_np.reshape(1, 1, S, d // 2))

    def proj(name):
        return mb.linear(x=hidden, weight=mb.const(val=w[name].astype(np.float32)))

    q = proj("q_proj")
    q = mb.reshape(x=q, shape=[1, S, n_q, d])
    q = rms_norm(q, mb.const(val=w["q_norm"].astype(np.float32)), eps=eps)
    q = _rope(mb.transpose(x=q, perm=[0, 2, 1, 3]), cos_v, sin_v, d)
    if shared_kv is not None:
        k, v = shared_kv
    else:
        k = mb.reshape(x=proj("k_proj"), shape=[1, S, n_kv, d])
        k = rms_norm(k, mb.const(val=w["k_norm"].astype(np.float32)), eps=eps)
        k = _rope(mb.transpose(x=k, perm=[0, 2, 1, 3]), cos_v, sin_v, d)
        v = mb.reshape(x=proj("v_proj"), shape=[1, S, n_kv, d])
        v = rms_norm(v, None, eps=eps)
        v = mb.transpose(x=v, perm=[0, 2, 1, 3])
    if gqa > 1:
        k = _repeat_kv(k, n_kv, gqa, d, S)
        v = _repeat_kv(v, n_kv, gqa, d, S)
    attn = mb.matmul(x=q, y=mb.transpose(x=k, perm=[0, 1, 3, 2]))   # scaling=1.0
    attn = mb.add(x=attn, y=mb.const(val=mask_np.reshape(1, 1, S, S)))
    attn = mb.softmax(x=attn, axis=-1)
    out = mb.matmul(x=attn, y=v)
    out = mb.reshape(x=mb.transpose(x=out, perm=[0, 2, 1, 3]), shape=[1, S, n_q * d])
    return mb.linear(x=out, weight=mb.const(val=w["o_proj"].astype(np.float32)))


def _gelu_tanh(x):
    return mb.gelu(x=x, mode="TANH_APPROXIMATION")


def build_decoder_layer(hidden, per_layer_input, *, w, eps, n_q, n_kv, d, S,
                        cos_np, sin_np, mask_np, shared_kv=None, emit_kv=False):
    """4-norm residual + attention + gelu_tanh MLP + per-layer-input + layer_scalar."""
    def c(name): return mb.const(val=w[name].astype(np.float32))

    r = hidden
    h = rms_norm(hidden, c("input_layernorm"), eps=eps)
    produced_kv = None
    if emit_kv:
        produced_kv = build_kv(h, w=w, eps=eps, n_kv=n_kv, d=d, S=S, cos_np=cos_np, sin_np=sin_np)
    h = build_attention(h, w=w, eps=eps, n_q=n_q, n_kv=n_kv, d=d, S=S,
                        cos_np=cos_np, sin_np=sin_np, mask_np=mask_np, shared_kv=shared_kv)
    h = rms_norm(h, c("post_attention_layernorm"), eps=eps)
    h = mb.add(x=r, y=h)

    r = h
    h2 = rms_norm(h, c("pre_feedforward_layernorm"), eps=eps)
    h2 = mb.mul(x=_gelu_tanh(mb.linear(x=h2, weight=c("gate_proj"))),
                y=mb.linear(x=h2, weight=c("up_proj")))
    h2 = mb.linear(x=h2, weight=c("down_proj"))
    h2 = rms_norm(h2, c("post_feedforward_layernorm"), eps=eps)
    h = mb.add(x=r, y=h2)

    r = h
    h3 = _gelu_tanh(mb.linear(x=h, weight=c("per_layer_input_gate")))
    h3 = mb.mul(x=h3, y=per_layer_input)
    h3 = mb.linear(x=h3, weight=c("per_layer_projection"))
    h3 = rms_norm(h3, c("post_per_layer_input_norm"), eps=eps)
    h = mb.add(x=r, y=h3)

    ls = float(np.asarray(w["layer_scalar"]).reshape(-1)[0])
    h = mb.mul(x=h, y=np.float32(ls))
    return (h, produced_kv) if emit_kv else h


def build_e4b_residual(input_ids, *, cfg, layers, embed_table, ple_table, W_proj,
                       w_pln, final_norm_w, S, checkpoints=()):
    """Full 42-layer residual stream. input_ids:[1,S] (reduced-table indices
    0..S-1). layers: list of per-layer weight dicts. Returns dict of named vars
    (final normed hidden + any requested checkpoints)."""
    N = cfg["num_layers"]; P = cfg["hidden_size_per_layer_input"]
    n_q = cfg["num_attention_heads"]; n_kv = cfg["num_key_value_heads"]
    eps = cfg["rms_norm_eps"]; full = full_layer_set(cfg)
    ropes = {"s": rope_tables(S, cfg["head_dim_sliding"], cfg["rope_theta_sliding"], 1.0),
             "f": rope_tables(S, cfg["head_dim_full"], cfg["rope_theta_full"], cfg["partial_rotary_full"])}
    masks = {"s": causal_mask(S), "f": causal_mask(S)}

    ie = mb.gather(x=mb.const(val=embed_table.astype(np.float32)), indices=input_ids, axis=0)
    ie = mb.mul(x=ie, y=np.float32(cfg["embed_scale"]))

    proj = mb.linear(x=ie, weight=mb.const(val=W_proj.astype(np.float32)))
    proj = mb.mul(x=proj, y=np.float32(cfg["per_layer_model_projection_scale"]))
    proj = mb.reshape(x=proj, shape=[1, S, N, P])
    proj = rms_norm(proj, mb.const(val=w_pln.astype(np.float32)), eps=eps)
    ple = mb.gather(x=mb.const(val=ple_table.astype(np.float32)), indices=input_ids, axis=0)
    ple = mb.mul(x=ple, y=np.float32(cfg["embed_scale_per_layer"]))
    ple = mb.reshape(x=ple, shape=[1, S, N, P])
    pli = mb.mul(x=mb.add(x=proj, y=ple), y=np.float32(cfg["per_layer_input_scale"]))

    first_shared = N - cfg["num_kv_shared_layers"]
    producers = {kv_shared_index(cfg, i) for i in range(first_shared, N)}
    h = ie; stash = {}; outs = {}
    for i in range(N):
        typ = "f" if i in full else "s"
        d = cfg["head_dim_full"] if i in full else cfg["head_dim_sliding"]
        cos_np, sin_np = ropes[typ]
        pli_i = mb.slice_by_index(
            x=pli, begin=[0, 0, i, 0], end=[0, 0, i + 1, 0],
            end_mask=[True, True, False, True], squeeze_mask=[False, False, True, False])
        shared = stash[kv_shared_index(cfg, i)] if i >= first_shared else None
        emit = i in producers
        res = build_decoder_layer(h, pli_i, w=layers[i], eps=eps, n_q=n_q, n_kv=n_kv,
                                  d=d, S=S, cos_np=cos_np, sin_np=sin_np, mask_np=masks[typ],
                                  shared_kv=shared, emit_kv=emit)
        if emit:
            h, stash[i] = res
        else:
            h = res
        if i in checkpoints:
            outs[f"h_L{i}"] = mb.identity(x=h, name=f"h_L{i}")
    h = rms_norm(h, mb.const(val=final_norm_w.astype(np.float32)), eps=eps)
    outs["h_final"] = mb.identity(x=h, name="h_final")
    return outs
