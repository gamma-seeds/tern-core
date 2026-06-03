"""HF-vs-CoreML attention parity gate for the CoreML export path.

Guards the GQA head-pairing class of bug: `build_llama_coreml` expands KV heads
for grouped-query attention, and the expansion MUST match HF `repeat_kv`
(each KV head repeated consecutively) so query head h pairs with KV head
h // n_rep. A regression to block-tile ([kv0,kv1,kv0,kv1,…]) silently scrambles
attention output while leaving shapes (and throughput) intact — exactly the
failure this gate exists to catch.

Two checks:
  * `test_repeat_kv_matches_hf_ordering` — the `_repeat_kv` helper reproduces
    HF `repeat_kv`'s head ordering exactly (fast, catches the pairing directly).
  * `test_coreml_attention_parity_vs_hf` — a full single GQA attention block
    built with the export path's `_repeat_kv`, converted to a CoreML mlprogram
    (fp32), matches an HF-`repeat_kv` torch reference (cos ≈ 1).

Both `importorskip` coremltools so the suite stays green without it installed.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch


def _hf_repeat_kv():
    from transformers.models.llama.modeling_llama import repeat_kv
    return repeat_kv


def test_repeat_kv_matches_hf_ordering():
    """_repeat_kv head ordering == HF repeat_kv (interleave, not block-tile)."""
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil.mil import Builder as mb, types
    from terncore.coreml_export import _repeat_kv

    n_kv, n_rep, S, d = 3, 4, 5, 2
    # distinguishable per-head signature so any mis-ordering shows
    x = np.arange(n_kv, dtype=np.float32).reshape(1, n_kv, 1, 1)
    x = np.broadcast_to(x, (1, n_kv, S, d)).copy()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, n_kv, S, d), dtype=types.fp32)],
                opset_version=ct.target.iOS18)
    def prog(t):
        return _repeat_kv(t, n_kv, n_rep, d)

    m = ct.convert(prog, source="milinternal", convert_to="mlprogram",
                   minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT32,
                   compute_units=ct.ComputeUnit.CPU_ONLY)
    got = np.asarray(list(m.predict({"t": x}).values())[0])

    repeat_kv = _hf_repeat_kv()
    ref = repeat_kv(torch.tensor(x), n_rep).numpy()
    assert got.shape == ref.shape == (1, n_kv * n_rep, S, d)
    np.testing.assert_allclose(got, ref, atol=0)
    # the head axis must read [0,0,0,0, 1,1,1,1, 2,2,2,2] — NOT [0,1,2,0,1,2,…]
    head_sig = got[0, :, 0, 0]
    assert list(head_sig) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_coreml_attention_parity_vs_hf():
    """Full GQA attention block (export-path _repeat_kv) vs HF-repeat_kv ref."""
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil.mil import Builder as mb, types
    from terncore.coreml_export import _repeat_kv

    n_q, n_kv, d, S, H = 8, 2, 8, 6, 64
    n_rep = n_q // n_kv
    rng = np.random.default_rng(0)
    qw = rng.standard_normal((n_q * d, H)).astype(np.float32)
    kw = rng.standard_normal((n_kv * d, H)).astype(np.float32)
    vw = rng.standard_normal((n_kv * d, H)).astype(np.float32)
    ow = rng.standard_normal((H, n_q * d)).astype(np.float32)
    x = rng.standard_normal((1, S, H)).astype(np.float32)

    inv = 1.0 / (10000.0 ** (np.arange(0, d, 2) / d))
    fr = np.outer(np.arange(S), inv)
    cos_np, sin_np = np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)
    mask = np.triu(np.full((S, S), -np.inf, np.float32), 1)
    scale = 1.0 / math.sqrt(d)

    # ---- HF-repeat_kv torch reference ----
    repeat_kv = _hf_repeat_kv()
    xt = torch.tensor(x)
    q = (xt @ torch.tensor(qw).T).view(1, S, n_q, d).transpose(1, 2)
    k = (xt @ torch.tensor(kw).T).view(1, S, n_kv, d).transpose(1, 2)
    v = (xt @ torch.tensor(vw).T).view(1, S, n_kv, d).transpose(1, 2)
    cos = torch.tensor(np.concatenate([cos_np, cos_np], -1))[None, None]
    sin = torch.tensor(np.concatenate([sin_np, sin_np], -1))[None, None]

    def rot(t):
        t1, t2 = t[..., :d // 2], t[..., d // 2:]
        return t * cos + torch.cat([-t2, t1], -1) * sin

    q, k = rot(q), rot(k)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    a = (q @ k.transpose(2, 3)) * scale + torch.tensor(mask)
    a = a.softmax(-1)
    o = (a @ v).transpose(1, 2).reshape(1, S, n_q * d)
    ref = (o @ torch.tensor(ow).T).numpy()

    # ---- CoreML micro-graph using the export path's _repeat_kv ----
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, S, H), dtype=types.fp32)],
                opset_version=ct.target.iOS18)
    def prog(hidden):
        cv = mb.const(val=cos_np.reshape(1, 1, S, d // 2))
        sv = mb.const(val=sin_np.reshape(1, 1, S, d // 2))

        def rope(t):
            t1 = mb.slice_by_index(x=t, begin=[0, 0, 0, 0], end=[0, 0, 0, d // 2],
                                   end_mask=[True, True, True, False])
            t2 = mb.slice_by_index(x=t, begin=[0, 0, 0, d // 2], end=[0, 0, 0, 0],
                                   end_mask=[True, True, True, True])
            r1 = mb.sub(x=mb.mul(x=t1, y=cv), y=mb.mul(x=t2, y=sv))
            r2 = mb.add(x=mb.mul(x=t2, y=cv), y=mb.mul(x=t1, y=sv))
            return mb.concat(values=[r1, r2], axis=-1)

        q = mb.transpose(x=mb.reshape(x=mb.linear(x=hidden, weight=mb.const(val=qw)),
                                      shape=[1, S, n_q, d]), perm=[0, 2, 1, 3])
        k = mb.transpose(x=mb.reshape(x=mb.linear(x=hidden, weight=mb.const(val=kw)),
                                      shape=[1, S, n_kv, d]), perm=[0, 2, 1, 3])
        v = mb.transpose(x=mb.reshape(x=mb.linear(x=hidden, weight=mb.const(val=vw)),
                                      shape=[1, S, n_kv, d]), perm=[0, 2, 1, 3])
        q, k = rope(q), rope(k)
        k = _repeat_kv(k, n_kv, n_rep, d)
        v = _repeat_kv(v, n_kv, n_rep, d)
        a = mb.matmul(x=q, y=mb.transpose(x=k, perm=[0, 1, 3, 2]))
        a = mb.mul(x=a, y=np.float32(scale))
        a = mb.add(x=a, y=mb.const(val=mask.reshape(1, 1, S, S)))
        a = mb.softmax(x=a, axis=-1)
        o = mb.matmul(x=a, y=v)
        o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1, 3]), shape=[1, S, n_q * d])
        return mb.linear(x=o, weight=mb.const(val=ow), name="out")

    m = ct.convert(prog, source="milinternal", convert_to="mlprogram",
                   minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT32,
                   compute_units=ct.ComputeUnit.CPU_ONLY)
    got = np.asarray(list(m.predict({"hidden": x}).values())[0]).ravel()

    a, b = ref.ravel().astype(np.float64), got.astype(np.float64)
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos_sim > 0.9999, f"cos={cos_sim} (GQA pairing regressed?)"
    assert np.abs(a - b).max() < 1e-3
