"""Subprocess helper for the gemma4 attention CI parity gate.

Runs the CoreML convert+predict and writes a JSON verdict, then os._exit(0) to
skip interpreter finalization — the coremltools macOS teardown race (an
MLE5ExecutionStream worker grabbing the GIL at Py_FinalizeEx) SIGKILLs at exit
*after* predict returns, which would otherwise redden CI. The parent test judges
on the written verdict, never on this process's exit code.

Compares terncore.coreml_gemma4.build_attention against a torch reference that
replicates Gemma4TextAttention's gemma4-specific math: per-type head_dim, per-type
RoPE with partial-rotary 0.25 (full), q/k/v-norm (v_norm with_scale=False),
scaling=1.0, causal GQA. Tiny random config — no model download.

Usage: python _coreml_gemma4_parity_subproc.py <out.json>
Copyright (c) 2026 Robert Lakelin. Inventor: Robert Lakelin.
"""
from __future__ import annotations
import sys, json, math, numpy as np, torch
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
from terncore.coreml_gemma4 import build_attention, rope_tables, causal_mask, rms_norm  # noqa

OUT = sys.argv[1] if len(sys.argv) > 1 else "gemma4_attn_verdict.json"
H, n_q, n_kv, S, eps = 64, 4, 2, 6, 1e-6


def torch_rmsnorm(x, w, eps):           # gemma form, fp32, no +1 shift
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def run_case(name, d, theta, partial):
    rng = np.random.default_rng(hash(name) % 2**31)
    w = {"q_proj": rng.standard_normal((n_q * d, H)).astype(np.float32),
         "k_proj": rng.standard_normal((n_kv * d, H)).astype(np.float32),
         "v_proj": rng.standard_normal((n_kv * d, H)).astype(np.float32),
         "o_proj": rng.standard_normal((H, n_q * d)).astype(np.float32),
         "q_norm": rng.standard_normal((d,)).astype(np.float32),
         "k_norm": rng.standard_normal((d,)).astype(np.float32)}
    x = rng.standard_normal((1, S, H)).astype(np.float32)
    cos_np, sin_np = rope_tables(S, d, theta, partial)
    mask_np = causal_mask(S)

    # ---- torch reference (gemma4 attention math) ----
    xt = torch.tensor(x)
    cos = torch.tensor(np.concatenate([cos_np, cos_np], -1))[None, None]   # [1,1,S,d]
    sin = torch.tensor(np.concatenate([sin_np, sin_np], -1))[None, None]

    def proj(n): return xt @ torch.tensor(w[n]).T
    q = proj("q_proj").view(1, S, n_q, d).transpose(1, 2)
    q = torch_rmsnorm(q, torch.tensor(w["q_norm"]), eps)
    q = q * cos + rotate_half(q) * sin
    k = proj("k_proj").view(1, S, n_kv, d).transpose(1, 2)
    k = torch_rmsnorm(k, torch.tensor(w["k_norm"]), eps)
    k = k * cos + rotate_half(k) * sin
    v = proj("v_proj").view(1, S, n_kv, d).transpose(1, 2)
    v = torch_rmsnorm(v, None, eps)                         # v_norm with_scale=False
    n_rep = n_q // n_kv
    k = k.repeat_interleave(n_rep, dim=1); v = v.repeat_interleave(n_rep, dim=1)
    attn = q @ k.transpose(2, 3)                            # scaling=1.0
    attn = attn + torch.tensor(mask_np)[None, None]
    attn = attn.softmax(-1)
    o = (attn @ v).transpose(1, 2).reshape(1, S, n_q * d)
    ref = (o @ torch.tensor(w["o_proj"]).T).numpy()

    # ---- CoreML via terncore.coreml_gemma4.build_attention ----
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, S, H), dtype=types.fp32)],
                opset_version=ct.target.iOS18)
    def prog(hidden):
        return build_attention(hidden, w=w, eps=eps, n_q=n_q, n_kv=n_kv, d=d, S=S,
                               cos_np=cos_np, sin_np=sin_np, mask_np=mask_np)
    m = ct.convert(prog, source="milinternal", convert_to="mlprogram",
                   minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
    got = np.asarray(list(m.predict({"hidden": x}).values())[0]).ravel()
    a, b = ref.ravel().astype(np.float64), got.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), float(np.abs(a - b).max())


verdict = {}
cs, mx = run_case("sliding", 16, 1e4, 1.0)
verdict["sliding_d16_fullrotary"] = {"cos": cs, "maxabs": mx}
cf, mf = run_case("full", 32, 1e6, 0.25)
verdict["full_d32_partialrotary0.25"] = {"cos": cf, "maxabs": mf}
verdict["pass"] = bool(cs > 0.99999 and cf > 0.99999)
json.dump(verdict, open(OUT, "w"), indent=2)
print("VERDICT", json.dumps(verdict), flush=True)
import os; os._exit(0)
