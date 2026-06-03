"""CI parity gate — gemma4 attention (per-type head_dim, partial-rotary, v-norm).

Runs the CoreML predict in a SUBPROCESS that os._exit(0)s after writing its
verdict, so the coremltools macOS teardown SIGKILL (fires at Py_FinalizeEx, after
predict returns) cannot redden this build. The test judges on the written verdict
file, never on a process exit code.
"""
from __future__ import annotations
import os, sys, json, subprocess, tempfile
import pytest


def test_coreml_gemma4_attention_parity():
    pytest.importorskip("coremltools")
    pytest.importorskip("torch")
    helper = os.path.join(os.path.dirname(__file__), "_coreml_gemma4_parity_subproc.py")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "verdict.json")
        r = subprocess.run([sys.executable, helper, out], capture_output=True, text=True)
        assert os.path.exists(out), (
            f"subprocess produced no verdict (rc={r.returncode})\n"
            f"STDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}")
        v = json.load(open(out))
    assert v["pass"], f"gemma4 attention parity FAIL: {v}"
    for k, m in v.items():
        if k != "pass":
            assert m["cos"] > 0.99999, f"{k}: cos={m['cos']}"
