"""
Tests for ``terncore.convert._load_sensitivity_map``.

The dormant-scan finding (2026-05-28): ``full_convert`` previously
hard-coded its sensitivity map source to ``benchmarks/gemma4_e4b_dryrun.json``,
which name-matches only Gemma-4-family weights — silently giving zero INT4
routing for any non-Gemma architecture (Qwen3-MoE, etc.). The patched
helper accepts an explicit caller-supplied path and supports both the new
sensitivity-scan schema (``layers``) and the legacy dryrun schema
(``tolerance_scan``).

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terncore.convert import _load_sensitivity_map


def _noop(_msg):  # log sink for the helper
    return None


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_loads_new_layers_schema(tmp_path):
    p = _write(tmp_path / "scan.json", {
        "layers": [
            {"name": "model.layers.0.q_proj.weight", "relative_error": 0.55},
            {"name": "model.layers.1.q_proj.weight", "relative_error": 0.42},
        ],
    })
    smap, src = _load_sensitivity_map(p, Path("/no/such/legacy.json"), _noop)
    assert src == p
    assert smap == {
        "model.layers.0.q_proj.weight": 0.55,
        "model.layers.1.q_proj.weight": 0.42,
    }


def test_loads_legacy_tolerance_scan_schema(tmp_path):
    p = _write(tmp_path / "legacy.json", {
        "tolerance_scan": [
            {"name": "model.layers.5.v_proj.weight", "relative_error": 0.61},
        ],
    })
    smap, src = _load_sensitivity_map(None, p, _noop)
    assert src == p
    assert smap == {"model.layers.5.v_proj.weight": 0.61}


def test_explicit_path_overrides_legacy(tmp_path):
    explicit = _write(tmp_path / "explicit.json", {
        "layers": [{"name": "A", "relative_error": 0.7}]
    })
    legacy = _write(tmp_path / "legacy.json", {
        "tolerance_scan": [{"name": "B", "relative_error": 0.3}]
    })
    smap, src = _load_sensitivity_map(explicit, legacy, _noop)
    assert src == explicit
    assert smap == {"A": 0.7}


def test_missing_paths_return_empty(tmp_path):
    smap, src = _load_sensitivity_map(None, tmp_path / "absent.json", _noop)
    assert smap == {}
    assert src is None


def test_entries_missing_fields_skipped(tmp_path):
    p = _write(tmp_path / "scan.json", {
        "layers": [
            {"name": "ok", "relative_error": 0.5},
            {"name": "no_re"},
            {"relative_error": 0.1},     # missing name
            {"name": "ok2", "relative_error": 0.6},
        ],
    })
    smap, _ = _load_sensitivity_map(p, Path("/none"), _noop)
    assert smap == {"ok": 0.5, "ok2": 0.6}


def test_empty_lists_load_to_empty_map(tmp_path):
    p = _write(tmp_path / "empty.json", {"layers": []})
    smap, src = _load_sensitivity_map(p, Path("/none"), _noop)
    assert smap == {}
    assert src == p
