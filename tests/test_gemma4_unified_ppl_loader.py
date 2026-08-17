"""Unit tests for the Gemma 4 Unified PPL loader (tools/gemma4_unified_ppl.py).

Covers the two pieces the stock ``tern_ppl_bench`` loader lacks and which this
tool exists to provide:

1. ``translate_name`` — bridges the manifest's logical Gemma names
   (``model.layers.*`` …) to the transformers-5.10 ``model.language_model.*``
   tree, passing the multimodal embedder names through unchanged.
2. ``overlay`` strict-quant gate — places decoder-tower entries, tolerates
   text-irrelevant multimodal FP16 entries that are absent from a text
   ``CausalLM``, and flags any *ternary/INT4* entry that fails to place
   (a silent decoder-tower miss would corrupt the measurement).

These exercise the load-bearing logic with lightweight fakes — no model
download, no transformers forward.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# ── Import the loader module via path (lives in tools/, not src/) ──────
_TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "gemma4_unified_ppl.py"
_spec = importlib.util.spec_from_file_location("gemma4_unified_ppl", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
gemma4_unified_ppl = importlib.util.module_from_spec(_spec)
sys.modules["gemma4_unified_ppl"] = gemma4_unified_ppl
_spec.loader.exec_module(gemma4_unified_ppl)

translate_name = gemma4_unified_ppl.translate_name
overlay = gemma4_unified_ppl.overlay


# ── translate_name: text-tower bridge + multimodal passthrough ─────────

def test_translate_text_tower_prefixes():
    cases = {
        "model.layers.0.self_attn.q_proj.weight":
            "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.layers.47.mlp.down_proj.weight":
            "model.language_model.layers.47.mlp.down_proj.weight",
        "model.embed_tokens.weight":
            "model.language_model.embed_tokens.weight",
        "model.norm.weight":
            "model.language_model.norm.weight",
    }
    for raw, expected in cases.items():
        assert translate_name(raw) == expected


def test_translate_multimodal_and_unknown_passthrough():
    # Vision/audio embedder names and already-nested names are untouched.
    passthrough = [
        "model.embed_vision.embedding_projection.weight",
        "model.embed_audio.embedding_projection.weight",
        "model.vision_embedder.patch_dense.weight",
        "lm_head.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    ]
    for name in passthrough:
        assert translate_name(name) == name


# ── overlay: placement + strict-quant gate ─────────────────────────────

def _fake_model(keyed_params):
    """A minimal model exposing named_parameters()/named_buffers()."""
    items = [(k, torch.nn.Parameter(torch.ones(*shape), requires_grad=False))
             for k, shape in keyed_params.items()]
    return SimpleNamespace(
        named_parameters=lambda: list(items),
        named_buffers=lambda: [],
    )


def _fake_reader(entries):
    """A reader stub: manifest entries + reconstruct_layer by name."""
    by_name = {e["name"]: e for e in entries}

    def reconstruct_layer(name):
        return {"weight": torch.zeros(*by_name[name]["shape"])}

    return SimpleNamespace(
        manifest={"layers": entries},
        reconstruct_layer=reconstruct_layer,
    )


def test_overlay_places_decoder_skips_multimodal():
    # Decoder ternary entry resolves via the text-tower map; an absent
    # multimodal FP16 entry is skipped but NOT a quant miss.
    model = _fake_model({
        "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
    })
    reader = _fake_reader([
        {"name": "model.layers.0.self_attn.q_proj.weight",
         "dtype": "ternary2", "shape": (4, 4)},
        {"name": "model.embed_vision.embedding_projection.weight",
         "dtype": "float16", "shape": (2, 2)},
    ])
    rep = overlay(reader, model)
    assert rep["placed"] == 1
    assert rep["fail"] == []
    assert rep["skipped_quant"] == []          # gate clean
    assert len(rep["skipped"]) == 1            # the multimodal FP16 entry


def test_overlay_gate_flags_missing_quant_entry():
    # A ternary entry whose (translated) key is absent must surface as a
    # quant miss so the caller aborts rather than measure a corrupt model.
    model = _fake_model({
        "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
    })
    reader = _fake_reader([
        {"name": "model.layers.99.self_attn.k_proj.weight",   # no such layer
         "dtype": "ternary2", "shape": (4, 4)},
    ])
    rep = overlay(reader, model)
    assert rep["placed"] == 0
    assert len(rep["skipped_quant"]) == 1


def test_overlay_flags_shape_mismatch():
    model = _fake_model({
        "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
    })
    reader = _fake_reader([
        {"name": "model.layers.0.self_attn.q_proj.weight",
         "dtype": "ternary2", "shape": (8, 8)},   # wrong shape
    ])
    rep = overlay(reader, model)
    assert len(rep["fail"]) == 1
    assert rep["placed"] == 0
