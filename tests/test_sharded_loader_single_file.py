"""
Tests for ShardedWeightIterator single-file (no-index) support.

Gemma 4 12B-it ships a single ``model.safetensors`` with no
``model.safetensors.index.json``. The loader must synthesize a
weight_map from the file header rather than raising.

Copyright (c) 2025-2026 Robert Lakelin. All rights reserved.
"""

import pytest
import torch
from safetensors.torch import save_file

from terncore.sharded_loader import (
    NonBlockWeights,
    ShardedWeightIterator,
    WeightBlock,
)


def _write_single_file(model_dir):
    """Two transformer blocks + non-block weights, one model.safetensors."""
    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8),
        "model.norm.weight": torch.randn(8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
        "model.layers.0.input_layernorm.weight": torch.randn(8),
        # Layer 1 mimics a value-free global layer: no v_proj.
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.1.input_layernorm.weight": torch.randn(8),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors


def test_single_file_synthesizes_weight_map(tmp_path):
    written = _write_single_file(tmp_path)
    loader = ShardedWeightIterator(tmp_path)

    assert loader.num_weights == len(written)
    # Every tensor maps to the lone file.
    assert set(loader.weight_map.values()) == {"model.safetensors"}
    assert loader.total_size == (tmp_path / "model.safetensors").stat().st_size
    assert loader.block_indices == [0, 1]
    assert loader.num_blocks == 2


def test_single_file_iter_blocks_loads_tensors(tmp_path):
    written = _write_single_file(tmp_path)
    loader = ShardedWeightIterator(tmp_path)

    blocks = {}
    non_block = None
    for item in loader:
        if isinstance(item, WeightBlock):
            blocks[item.block_idx] = item
        elif isinstance(item, NonBlockWeights):
            non_block = item

    assert set(blocks) == {0, 1}
    # Value-free global layer (block 1) loads with no v_proj, no crash.
    assert "model.layers.1.self_attn.q_proj.weight" in blocks[1].weights
    assert "model.layers.1.self_attn.v_proj.weight" not in blocks[1].weights
    # Byte-faithful round-trip on a sampled tensor.
    got = blocks[0].weights["model.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(got, written["model.layers.0.self_attn.q_proj.weight"])
    assert non_block is not None
    assert "model.embed_tokens.weight" in non_block.weights


def test_missing_both_index_and_single_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShardedWeightIterator(tmp_path)
