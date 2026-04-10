"""
Integration tests for tern-core v0.5.0 streaming conversion pipeline.

Tests ShardedWeightIterator, StreamingConverter, streaming_scan,
and MixedPrecisionConverter.from_protection_list().

Uses a synthetic sharded safetensors model (2 blocks, 2 shards) to
avoid real model downloads.  All tests run with zero external deps
beyond torch, safetensors, and the terncore package.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Fixture: synthetic 2-block, 2-shard model
# ---------------------------------------------------------------------------

HIDDEN = 64
INTER = 128
VOCAB = 256

def _layer_weights(block_idx: int) -> dict[str, torch.Tensor]:
    """Generate weight tensors for one transformer block."""
    torch.manual_seed(42 + block_idx)
    prefix = f"model.layers.{block_idx}"
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        f"{prefix}.self_attn.k_proj.weight": torch.randn(HIDDEN // 4, HIDDEN),
        f"{prefix}.self_attn.v_proj.weight": torch.randn(HIDDEN // 4, HIDDEN),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(INTER, HIDDEN),
        f"{prefix}.mlp.up_proj.weight": torch.randn(INTER, HIDDEN),
        f"{prefix}.mlp.down_proj.weight": torch.randn(HIDDEN, INTER),
        f"{prefix}.input_layernorm.weight": torch.randn(HIDDEN),
        f"{prefix}.post_attention_layernorm.weight": torch.randn(HIDDEN),
    }


@pytest.fixture
def synthetic_model(tmp_path: Path) -> Path:
    """Create a synthetic 2-block sharded safetensors model."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Shard 1: block 0 + embed_tokens
    shard1_tensors = _layer_weights(0)
    shard1_tensors["model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN)

    # Shard 2: block 1 + lm_head + norm
    shard2_tensors = _layer_weights(1)
    shard2_tensors["lm_head.weight"] = torch.randn(VOCAB, HIDDEN)
    shard2_tensors["model.norm.weight"] = torch.randn(HIDDEN)

    save_file(shard1_tensors, model_dir / "model-00001-of-00002.safetensors")
    save_file(shard2_tensors, model_dir / "model-00002-of-00002.safetensors")

    # Build index
    weight_map = {}
    for name in shard1_tensors:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name in shard2_tensors:
        weight_map[name] = "model-00002-of-00002.safetensors"

    total_size = sum(t.numel() * 4 for t in shard1_tensors.values()) + \
                 sum(t.numel() * 4 for t in shard2_tensors.values())

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    return model_dir


# ---------------------------------------------------------------------------
# ShardedWeightIterator tests
# ---------------------------------------------------------------------------

class TestShardedWeightIterator:

    def test_loads_index(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator
        loader = ShardedWeightIterator(synthetic_model)
        assert loader.num_blocks == 2
        assert loader.num_weights == 21  # 9*2 blocks + 3 non-block

    def test_block_indices(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator
        loader = ShardedWeightIterator(synthetic_model)
        assert loader.block_indices == [0, 1]

    def test_eligible_linear_names(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator
        loader = ShardedWeightIterator(synthetic_model)
        eligible = loader.eligible_linear_names()
        # 7 linear per block * 2 blocks = 14
        assert len(eligible) == 14
        # Should not include layernorm
        assert not any("layernorm" in n for n in eligible)
        # Should not include embed or lm_head
        assert not any("embed" in n for n in eligible)
        assert not any("lm_head" in n for n in eligible)

    def test_iter_blocks_yields_all(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator, WeightBlock, NonBlockWeights
        loader = ShardedWeightIterator(synthetic_model)
        items = list(loader)

        blocks = [i for i in items if isinstance(i, WeightBlock)]
        non_blocks = [i for i in items if isinstance(i, NonBlockWeights)]

        assert len(blocks) == 2
        assert len(non_blocks) == 1
        assert blocks[0].block_idx == 0
        assert blocks[1].block_idx == 1

    def test_block_has_correct_weights(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator, WeightBlock
        loader = ShardedWeightIterator(synthetic_model)
        block0 = next(iter(loader))
        assert isinstance(block0, WeightBlock)
        assert len(block0.weights) == 9
        assert "model.layers.0.self_attn.q_proj.weight" in block0.weights
        assert block0.weights["model.layers.0.self_attn.q_proj.weight"].shape == (HIDDEN, HIDDEN)

    def test_non_block_has_embed_lm_head_norm(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator, NonBlockWeights
        loader = ShardedWeightIterator(synthetic_model)
        items = list(loader)
        nb = [i for i in items if isinstance(i, NonBlockWeights)][0]
        assert "model.embed_tokens.weight" in nb.weights
        assert "lm_head.weight" in nb.weights
        assert "model.norm.weight" in nb.weights

    def test_iter_tensors_yields_all(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator
        loader = ShardedWeightIterator(synthetic_model)
        tensors = list(loader.iter_tensors())
        assert len(tensors) == 21
        names = [t[0] for t in tensors]
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "lm_head.weight" in names

    def test_linear_names_property(self, synthetic_model):
        from terncore.sharded_loader import ShardedWeightIterator, WeightBlock
        loader = ShardedWeightIterator(synthetic_model)
        block0 = next(iter(loader))
        # 7 linear (2-D), 2 layernorm (1-D)
        assert len(block0.linear_names) == 7

    def test_missing_index_raises(self, tmp_path):
        from terncore.sharded_loader import ShardedWeightIterator
        with pytest.raises(FileNotFoundError, match="safetensors index"):
            ShardedWeightIterator(tmp_path)


# ---------------------------------------------------------------------------
# StreamingConverter tests
# ---------------------------------------------------------------------------

class TestStreamingConverter:

    def test_convert_produces_tern_model(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        output = tmp_path / "test.tern-model"
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=output,
            verbose=False,
        )
        report = converter.convert()
        assert output.exists()
        assert report.output_size_bytes > 0

    def test_all_weights_accounted(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            verbose=False,
        )
        report = converter.convert()
        assert report.total_weights == 21
        assert report.ternary_weights + report.protected_weights == report.total_weights

    def test_protection_list_respected(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        protect = ["model.layers.0.self_attn.q_proj.weight"]
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            protection_list=protect,
            verbose=False,
        )
        report = converter.convert()
        protected_names = [l["name"] for l in report.per_layer if l["dtype"] == "float16"]
        assert "model.layers.0.self_attn.q_proj.weight" in protected_names

    def test_always_protected_patterns(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            verbose=False,
        )
        report = converter.convert()
        protected_names = {l["name"] for l in report.per_layer if l["dtype"] == "float16"}
        # embed, lm_head, norm, layernorms are always protected
        assert "model.embed_tokens.weight" in protected_names
        assert "lm_head.weight" in protected_names
        assert "model.norm.weight" in protected_names

    def test_compression_ratio_gt_1(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            verbose=False,
        )
        report = converter.convert()
        assert report.compression_ratio > 1.0

    def test_1d_weights_always_protected(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            verbose=False,
        )
        report = converter.convert()
        for layer in report.per_layer:
            if len(layer["shape"]) < 2:
                assert layer["dtype"] == "float16", f"1-D tensor {layer['name']} should be protected"

    def test_blocks_processed_count(self, synthetic_model, tmp_path):
        from terncore.streaming_convert import StreamingConverter
        converter = StreamingConverter(
            model_dir=synthetic_model,
            output_path=tmp_path / "out.tern-model",
            verbose=False,
        )
        report = converter.convert()
        assert report.blocks_processed == 2


# ---------------------------------------------------------------------------
# Layer sensitivity tests
# ---------------------------------------------------------------------------

class TestLayerSensitivity:

    def test_compute_layer_sensitivity(self):
        from terncore.autoscan import _compute_layer_sensitivity
        torch.manual_seed(42)
        w = torch.randn(64, 32)
        sens = _compute_layer_sensitivity("test.layer", w, threshold=0.7)
        assert sens.name == "test.layer"
        assert 0.0 < sens.relative_error < 1.0
        assert sens.num_params == 64 * 32
        assert 0.0 <= sens.sparsity <= 1.0
        assert sens.alpha > 0.0

    def test_zero_weight_no_crash(self):
        from terncore.autoscan import _compute_layer_sensitivity
        w = torch.zeros(32, 16)
        sens = _compute_layer_sensitivity("zero.layer", w, threshold=0.7)
        assert sens.relative_error == 0.0
        assert sens.sparsity == 1.0

    def test_higher_threshold_more_sparsity(self):
        from terncore.autoscan import _compute_layer_sensitivity
        torch.manual_seed(42)
        w = torch.randn(64, 32)
        low = _compute_layer_sensitivity("test", w, threshold=0.3)
        high = _compute_layer_sensitivity("test", w, threshold=0.9)
        assert high.sparsity > low.sparsity

    def test_sensitivity_sort_matches_autoscan(self):
        """Verify that lower reconstruction error = more tolerant."""
        from terncore.autoscan import _compute_layer_sensitivity
        torch.manual_seed(42)
        # Sparse weight (mostly zeros) should have lower error
        sparse_w = torch.zeros(64, 32)
        sparse_w[0, 0] = 1.0
        dense_w = torch.randn(64, 32)
        sparse_sens = _compute_layer_sensitivity("sparse", sparse_w, threshold=0.7)
        dense_sens = _compute_layer_sensitivity("dense", dense_w, threshold=0.7)
        assert sparse_sens.relative_error < dense_sens.relative_error


# ---------------------------------------------------------------------------
# streaming_scan tests (unit — no real model)
# ---------------------------------------------------------------------------

class TestStreamingScan:

    def test_streaming_scan_with_synthetic(self, synthetic_model):
        from terncore.autoscan import streaming_scan
        result = streaming_scan(
            model_id=str(synthetic_model),
            threshold=0.7,
            ppl_headroom=0.05,
            use_cache=False,
            baseline_ppl=6.0,  # Skip pass 1
        )
        assert result.model_id == str(synthetic_model)
        assert result.baseline_ppl == 6.0
        assert result.ppl_ceiling == 6.3  # 6.0 * 1.05
        assert result.total_eligible == 14
        assert result.layers_converted >= 0
        assert result.layers_converted <= result.total_eligible
        assert len(result.protection_list) + result.layers_converted == result.total_eligible
        assert result.compression_ratio >= 1.0

    def test_streaming_scan_tight_budget_converts_fewer(self, synthetic_model):
        from terncore.autoscan import streaming_scan
        loose = streaming_scan(
            model_id=str(synthetic_model), threshold=0.7, ppl_headroom=0.20,
            use_cache=False, baseline_ppl=6.0,
        )
        tight = streaming_scan(
            model_id=str(synthetic_model), threshold=0.7, ppl_headroom=0.01,
            use_cache=False, baseline_ppl=6.0,
        )
        assert tight.layers_converted <= loose.layers_converted

    def test_streaming_scan_produces_sweep_trace(self, synthetic_model):
        from terncore.autoscan import streaming_scan
        result = streaming_scan(
            model_id=str(synthetic_model), threshold=0.7, ppl_headroom=0.05,
            use_cache=False, baseline_ppl=6.0,
        )
        assert len(result.sweep_trace) > 0
        first = result.sweep_trace[0]
        assert "layer" in first
        assert "relative_error" in first
        assert "within_budget" in first


# ---------------------------------------------------------------------------
# MixedPrecisionConverter.from_protection_list tests
# ---------------------------------------------------------------------------

class TestFromProtectionList:

    def test_creates_converter(self):
        from terncore.mixed_precision import MixedPrecisionConverter
        protect = ["model.layers.0.self_attn.q_proj"]
        conv = MixedPrecisionConverter.from_protection_list(protect, threshold=0.5)
        assert conv.threshold == 0.5
        assert conv.auto is False
        assert "model.layers.0.self_attn.q_proj" in conv.protection_list

    def test_skips_auto_scan(self):
        from terncore.mixed_precision import MixedPrecisionConverter
        conv = MixedPrecisionConverter.from_protection_list([], threshold=0.7)
        assert conv._explicit_protection is True
        assert conv.auto is False

    def test_convert_uses_provided_list(self):
        """Convert a tiny model and verify the protection list is used."""
        from terncore.mixed_precision import MixedPrecisionConverter
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = nn.Linear(16, 16)
                self.layer1 = nn.Linear(16, 16)

            def forward(self, x):
                return self.layer1(self.layer0(x))

        model = TinyModel()
        conv = MixedPrecisionConverter.from_protection_list(
            protection_list=["layer0"],
            threshold=0.7,
        )
        report = conv.convert(model)
        # layer0 protected, layer1 converted
        assert report.converted_layers == 1
        assert report.skipped_layers == 1
