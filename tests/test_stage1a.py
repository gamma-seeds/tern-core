"""
Test suite for terncore Stage 1A: CPU reference implementation.

Run with: pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from terncore.arithmetic.quantizer import TernaryQuantizer, SensitivityAnalyzer
from terncore.arithmetic.linear import TernaryLinear, TernaryConv2d
from terncore.engine.inference import TernaryInferenceEngine
from terncore.sparse import (
    generate_sparsity_bitmap,
    pack_ternary_weights,
    unpack_ternary_weights,
    sparsity_info,
)
from terncore.memory import profile_model_memory
from terncore.model_loader import TernModelWriter, TernModelReader


# ═══════════════════════════════════════════════════════════════
# TernaryQuantizer Tests
# ═══════════════════════════════════════════════════════════════


class TestTernaryQuantizer:
    """Tests for Patent 1: ternary weight encoding."""

    def test_output_values(self):
        """Quantised weights must only contain {-1, 0, +1}."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(100, 100)
        ternary, alpha = q.quantize(weights)

        unique = set(ternary.unique().tolist())
        assert unique <= {-1.0, 0.0, 1.0}, f"Unexpected values: {unique}"

    def test_alpha_positive(self):
        """Scaling factor must be positive."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(50, 50)
        _, alpha = q.quantize(weights)
        assert alpha.item() > 0

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        q = TernaryQuantizer(threshold=0.7)
        for shape in [(10,), (10, 20), (10, 20, 30)]:
            weights = torch.randn(*shape)
            ternary, _ = q.quantize(weights)
            assert ternary.shape == weights.shape

    def test_reconstruction_error_bounded(self):
        """Reconstruction error must be finite and reasonable."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(100, 100)
        stats = q.stats(weights)
        assert stats.reconstruction_mse < 10.0
        assert stats.reconstruction_mse >= 0.0

    def test_sparsity_increases_with_threshold(self):
        """Higher threshold should produce more zeros."""
        weights = torch.randn(200, 200)
        sparsities = []
        for t in [0.3, 0.5, 0.7, 0.9]:
            q = TernaryQuantizer(threshold=t)
            stats = q.stats(weights)
            sparsities.append(stats.sparsity)

        # Sparsity should generally increase (allow some tolerance)
        assert sparsities[-1] > sparsities[0]

    def test_dequantize_roundtrip(self):
        """Dequantize(quantize(W)) should approximate W."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(50, 50)
        ternary, alpha = q.quantize(weights)
        reconstructed = q.dequantize(ternary, alpha)
        assert reconstructed.shape == weights.shape
        # Not exact, but should be in the right ballpark
        assert torch.allclose(reconstructed.mean(), weights.mean(), atol=0.5)

    def test_known_values(self):
        """Test with known weights to verify deterministic output."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.tensor([[2.0, -0.1, 0.05, -3.0]])
        ternary, alpha = q.quantize(weights)

        # 2.0 and -3.0 are clearly above/below threshold
        assert ternary[0, 0].item() == 1.0   # large positive → +1
        assert ternary[0, 3].item() == -1.0   # large negative → -1

    def test_invalid_threshold(self):
        """Threshold outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            TernaryQuantizer(threshold=0.0)
        with pytest.raises(ValueError):
            TernaryQuantizer(threshold=1.0)
        with pytest.raises(ValueError):
            TernaryQuantizer(threshold=-0.5)

    def test_all_zeros_input(self):
        """All-zero input should not crash."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.zeros(10, 10)
        ternary, alpha = q.quantize(weights)
        assert (ternary == 0).all()

    def test_stats_fractions_sum_to_one(self):
        """Positive + negative + zero fractions must sum to 1."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(100, 100)
        stats = q.stats(weights)
        total = stats.positive_frac + stats.negative_frac + stats.sparsity
        assert abs(total - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# TernaryLinear Tests
# ═══════════════════════════════════════════════════════════════


class TestTernaryLinear:
    """Tests for Patent 1: ternary linear layer with STE."""

    def test_forward_shape(self):
        """Output shape must be correct."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_forward_eval_shape(self):
        """Eval mode must produce same shape."""
        layer = TernaryLinear(64, 32)
        layer.eval()
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_deterministic_eval(self):
        """Two eval passes with same input must produce identical output.
        Patent 36, Claim 14: deterministic reproducibility."""
        layer = TernaryLinear(64, 32)
        layer.eval()
        x = torch.randn(4, 64)

        y1 = layer(x)
        y2 = layer(x)
        assert torch.equal(y1, y2), "Eval mode must be deterministic"

    def test_gradient_flows(self):
        """Gradients must flow through STE during training."""
        layer = TernaryLinear(32, 16)
        layer.train()
        x = torch.randn(4, 32)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.alpha.grad is not None
        assert not torch.isnan(layer.weight.grad).any()

    def test_no_bias(self):
        """Layer without bias must work."""
        layer = TernaryLinear(32, 16, bias=False)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)
        assert layer.bias is None

    def test_sparsity_property(self):
        """Sparsity should be between 0 and 1."""
        layer = TernaryLinear(64, 32)
        layer.eval()
        s = layer.sparsity
        assert 0.0 <= s <= 1.0

    def test_cache_invalidation(self):
        """Invalidating cache should clear cached weights."""
        layer = TernaryLinear(32, 16)
        layer.eval()
        _ = layer(torch.randn(1, 32))  # triggers caching
        assert layer._cached_ternary is not None

        layer.invalidate_cache()
        assert layer._cached_ternary is None


# ═══════════════════════════════════════════════════════════════
# TernaryConv2d Tests
# ═══════════════════════════════════════════════════════════════


class TestTernaryConv2d:
    """Tests for ternary convolution layer."""

    def test_forward_shape(self):
        layer = TernaryConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        y = layer(x)
        assert y.shape == (2, 16, 32, 32)

    def test_eval_deterministic(self):
        layer = TernaryConv2d(3, 16, kernel_size=3, padding=1)
        layer.eval()
        x = torch.randn(1, 3, 16, 16)
        y1 = layer(x)
        y2 = layer(x)
        assert torch.equal(y1, y2)


# ═══════════════════════════════════════════════════════════════
# Sparse Module Tests
# ═══════════════════════════════════════════════════════════════


class TestSparse:
    """Tests for Patents 7-9: sparsity optimisation."""

    def test_bitmap_shape(self):
        weights = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
        bitmap = generate_sparsity_bitmap(weights)
        assert bitmap.shape == weights.shape

    def test_bitmap_correctness(self):
        weights = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
        bitmap = generate_sparsity_bitmap(weights)
        expected = torch.tensor([True, False, True, False, True])
        assert torch.equal(bitmap, expected)

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack must recover original ternary weights."""
        original = torch.tensor([[1.0, -1.0, 0.0, 1.0],
                                  [0.0, -1.0, 1.0, 0.0]])
        packed, bitmap = pack_ternary_weights(original)
        recovered = unpack_ternary_weights(packed, original.shape)
        assert torch.equal(original, recovered)

    def test_pack_unpack_large(self):
        """Roundtrip on larger tensor."""
        q = TernaryQuantizer(threshold=0.7)
        weights = torch.randn(64, 64)
        ternary, _ = q.quantize(weights)

        packed, bitmap = pack_ternary_weights(ternary)
        recovered = unpack_ternary_weights(packed, ternary.shape)
        assert torch.equal(ternary, recovered)

    def test_compression_ratio(self):
        """Packed size should be ~4x smaller than dense float32."""
        ternary = torch.zeros(1024)  # 4096 bytes as float32
        packed, _ = pack_ternary_weights(ternary)
        # 1024 weights / 4 per byte = 256 bytes
        assert packed.numel() == 256

    def test_sparsity_info(self):
        weights = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        info = sparsity_info(weights)
        assert info.total_weights == 8
        assert info.zero_weights == 5
        assert info.nonzero_weights == 3
        assert abs(info.sparsity_ratio - 5 / 8) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Inference Engine Tests
# ═══════════════════════════════════════════════════════════════


class SimpleModel(nn.Module):
    """Small model for testing conversion."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TestInferenceEngine:
    """Tests for the auto-conversion engine."""

    def test_convert_simple_model(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        report = engine.convert(model, sensitivity_analysis=False)

        assert report.converted_layers >= 1
        assert report.total_layers == 2
        assert report.compression_ratio > 1.0

    def test_inference_produces_output(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        x = torch.randn(4, 64)
        result = engine.infer(model, x)

        assert result.output.shape == (4, 10)
        assert result.latency_ms > 0
        assert result.deterministic is True

    def test_deterministic_inference(self):
        """Same input must produce identical output. Patent 36."""
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        x = torch.randn(4, 64)
        r1 = engine.infer(model, x)
        r2 = engine.infer(model, x)

        assert torch.equal(r1.output, r2.output)

    def test_conversion_report(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        report = engine.convert(model, sensitivity_analysis=False)

        assert report.total_params > 0
        assert report.ternary_params > 0
        assert report.original_size_mb > 0
        assert report.ternary_size_mb > 0


# ═══════════════════════════════════════════════════════════════
# Sensitivity Analyzer Tests
# ═══════════════════════════════════════════════════════════════


class TestSensitivityAnalyzer:
    """Tests for Patent 4: sensitivity analysis."""

    def test_analyze_layer(self):
        analyzer = SensitivityAnalyzer()
        weights = torch.randn(64, 32)
        result = analyzer.analyze_layer("test_layer", weights)

        assert result["name"] == "test_layer"
        assert len(result["results"]) == 5  # default 5 thresholds
        assert "recommended_threshold" in result
        assert "precision_critical" in result

    def test_analyze_model(self):
        model = SimpleModel()
        analyzer = SensitivityAnalyzer()
        analyses = analyzer.analyze_model(model)

        assert len(analyses) == 2  # two Linear layers
        for a in analyses:
            assert "name" in a
            assert "recommended_threshold" in a


# ═══════════════════════════════════════════════════════════════
# Memory Profile Tests
# ═══════════════════════════════════════════════════════════════


class TestMemoryProfile:
    """Tests for Patent 3: memory architecture."""

    def test_profile_converted_model(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        profile = profile_model_memory(model)
        assert profile.ternary_params > 0
        assert profile.compression_ratio > 1.0
        assert profile.total_bytes < profile.original_fp16_bytes


# ═══════════════════════════════════════════════════════════════
# .tern-model Format Tests
# ═══════════════════════════════════════════════════════════════


class TestTernModel:
    """Tests for Patent 17/22: .tern-model format."""

    def test_write_and_read_metadata(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        writer = TernModelWriter()
        meta = writer.save(model, path, source="test-model")

        reader = TernModelReader()
        loaded_meta = reader.read_metadata(path)

        assert loaded_meta["source"] == "test-model"
        assert loaded_meta["version"] == 1
        assert loaded_meta["num_layers"] > 0

        Path(path).unlink()

    def test_verify_checksum(self):
        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        writer = TernModelWriter()
        writer.save(model, path)

        reader = TernModelReader()
        assert reader.verify(path) is True

        # Corrupt the file
        with open(path, "r+b") as f:
            f.seek(20)
            f.write(b"\x00\x00\x00\x00")

        assert reader.verify(path) is False

        Path(path).unlink()


# ═══════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end: model → convert → save → verify → infer."""

    def test_full_pipeline(self):
        # 1. Create model
        model = SimpleModel()

        # 2. Convert to ternary
        engine = TernaryInferenceEngine()
        report = engine.convert(model, sensitivity_analysis=True)
        assert report.converted_layers >= 1

        # 3. Save to .tern-model
        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        writer = TernModelWriter()
        meta = writer.save(model, path, source="integration-test")
        assert meta["num_layers"] > 0

        # 4. Verify checksum
        reader = TernModelReader()
        assert reader.verify(path) is True

        # 5. Run inference
        x = torch.randn(2, 64)
        r1 = engine.infer(model, x)
        r2 = engine.infer(model, x)

        assert r1.output.shape == (2, 10)
        assert torch.equal(r1.output, r2.output)  # deterministic

        # 6. Memory profile
        profile = profile_model_memory(model)
        assert profile.compression_ratio > 1.0

        Path(path).unlink()
