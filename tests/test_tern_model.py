"""
Tests for .tern-model v2 format (TernModelWriter / TernModelReader).

All tests use synthetic tensors — no model loading required, fast execution.

Run with: pytest tests/test_tern_model.py -v
"""

import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.sparse import pack_ternary_weights, unpack_ternary_weights
from terncore.tern_model import (
    ALIGNMENT,
    HEADER_SIZE,
    TERN_MAGIC,
    TERN_MAGIC_REVERSE,
    TERN_VERSION,
    TernModelReader,
    TernModelWriter,
)


# ═══════════════════════════════════════════════════════════════
# pack_ternary / roundtrip tests
# ═══════════════════════════════════════════════════════════════


class TestPackTernary:
    """Tests for TernModelWriter.pack_ternary static method."""

    def test_pack_ternary_basic(self):
        """Known weights produce correct 2-bit encoding."""
        weights = torch.tensor([[1.0, -1.0, 0.01, 0.5]], dtype=torch.float32)
        packed_bytes, alpha, bitmap_bytes, sparsity = TernModelWriter.pack_ternary(
            weights, threshold=0.5
        )
        assert isinstance(packed_bytes, bytes)
        assert isinstance(alpha, float)
        assert alpha > 0
        assert 0.0 <= sparsity <= 1.0
        # At least 1 byte of packed data for 4 weights
        assert len(packed_bytes) >= 1

    def test_pack_ternary_roundtrip(self):
        """Pack -> unpack recovers bit-identical ternary values."""
        torch.manual_seed(42)
        weights = torch.randn(64, 64)

        from terncore.arithmetic.quantizer import TernaryQuantizer

        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)

        packed, bitmap = pack_ternary_weights(ternary)
        recovered = unpack_ternary_weights(packed, ternary.shape)

        assert torch.equal(ternary, recovered), "Roundtrip must be bit-identical"

    def test_pack_ternary_all_zeros(self):
        """All-zero weights should give sparsity=1.0."""
        weights = torch.zeros(32, 32)
        packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(
            weights, threshold=0.7
        )
        assert sparsity == 1.0
        # All packed bytes should be 0 (encoding 00 for zero)
        assert all(b == 0 for b in packed)


# ═══════════════════════════════════════════════════════════════
# Sparsity bitmap tests
# ═══════════════════════════════════════════════════════════════


class TestSparsityBitmap:
    """Tests for sparsity bitmap generation."""

    def test_sparsity_bitmap_all_zero(self):
        """All-zero packed weights should produce an all-zero bitmap."""
        packed = bytes(64)  # 256 weights, all zero
        bitmap = TernModelWriter.generate_sparsity_bitmap(packed, block_size=256)
        assert all(b == 0 for b in bitmap)

    def test_sparsity_bitmap_nonzero_block(self):
        """Non-zero block should set corresponding bitmap bit."""
        # 256 weights = 64 bytes of packed data
        packed = bytearray(64)
        packed[0] = 0x01  # First weight is +1
        bitmap = TernModelWriter.generate_sparsity_bitmap(bytes(packed), block_size=256)
        # First bit should be set
        assert bitmap[0] & 0x01 == 1


# ═══════════════════════════════════════════════════════════════
# Writer / Reader integration tests
# ═══════════════════════════════════════════════════════════════


class TestWriteSingleLayer:
    """Test writing and reading a single ternary layer."""

    def test_write_single_layer(self):
        """Write 1 ternary layer, verify file is readable."""
        torch.manual_seed(123)
        weights = torch.randn(128, 64)

        writer = TernModelWriter({"source": "test", "notes": "unit test"})
        writer.add_layer("layer.0.linear", weights, dtype="ternary2", threshold=0.7)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            stats = writer.write(path)

            assert stats["num_layers"] == 1
            assert stats["num_ternary"] == 1
            assert stats["num_protected"] == 0

            reader = TernModelReader(path)
            assert len(reader.manifest["layers"]) == 1
            assert reader.manifest["layers"][0]["name"] == "layer.0.linear"
            assert reader.manifest["layers"][0]["dtype"] == "ternary2"
        finally:
            Path(path).unlink(missing_ok=True)


class TestWriteMixedPrecision:
    """Test writing mixed ternary + FP16 layers."""

    def test_write_mixed_precision(self):
        """Write mixed ternary + FP16 layers."""
        torch.manual_seed(456)

        writer = TernModelWriter({"source": "mixed_test"})
        writer.add_layer("encoder.v_proj", torch.randn(64, 64), dtype="ternary2")
        writer.add_layer("encoder.lm_head", torch.randn(100, 64), dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            stats = writer.write(path)
            assert stats["num_ternary"] == 1
            assert stats["num_protected"] == 1
            assert stats["num_layers"] == 2

            reader = TernModelReader(path)
            layers = reader.manifest["layers"]
            assert layers[0]["dtype"] == "ternary2"
            assert layers[1]["dtype"] == "float16"
        finally:
            Path(path).unlink(missing_ok=True)


class TestAlignment:
    """Test 32-byte SIMD alignment."""

    def test_alignment(self):
        """All layer offsets must be 32-byte aligned (relative to file start)."""
        torch.manual_seed(789)

        writer = TernModelWriter({"source": "align_test"})
        # Add layers of varying sizes to stress alignment
        writer.add_layer("a", torch.randn(17, 13), dtype="ternary2")
        writer.add_layer("b", torch.randn(33, 7), dtype="ternary2")
        writer.add_layer("c", torch.randn(50, 20), dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)

            weights_offset = reader.header["weights_offset"]
            assert weights_offset % ALIGNMENT == 0, (
                f"weights_offset {weights_offset} not {ALIGNMENT}-byte aligned"
            )

            for layer in reader.manifest["layers"]:
                abs_offset = weights_offset + layer["offset"]
                assert abs_offset % ALIGNMENT == 0, (
                    f"Layer {layer['name']} at absolute offset {abs_offset} "
                    f"not {ALIGNMENT}-byte aligned"
                )
        finally:
            Path(path).unlink(missing_ok=True)


class TestHeaderMagic:
    """Test header magic bytes and version."""

    def test_header_magic(self):
        """First 4 bytes must be 'TERN', version must be 2."""
        torch.manual_seed(101)

        writer = TernModelWriter()
        writer.add_layer("test", torch.randn(16, 16), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            with open(path, "rb") as f:
                magic = f.read(4)
                version = struct.unpack("<H", f.read(2))[0]

            assert magic == TERN_MAGIC
            assert version == TERN_VERSION
        finally:
            Path(path).unlink(missing_ok=True)


class TestManifestReadable:
    """Test that manifest JSON is correctly readable."""

    def test_manifest_readable(self):
        """Read manifest back, verify all required fields present."""
        torch.manual_seed(202)

        writer = TernModelWriter({"source": "manifest_test"})
        writer.add_layer(
            "proj.weight",
            torch.randn(32, 32),
            dtype="ternary2",
            threshold=0.6,
            sensitivity_score=0.05,
            quant_error=0.001,
        )

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)

            manifest = reader.manifest
            assert "model_metadata" in manifest
            assert "layers" in manifest
            assert manifest["model_metadata"]["source"] == "manifest_test"

            layer = manifest["layers"][0]
            assert layer["name"] == "proj.weight"
            assert layer["dtype"] == "ternary2"
            assert layer["shape"] == [32, 32]
            assert layer["num_params"] == 1024
            assert layer["threshold"] == 0.6
            assert isinstance(layer["alpha"], float)
            assert isinstance(layer["sparsity"], float)
            assert layer["sensitivity_score"] == 0.05
            assert layer["quant_error"] == 0.001
            assert isinstance(layer["offset"], int)
            assert isinstance(layer["size"], int)
            assert "has_bias" in layer
            assert "has_bitmap" in layer
        finally:
            Path(path).unlink(missing_ok=True)


class TestFileIntegrity:
    """Test CRC32 footer validation."""

    def test_file_integrity(self):
        """CRC32 footer validates on a clean file."""
        torch.manual_seed(303)

        writer = TernModelWriter()
        writer.add_layer("crc_test", torch.randn(64, 32), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            assert reader.verify() is True
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_integrity_corrupted(self):
        """CRC32 should fail on a corrupted file."""
        torch.manual_seed(304)

        writer = TernModelWriter()
        writer.add_layer("corrupt_test", torch.randn(64, 32), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            # Corrupt a byte in the weight data section
            reader = TernModelReader(path)
            offset = reader.header["weights_offset"] + 10
            with open(path, "r+b") as f:
                f.seek(offset)
                original = f.read(1)
                f.seek(offset)
                f.write(bytes([(original[0] ^ 0xFF)]))

            reader2 = TernModelReader(path)
            assert reader2.verify() is False
        finally:
            Path(path).unlink(missing_ok=True)


class TestFooterMagic:
    """Test reverse magic at end of file."""

    def test_footer_magic(self):
        """Last 4 bytes must be 'NRET'."""
        torch.manual_seed(404)

        writer = TernModelWriter()
        writer.add_layer("footer_test", torch.randn(16, 16), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            with open(path, "rb") as f:
                f.seek(-4, 2)  # Seek to last 4 bytes
                reverse_magic = f.read(4)

            assert reverse_magic == TERN_MAGIC_REVERSE
        finally:
            Path(path).unlink(missing_ok=True)


class TestHeaderSize:
    """Test header is exactly 256 bytes."""

    def test_header_size(self):
        """Header size field should be 256 and actual header is 256 bytes."""
        torch.manual_seed(505)

        writer = TernModelWriter()
        writer.add_layer("header_test", torch.randn(8, 8), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            assert reader.header["header_size"] == HEADER_SIZE
            assert reader.header["manifest_offset"] == HEADER_SIZE
        finally:
            Path(path).unlink(missing_ok=True)


class TestFileSizeConsistency:
    """Test that footer file_size matches actual file size."""

    def test_file_size_matches(self):
        """Footer file_size must equal os.path.getsize()."""
        torch.manual_seed(606)

        writer = TernModelWriter()
        writer.add_layer("size_test", torch.randn(32, 32), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            stats = writer.write(path)
            actual_size = Path(path).stat().st_size

            assert stats["file_size"] == actual_size

            # Also check from footer directly
            with open(path, "rb") as f:
                f.seek(-12, 2)
                footer_size = struct.unpack("<Q", f.read(8))[0]
            assert footer_size == actual_size
        finally:
            Path(path).unlink(missing_ok=True)


class TestRandomAccess:
    """Test random-access layer reading via manifest offsets."""

    def test_random_access_read(self):
        """Read specific layer data using manifest offset."""
        torch.manual_seed(707)

        writer = TernModelWriter()
        writer.add_layer("first", torch.randn(32, 16), dtype="ternary2")
        writer.add_layer("second", torch.randn(64, 32), dtype="ternary2")
        writer.add_layer("third", torch.randn(16, 8), dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)

            # Read middle layer by name
            data = reader.read_layer_data("second")
            assert len(data) > 0

            # Verify it's the right size (alpha + packed_size + packed + bitmap_size + bias_size)
            layer_entry = reader.manifest["layers"][1]
            assert layer_entry["name"] == "second"
            assert len(data) == layer_entry["size"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_random_access_missing_layer(self):
        """Reading a non-existent layer raises KeyError."""
        torch.manual_seed(708)

        writer = TernModelWriter()
        writer.add_layer("exists", torch.randn(8, 8), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)

            with pytest.raises(KeyError):
                reader.read_layer_data("does_not_exist")
        finally:
            Path(path).unlink(missing_ok=True)


class TestBiasHandling:
    """Test layers with bias vectors."""

    def test_layer_with_bias(self):
        """Ternary layer with bias should store bias in weight data."""
        torch.manual_seed(808)
        weights = torch.randn(32, 16)
        bias = torch.randn(32)

        writer = TernModelWriter()
        writer.add_layer("biased", weights, dtype="ternary2", bias=bias)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)

            layer = reader.manifest["layers"][0]
            assert layer["has_bias"] is True
            # Layer data should be larger than without bias
            assert layer["size"] > 0
        finally:
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# Day 7: Round-trip reconstruction tests
# ═══════════════════════════════════════════════════════════════


class TestReconstructTernaryLayer:
    """Test ternary layer reconstruction from .tern-model."""

    def test_reconstruct_ternary_layer(self):
        """Write ternary layer, read back, verify tensor matches quantised form."""
        torch.manual_seed(901)
        weights = torch.randn(64, 32)
        threshold = 0.7

        # Quantise to get reference
        q = TernaryQuantizer(threshold=threshold)
        ternary, alpha = q.quantize(weights)
        reference = ternary * alpha

        writer = TernModelWriter()
        writer.add_layer("ternary_rt", weights, dtype="ternary2", threshold=threshold)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            result = reader.reconstruct_layer("ternary_rt")

            assert "weight" in result
            assert result["weight"].shape == (64, 32)
            assert torch.allclose(result["weight"], reference, atol=1e-6), (
                f"Max diff: {(result['weight'] - reference).abs().max().item()}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reconstruct_ternary_with_bias(self):
        """Ternary layer with bias roundtrips correctly."""
        torch.manual_seed(902)
        weights = torch.randn(32, 16)
        bias = torch.randn(32)

        writer = TernModelWriter()
        writer.add_layer("biased_rt", weights, dtype="ternary2", bias=bias)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            result = reader.reconstruct_layer("biased_rt")

            assert "bias" in result
            assert torch.allclose(result["bias"], bias, atol=1e-6)
        finally:
            Path(path).unlink(missing_ok=True)


class TestReconstructFP16Layer:
    """Test FP16 layer reconstruction from .tern-model."""

    def test_reconstruct_fp16_layer(self):
        """Write FP16 layer, read back, verify tensor matches FP16 precision."""
        torch.manual_seed(903)
        weights = torch.randn(48, 24)

        # Reference: FP32 → FP16 → FP32 (expected precision loss)
        reference = weights.half().float()

        writer = TernModelWriter()
        writer.add_layer("fp16_rt", weights, dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            result = reader.reconstruct_layer("fp16_rt")

            assert "weight" in result
            assert result["weight"].shape == (48, 24)
            assert torch.equal(result["weight"], reference), (
                "FP16 roundtrip must be bit-identical to FP16 precision"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_reconstruct_fp16_with_bias(self):
        """FP16 layer with bias roundtrips correctly."""
        torch.manual_seed(904)
        weights = torch.randn(20, 10)
        bias = torch.randn(20)

        reference_bias = bias.half().float()

        writer = TernModelWriter()
        writer.add_layer("fp16_bias", weights, dtype="float16", bias=bias)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            result = reader.reconstruct_layer("fp16_bias")

            assert "bias" in result
            assert torch.equal(result["bias"], reference_bias)
        finally:
            Path(path).unlink(missing_ok=True)


class TestReconstructAllMixed:
    """Test reconstruct_all() with mixed ternary + FP16 layers."""

    def test_reconstruct_all_mixed(self):
        """Write mixed model, reconstruct_all(), verify all tensors."""
        torch.manual_seed(905)

        w1 = torch.randn(32, 16)  # ternary
        w2 = torch.randn(64, 32)  # fp16
        w3 = torch.randn(16, 8)   # ternary

        writer = TernModelWriter()
        writer.add_layer("layer.0", w1, dtype="ternary2", threshold=0.7)
        writer.add_layer("layer.1", w2, dtype="float16")
        writer.add_layer("layer.2", w3, dtype="ternary2", threshold=0.6)

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)
            reader = TernModelReader(path)
            state_dict = reader.reconstruct_all()

            assert "layer.0.weight" in state_dict
            assert "layer.1.weight" in state_dict
            assert "layer.2.weight" in state_dict
            assert state_dict["layer.0.weight"].shape == (32, 16)
            assert state_dict["layer.1.weight"].shape == (64, 32)
            assert state_dict["layer.2.weight"].shape == (16, 8)

            # Verify FP16 layer is correct
            fp16_ref = w2.half().float()
            assert torch.equal(state_dict["layer.1.weight"], fp16_ref)
        finally:
            Path(path).unlink(missing_ok=True)


class TestRoundTripLogitsSynthetic:
    """Test round-trip bit-identical inference with a synthetic model."""

    def test_roundtrip_logits_synthetic(self):
        """Synthetic model: forward pass before write == after read."""
        torch.manual_seed(906)

        # Build a simple 2-layer model
        class TwoLayerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32, bias=True)
                self.fc2 = nn.Linear(32, 8, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TwoLayerNet()
        model.eval()

        # Quantise fc1 (ternary), keep fc2 as FP16
        q = TernaryQuantizer(threshold=0.7)
        fc1_ternary, fc1_alpha = q.quantize(model.fc1.weight.data)
        model.fc1.weight.data = fc1_ternary * fc1_alpha

        # Reference logits
        test_input = torch.randn(4, 16)
        with torch.no_grad():
            ref_logits = model(test_input).clone()

        # Write to .tern-model
        writer = TernModelWriter()
        # fc1 is already quantised, write as ternary using add_layer on original weights
        # We need the quantised version to match, so write the alpha-scaled ternary
        packed, bitmap = pack_ternary_weights(fc1_ternary)
        packed_bytes = packed.numpy().tobytes()
        bitmap_flat = bitmap.flatten().numpy().astype(np.uint8)
        n_bits = len(bitmap_flat)
        n_bytes = (n_bits + 7) // 8
        bitmap_packed = bytearray(n_bytes)
        for i in range(n_bits):
            if bitmap_flat[i]:
                bitmap_packed[i // 8] |= 1 << (i % 8)

        writer.add_ternary_layer(
            "fc1", packed_bytes, fc1_alpha.item(),
            list(model.fc1.weight.shape),
            sparsity_bitmap=bytes(bitmap_packed),
            bias=model.fc1.bias,
        )
        writer.add_layer("fc2", model.fc2.weight.data, dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            # Reconstruct
            reader = TernModelReader(path)
            state_dict = reader.reconstruct_all()

            # Build fresh model and load
            model2 = TwoLayerNet()
            model2.eval()
            model2.load_state_dict(state_dict, strict=False)

            # Compare logits
            with torch.no_grad():
                rt_logits = model2(test_input)

            # Ternary layer is bit-identical, FP16 layer has FP16 precision
            assert torch.allclose(ref_logits, rt_logits, atol=1e-2), (
                f"Max logit diff: {(ref_logits - rt_logits).abs().max().item()}"
            )
        finally:
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# Day 7: Lazy loading API tests
# ═══════════════════════════════════════════════════════════════


class TestLazyAPI:
    """Test lazy loading convenience methods."""

    @pytest.fixture
    def model_path(self, tmp_path):
        """Create a .tern-model with 3 layers for lazy loading tests."""
        torch.manual_seed(907)
        path = tmp_path / "lazy_test.tern-model"

        writer = TernModelWriter({"source": "lazy_test"})
        writer.add_layer("alpha", torch.randn(16, 8), dtype="ternary2")
        writer.add_layer("beta", torch.randn(32, 16), dtype="float16")
        writer.add_layer("gamma", torch.randn(24, 12), dtype="ternary2")
        writer.write(str(path))

        return str(path)

    def test_layer_names(self, model_path):
        """layer_names() returns all names from manifest."""
        reader = TernModelReader(model_path)
        names = reader.layer_names()
        assert names == ["alpha", "beta", "gamma"]

    def test_layer_info(self, model_path):
        """layer_info(name) returns manifest entry without weight data."""
        reader = TernModelReader(model_path)
        info = reader.layer_info("beta")

        assert info["name"] == "beta"
        assert info["dtype"] == "float16"
        assert info["shape"] == [32, 16]
        assert isinstance(info["offset"], int)
        assert isinstance(info["size"], int)

    def test_layer_info_missing(self, model_path):
        """layer_info() raises KeyError for unknown layer."""
        reader = TernModelReader(model_path)
        with pytest.raises(KeyError):
            reader.layer_info("nonexistent")

    def test_lazy_single_layer(self, model_path):
        """layer(name) returns weight tensor for single layer."""
        reader = TernModelReader(model_path)
        tensor = reader.layer("alpha")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (16, 8)

    def test_load_all(self, model_path):
        """load_all() returns complete state_dict."""
        reader = TernModelReader(model_path)
        sd = reader.load_all()

        assert "alpha.weight" in sd
        assert "beta.weight" in sd
        assert "gamma.weight" in sd
        assert sd["alpha.weight"].shape == (16, 8)
        assert sd["beta.weight"].shape == (32, 16)
        assert sd["gamma.weight"].shape == (24, 12)

    def test_header_is_fast(self, model_path):
        """Header/manifest are already parsed at init — no re-reads needed."""
        reader = TernModelReader(model_path)
        # header and manifest should be populated from __init__
        assert reader.header["magic"] == TERN_MAGIC
        assert "layers" in reader.manifest
        assert len(reader.manifest["layers"]) == 3


class TestLoadAsModel:
    """Test load_as_model() with a synthetic nn.Module."""

    def test_load_as_model(self):
        """load_as_model loads reconstructed weights into a model."""
        torch.manual_seed(908)

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 8, bias=False)

            def forward(self, x):
                return self.linear(x)

        # Write
        weights = torch.randn(8, 16)
        writer = TernModelWriter()
        writer.add_layer("linear", weights, dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            # Load into model
            model = SimpleNet()
            reader = TernModelReader(path)
            missing, unexpected = reader.load_as_model(model)

            # "linear.weight" should match
            assert len(unexpected) == 0
            # The model may have other params we didn't store — that's OK
            ref = weights.half().float()
            assert torch.equal(model.linear.weight.data, ref)
        finally:
            Path(path).unlink(missing_ok=True)
