#!/usr/bin/env python3
"""
pack_weights.py — Pack int8 ternary codes into 2-bit format for Metal kernel

Encoding (per uint32, 16 ternary values):
  bits [2i+1 : 2i] for value i (i = 0..15)
  0b00 = zero weight
  0b01 = +1
  0b10 = -1

Terncore · Cubey/Synapticode · 2026
"""

import numpy as np
import torch


def pack_ternary_codes(codes: torch.Tensor) -> np.ndarray:
    """Pack int8 ternary codes {-1, 0, +1} into 2-bit packed uint32 array.

    Args:
        codes: int8 tensor of shape (M, K) with values in {-1, 0, +1}

    Returns:
        uint32 numpy array of shape (M, ceil(K/16))
    """
    assert codes.ndim == 2, f"Expected 2D tensor, got {codes.ndim}D"
    codes_np = codes.cpu().numpy().astype(np.int8)
    M, K = codes_np.shape
    packed_K = (K + 15) // 16

    # Pad K to multiple of 16
    if K % 16 != 0:
        pad_width = 16 - (K % 16)
        codes_np = np.pad(codes_np, ((0, 0), (0, pad_width)),
                          mode='constant', constant_values=0)

    # Map: -1 → 0b10, 0 → 0b00, +1 → 0b01
    # Using: positive_bit = (code == 1), negative_bit = (code == -1)
    pos = (codes_np == 1).astype(np.uint32)   # bit 0
    neg = (codes_np == -1).astype(np.uint32)  # bit 1
    two_bit = pos | (neg << 1)  # 2-bit encoding per value

    # Reshape to (M, packed_K, 16) and pack 16 values into each uint32
    two_bit = two_bit.reshape(M, packed_K, 16)
    packed = np.zeros((M, packed_K), dtype=np.uint32)
    for i in range(16):
        packed |= two_bit[:, :, i] << (i * 2)

    return packed


def unpack_ternary_codes(packed: np.ndarray, K: int) -> np.ndarray:
    """Unpack 2-bit packed uint32 array back to int8 codes.

    Args:
        packed: uint32 array of shape (M, packed_K)
        K: original input feature dimension

    Returns:
        int8 numpy array of shape (M, K)
    """
    M, packed_K = packed.shape
    codes = np.zeros((M, packed_K * 16), dtype=np.int8)

    for i in range(16):
        two_bit = (packed >> (i * 2)) & 0x3
        # 0b01 → +1, 0b10 → -1, 0b00 → 0
        pos = (two_bit == 1).astype(np.int8)
        neg = (two_bit == 2).astype(np.int8)
        codes[:, i::16] = pos - neg

    # Handle interleaved storage: values are packed sequentially within each group of 16
    # Reorder: the i-th value in each group is at position group*16 + i
    codes_reordered = np.zeros_like(codes)
    for i in range(16):
        codes_reordered[:, i::16] = codes[:, i::16]

    return codes_reordered[:, :K]


def pack_model_weights(model) -> dict:
    """Pack all ternary-quantized Linear layers from a model.

    Expects each Linear module to have _tern_codes (int8) and _tern_scales (fp32).

    Returns:
        Dict mapping layer name to {packed_codes: np.ndarray, scales: np.ndarray,
                                     M: int, K: int}
    """
    packed_layers = {}

    for name, module in model.named_modules():
        if hasattr(module, '_tern_codes'):
            codes = module._tern_codes.cpu()
            scales = module._tern_scales.cpu().numpy().astype(np.float32)
            M, K = codes.shape

            packed = pack_ternary_codes(codes)
            packed_layers[name] = {
                'packed_codes': packed,
                'scales': scales,
                'M': M,
                'K': K,
            }

    return packed_layers


def compute_compression_stats(codes: torch.Tensor) -> dict:
    """Compute compression statistics for a ternary code matrix."""
    M, K = codes.shape
    packed_K = (K + 15) // 16

    fp16_bytes = M * K * 2
    int8_bytes = M * K * 1
    packed_bytes = M * packed_K * 4

    zeros = (codes == 0).sum().item()
    sparsity = zeros / codes.numel()

    return {
        'fp16_bytes': fp16_bytes,
        'int8_bytes': int8_bytes,
        'packed_2bit_bytes': packed_bytes,
        'compression_vs_fp16': fp16_bytes / packed_bytes,
        'compression_vs_int8': int8_bytes / packed_bytes,
        'sparsity': sparsity,
        'effective_bits': packed_bytes * 8 / (M * K),
    }


if __name__ == '__main__':
    # Self-test
    print("Testing pack/unpack round-trip...")

    for M, K in [(64, 128), (2048, 2048), (4096, 11008), (100, 37)]:
        codes = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        packed = pack_ternary_codes(codes)

        assert packed.shape == (M, (K + 15) // 16), \
            f"Shape mismatch: {packed.shape} vs ({M}, {(K+15)//16})"

        unpacked = unpack_ternary_codes(packed, K)
        codes_np = codes.numpy()
        assert np.array_equal(codes_np, unpacked), \
            f"Round-trip failed for ({M}, {K})"

        stats = compute_compression_stats(codes)
        print(f"  ({M:>5}, {K:>5}): packed {packed.shape[1]:>4} uint32s/row, "
              f"{stats['compression_vs_fp16']:.1f}x vs FP16, "
              f"{stats['sparsity']:.1%} sparse")

    print("All tests passed.")
