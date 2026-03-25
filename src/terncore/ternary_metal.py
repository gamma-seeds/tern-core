#!/usr/bin/env python3
"""
ternary_metal.py — Python interface to the Metal ternary inference engine

Wraps libternary.dylib via ctypes for GPU-accelerated ternary matrix-vector
multiply on Apple Silicon.

Usage:
    engine = TernaryEngine()
    output = engine.matvec(packed_codes, scales, input_vec)

Terncore · Cubey/Synapticode · 2026
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------

_LIB_DIR = Path(__file__).parent / "csrc" / "metal" / "build"
_LIB_PATH = _LIB_DIR / "libternary.dylib"


def _load_lib():
    if not _LIB_PATH.exists():
        raise FileNotFoundError(
            f"libternary.dylib not found at {_LIB_PATH}\n"
            f"Run 'make' in {Path(__file__).parent} to build it."
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    # Engine lifecycle
    lib.tern_engine_create.restype = ctypes.c_void_p
    lib.tern_engine_create.argtypes = []

    lib.tern_engine_destroy.restype = None
    lib.tern_engine_destroy.argtypes = [ctypes.c_void_p]

    # Buffer management
    lib.tern_buffer_create.restype = ctypes.c_void_p
    lib.tern_buffer_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    lib.tern_buffer_destroy.restype = None
    lib.tern_buffer_destroy.argtypes = [ctypes.c_void_p]

    lib.tern_buffer_read.restype = None
    lib.tern_buffer_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    lib.tern_buffer_size.restype = ctypes.c_size_t
    lib.tern_buffer_size.argtypes = [ctypes.c_void_p]

    # Matvec
    lib.tern_matvec.restype = ctypes.c_int
    lib.tern_matvec.argtypes = [
        ctypes.c_void_p,  # engine
        ctypes.c_void_p,  # packed_codes
        ctypes.c_void_p,  # scales
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_uint32,  # M
        ctypes.c_uint32,  # K
        ctypes.c_uint32,  # B
    ]

    lib.tern_matvec_fast.restype = ctypes.c_int
    lib.tern_matvec_fast.argtypes = [
        ctypes.c_void_p,  # engine
        ctypes.c_void_p,  # packed_codes
        ctypes.c_void_p,  # scales
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_uint32,  # M
        ctypes.c_uint32,  # K
        ctypes.c_uint32,  # B
    ]

    # Sync and diagnostics
    lib.tern_sync.restype = None
    lib.tern_sync.argtypes = [ctypes.c_void_p]

    lib.tern_device_name.restype = ctypes.c_char_p
    lib.tern_device_name.argtypes = [ctypes.c_void_p]

    lib.tern_last_error.restype = ctypes.c_char_p
    lib.tern_last_error.argtypes = [ctypes.c_void_p]

    return lib


# ---------------------------------------------------------------------------
# GPU Buffer wrapper
# ---------------------------------------------------------------------------

class GPUBuffer:
    """Wrapper around a Metal GPU buffer."""

    def __init__(self, lib, engine, data: np.ndarray = None, size: int = 0):
        self._lib = lib
        self._engine = engine

        if data is not None:
            data = np.ascontiguousarray(data)
            self._handle = lib.tern_buffer_create(
                engine, data.ctypes.data, data.nbytes
            )
            self._size = data.nbytes
        elif size > 0:
            self._handle = lib.tern_buffer_create(engine, None, size)
            self._size = size
        else:
            raise ValueError("Must provide data or size")

        if not self._handle:
            raise RuntimeError("Failed to create GPU buffer")

    @property
    def handle(self):
        return self._handle

    def read(self, dtype=np.float16, shape=None):
        """Read buffer contents back to numpy array."""
        nbytes = self._size
        buf = np.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
        self._lib.tern_buffer_read(self._handle, buf.ctypes.data, nbytes)
        if shape is not None:
            buf = buf.reshape(shape)
        return buf

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.tern_buffer_destroy(self._handle)


# ---------------------------------------------------------------------------
# TernaryEngine
# ---------------------------------------------------------------------------

class TernaryEngine:
    """Metal ternary inference engine.

    Provides GPU-accelerated ternary matrix-vector multiply using packed
    2-bit codes. No dequantization, no floating-point multiplies on weights.
    """

    def __init__(self):
        self._lib = _load_lib()
        self._engine = self._lib.tern_engine_create()
        if not self._engine:
            raise RuntimeError("Failed to create Metal engine")

    @property
    def device_name(self) -> str:
        return self._lib.tern_device_name(self._engine).decode()

    @property
    def last_error(self) -> str:
        return self._lib.tern_last_error(self._engine).decode()

    def create_buffer(self, data: np.ndarray = None, size: int = 0) -> GPUBuffer:
        """Allocate a GPU buffer, optionally initialized with data."""
        return GPUBuffer(self._lib, self._engine, data=data, size=size)

    def matvec(self, packed_codes: np.ndarray, scales: np.ndarray,
               input_vec: np.ndarray, fast: bool = True) -> np.ndarray:
        """Compute ternary matrix-vector multiply on GPU.

        Args:
            packed_codes: uint32 array (M, packed_K) — packed 2-bit ternary codes
            scales: float32 array (M,) — per-channel scales
            input_vec: float16 array (B, K) or (K,) — input vector(s)
            fast: use vectorized kernel variant (requires K % 16 == 0)

        Returns:
            float16 array (B, M) — output
        """
        if input_vec.ndim == 1:
            input_vec = input_vec.reshape(1, -1)

        B, K = input_vec.shape
        M = packed_codes.shape[0]

        # Ensure contiguous and correct dtypes
        packed_codes = np.ascontiguousarray(packed_codes, dtype=np.uint32)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        input_vec = np.ascontiguousarray(input_vec, dtype=np.float16)

        # Allocate GPU buffers
        codes_buf = self.create_buffer(packed_codes)
        scales_buf = self.create_buffer(scales)
        input_buf = self.create_buffer(input_vec)
        output_buf = self.create_buffer(size=B * M * 2)  # float16

        # Dispatch
        dispatch_fn = self._lib.tern_matvec_fast if (fast and K % 16 == 0) else self._lib.tern_matvec
        rc = dispatch_fn(
            self._engine,
            codes_buf.handle, scales_buf.handle,
            input_buf.handle, output_buf.handle,
            M, K, B,
        )

        if rc != 0:
            raise RuntimeError(f"Metal kernel failed: {self.last_error}")

        # Read back result
        result = output_buf.read(dtype=np.float16, shape=(B, M))
        return result

    def matvec_gpu(self, codes_buf: GPUBuffer, scales_buf: GPUBuffer,
                   input_buf: GPUBuffer, output_buf: GPUBuffer,
                   M: int, K: int, B: int, fast: bool = True) -> None:
        """Dispatch ternary matvec with pre-allocated GPU buffers (zero-copy).

        Use this for benchmarking to avoid host↔GPU transfer overhead.
        """
        dispatch_fn = self._lib.tern_matvec_fast if (fast and K % 16 == 0) else self._lib.tern_matvec
        rc = dispatch_fn(
            self._engine,
            codes_buf.handle, scales_buf.handle,
            input_buf.handle, output_buf.handle,
            M, K, B,
        )
        if rc != 0:
            raise RuntimeError(f"Metal kernel failed: {self.last_error}")

    def sync(self):
        """Wait for GPU to finish."""
        self._lib.tern_sync(self._engine)

    def __del__(self):
        if hasattr(self, '_engine') and self._engine:
            self._lib.tern_engine_destroy(self._engine)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test():
    """Correctness test: compare Metal kernel output against numpy reference."""
    from pack_weights import pack_ternary_codes

    print("Ternary Metal Engine — Correctness Test")
    print("=" * 50)

    engine = TernaryEngine()
    print(f"Device: {engine.device_name}")

    test_cases = [
        (64, 128, 1),
        (256, 256, 1),
        (2048, 2048, 1),
        (4096, 4096, 1),
        (4096, 11008, 1),
        (2048, 2048, 4),
    ]

    all_passed = True

    for M, K, B in test_cases:
        # Generate random ternary codes and input
        codes = np.random.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
        # ~43% sparsity to match real distribution
        mask = np.random.random((M, K)) < 0.43
        codes[mask] = 0

        scales = np.random.randn(M).astype(np.float32) * 0.1 + 0.5
        input_vec = np.random.randn(B, K).astype(np.float16)

        # Pack codes
        import torch
        packed = pack_ternary_codes(torch.from_numpy(codes))

        # Metal kernel
        result = engine.matvec(packed, scales, input_vec, fast=(K % 16 == 0))

        # Numpy reference: y[b, m] = scale[m] * sum_k(code[m,k] * x[b,k])
        ref = (codes.astype(np.float32) @ input_vec.astype(np.float32).T).T
        ref = ref * scales[np.newaxis, :]
        ref = ref.astype(np.float16)

        # Compare (allow half-precision tolerance)
        max_err = np.max(np.abs(result.astype(np.float32) - ref.astype(np.float32)))
        rel_err = max_err / (np.max(np.abs(ref.astype(np.float32))) + 1e-8)
        passed = rel_err < 0.02  # 2% relative tolerance for fp16

        status = "PASS" if passed else "FAIL"
        print(f"  ({M:>5}, {K:>5}) B={B}: max_err={max_err:.6f}, "
              f"rel_err={rel_err:.6f} [{status}]")

        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    if '--test' in sys.argv:
        _test()
    else:
        # Quick demo
        engine = TernaryEngine()
        print(f"Ternary Metal Engine ready on: {engine.device_name}")
        print("Run with --test for correctness verification.")
