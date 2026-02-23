/*
 * bindings.c — ctypes interface for ternary compute kernels
 *
 * Top-level dispatch layer loaded by the Python accel/ wrapper via
 * ctypes.  Provides:
 *
 *   1. ternary_matmul_f32()      — Primary entry point: selects the
 *                                  best scalar kernel automatically.
 *   2. ternary_matmul_f32_simd() — SIMD-accelerated entry point
 *                                  (Phase 2 stub, falls back to scalar).
 *   3. get_simd_support()        — Runtime SIMD feature detection.
 *   4. terncore_version()        — Library version string.
 *
 * All individual kernel functions (tern_matvec_f32, tern_packed_*,
 * tern_sparse64_*) are also available as exported symbols for direct
 * use from Python when fine-grained control is needed.
 *
 * Build (shared library):
 *   macOS:  cc -std=c11 -O2 -shared -fPIC -o libterncore.dylib *.c
 *   Linux:  cc -std=c11 -O2 -shared -fPIC -o libterncore.so *.c
 *
 * Patent 36: Deterministic execution.
 * Patent 38: Configurable precision → dual-path dispatch.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "sparse_skip.h"   /* includes ternary_packed.h → ternary_matmul.h */

/* ── SIMD capability flags ────────────────────────────────────────── */

#define TERN_SIMD_SCALAR  (1u << 0)   /* Scalar C (always available) */
#define TERN_SIMD_AVX2    (1u << 1)   /* x86 AVX2 (Phase 2) */
#define TERN_SIMD_AVX512  (1u << 2)   /* x86 AVX-512 (Phase 2) */
#define TERN_SIMD_NEON    (1u << 3)   /* ARM NEON (Phase 2) */

/* ══════════════════════════════════════════════════════════════════════
 * ternary_matmul_f32 — Primary dispatch entry point
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Selects the best available scalar kernel based on inputs:
 *
 *   bitmap != NULL  →  sparse64 packed kernel (bit-scan skip)
 *   bitmap == NULL  →  dense packed kernel (byte-level skip)
 *
 * Phase 2 will add SIMD kernel selection via get_simd_support().
 *
 * Parameters:
 *   packed_weights  [M * N/4] uint8_t, 2-bit packed ternary weights
 *   input           [B x N]   float32 row-major input activations
 *   output          [B x M]   float32 row-major (caller-allocated)
 *   bitmap          packed sparsity bitmap (ceil(M*N/8) bytes), or
 *                   NULL to skip bitmap-based optimization
 *   alpha           per-layer scaling factor
 *   bias            [M] float32 bias, or NULL
 *   M               output dimension (weight rows)
 *   N               input dimension (weight columns, multiple of 4)
 *   B               batch size
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 *
 * Patent 38: Configurable precision — dispatch to best available
 *            execution path.
 * ═════════════════════════════════════════════════════════════════════*/
int ternary_matmul_f32(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B)
{
    if (bitmap != NULL) {
        return tern_sparse64_packed_matmul_f32(
            packed_weights, input, output, bitmap, M, N, B, alpha, bias);
    }

    return tern_packed_matmul_f32(
        packed_weights, input, output, M, N, B, alpha, bias);
}

/* ══════════════════════════════════════════════════════════════════════
 * ternary_matmul_f32_simd — SIMD-accelerated dispatch
 *
 * Phase 2 entry point.  When SIMD kernels are available, this will
 * dispatch to AVX2/AVX-512/NEON implementations.  Currently falls
 * back to the scalar dispatch in ternary_matmul_f32().
 *
 * Same parameters and return values as ternary_matmul_f32().
 * ═════════════════════════════════════════════════════════════════════*/
int ternary_matmul_f32_simd(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B)
{
    /*
     * Phase 2 will add:
     *   uint32_t caps = get_simd_support();
     *   if (caps & TERN_SIMD_AVX512) return avx512_dispatch(...);
     *   if (caps & TERN_SIMD_AVX2)   return avx2_dispatch(...);
     *   if (caps & TERN_SIMD_NEON)   return neon_dispatch(...);
     */
    return ternary_matmul_f32(
        packed_weights, input, output, bitmap, alpha, bias, M, N, B);
}

/* ══════════════════════════════════════════════════════════════════════
 * get_simd_support — Runtime SIMD feature detection
 *
 * Returns a bitmask of available instruction sets:
 *   TERN_SIMD_SCALAR  (0x01) — always set
 *   TERN_SIMD_AVX2    (0x02) — Phase 2: CPUID check
 *   TERN_SIMD_AVX512  (0x04) — Phase 2: CPUID check
 *   TERN_SIMD_NEON    (0x08) — Phase 2: auxiliary vector check
 *
 * Patent 38: Configurable precision — runtime capability detection.
 * ═════════════════════════════════════════════════════════════════════*/
uint32_t get_simd_support(void)
{
    uint32_t support = TERN_SIMD_SCALAR;

    /*
     * Phase 2 will add runtime detection:
     *
     * #if defined(__x86_64__) || defined(_M_X64)
     *     uint32_t eax, ebx, ecx, edx;
     *     __cpuid_count(7, 0, eax, ebx, ecx, edx);
     *     if (ebx & (1u << 5))  support |= TERN_SIMD_AVX2;
     *     if (ebx & (1u << 16)) support |= TERN_SIMD_AVX512;
     * #elif defined(__aarch64__) || defined(_M_ARM64)
     *     support |= TERN_SIMD_NEON;   // Always available on AArch64
     * #endif
     */

    return support;
}

/* ══════════════════════════════════════════════════════════════════════
 * terncore_version — Library version string
 *
 * Returns a pointer to a static string.  Matches the version in
 * pyproject.toml.
 * ═════════════════════════════════════════════════════════════════════*/
const char *terncore_version(void)
{
    return "0.1.0";
}
