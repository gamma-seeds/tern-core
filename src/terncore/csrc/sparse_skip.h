/*
 * sparse_skip.h — 64-bit word bitmap-driven zero-skip engine
 *
 * Processes 64 weights per bitmap word using bit-scan iteration.
 * When a uint64 bitmap word is zero, the entire 64-weight block is
 * skipped.  For non-zero words, only the set bits are visited via
 * count-trailing-zeros (CTZ), so compute is proportional to the
 * number of non-zero weights, not total weights.
 *
 * With 60-70% sparsity, only 30-40% of weights are visited.
 *
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic
 * Patent 39: Ternary-native memory → packed trit storage format
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#ifndef SPARSE_SKIP_H
#define SPARSE_SKIP_H

#include "ternary_packed.h"   /* TERN_OK, TERN_ERR_*, TRIT_* */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Unpacked (int8) + 64-bit bitmap ─────────────────────────────── */

/*
 * tern_sparse64_matvec_f32 — Sparse matvec with 64-bit word bitmap
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Uses CTZ bit-scan on uint64 bitmap words to iterate only over
 * non-zero weight positions.  Entire 64-weight blocks are skipped
 * when their bitmap word is zero.  (Patent 37)
 *
 * Bitmap format: same flat LSB-first layout as ternary_matmul.h,
 * read as uint64 little-endian words (8 bitmap bytes per word).
 *
 * Parameters:
 *   weights  [M x N] int8_t row-major, values in {-1, 0, +1}
 *   input    [N]     float32
 *   output   [M]     float32 (caller-allocated)
 *   bitmap   packed sparsity bitmap, ceil(M*N / 8) bytes
 *   M, N     matrix dimensions
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 */
int tern_sparse64_matvec_f32(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_sparse64_matmul_f32 — Batched sparse matmul with 64-bit bitmap
 */
int tern_sparse64_matmul_f32(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias);

/* ── Packed (2-bit) + 64-bit bitmap ──────────────────────────────── */

/*
 * tern_sparse64_packed_matvec_f32 — Packed sparse matvec, 64-bit bitmap
 *
 * Combines packed 2-bit weights with 64-bit word bitmap bit-scan.
 * Individual trits are extracted on-the-fly from the packed array
 * at the positions indicated by the bitmap.
 *
 * N must be a multiple of 4 (TERN_ERR_ALIGN otherwise).
 *
 * Parameters:
 *   packed   [M * N/4] uint8_t packed weights
 *   input    [N]       float32
 *   output   [M]       float32 (caller-allocated)
 *   bitmap   packed sparsity bitmap, ceil(M*N / 8) bytes
 *   M, N     matrix dimensions (N must be multiple of 4)
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 */
int tern_sparse64_packed_matvec_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_sparse64_packed_matmul_f32 — Batched packed sparse matmul
 */
int tern_sparse64_packed_matmul_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias);

#ifdef __cplusplus
}
#endif

#endif /* SPARSE_SKIP_H */
