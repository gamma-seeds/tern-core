/*
 * ternary_packed.h — Packed 2-bit ternary matrix multiplication interface
 *
 * Operates directly on packed weights (4 trits per byte) without
 * unpacking to int8.  Reduces memory bandwidth by 4x compared
 * to the unpacked int8 format in ternary_matmul.h.
 *
 * Patent 36: Ternary weight encoding, deterministic execution
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic
 * Patent 39: Ternary-native memory → packed trit storage format
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#ifndef TERNARY_PACKED_H
#define TERNARY_PACKED_H

#include "ternary_matmul.h"   /* TERN_OK, TERN_ERR_NULL, TERN_ERR_DIM */

/* ── Additional error code for packed operations ──────────────────── */

#define TERN_ERR_ALIGN  -3   /* N not a multiple of 4 */

/* ── 2-bit trit encoding (matches Python sparse/__init__.py) ──────
 *
 * Byte layout (4 trits per uint8, LSB-first):
 *
 *   bit position   7  6 │ 5  4 │ 3  2 │ 1  0
 *                 ──────┼──────┼──────┼──────
 *   trit index      3   │  2   │  1   │  0
 *
 * Extraction:
 *   trit 0 = byte        & 0x03
 *   trit 1 = (byte >> 2) & 0x03
 *   trit 2 = (byte >> 4) & 0x03
 *   trit 3 = byte >> 6
 * ─────────────────────────────────────────────────────────────────── */

#define TRIT_ZERO     0x00   /*  0: skip (no operation) */
#define TRIT_POS      0x01   /* +1: add input           */
#define TRIT_NEG      0x02   /* -1: subtract input      */
#define TRIT_RESERVED 0x03   /* reserved — must not appear */
#define TRIT_MASK     0x03   /* 2-bit mask for one trit  */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Packed operations (no bitmap) ────────────────────────────────── */

/*
 * tern_packed_matvec_f32 — Packed ternary matrix-vector multiply
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Reads weights directly from the 2-bit packed format (4 trits per
 * byte).  Each packed byte is decoded inline: for each 2-bit trit,
 * 0b01 → add, 0b10 → subtract, 0b00 → skip.  No unpacking step.
 *
 * Zero-skip: when a packed byte is 0x00, all 4 trits are zero and
 * the entire 4-weight group is skipped.  (Patent 37)
 *
 * IMPORTANT: N must be a multiple of 4 so that each row occupies
 * exactly N/4 packed bytes with no byte-straddling across rows.
 *
 * Parameters:
 *   packed   [M * N/4] uint8_t, flat row-major packed weights
 *   input    [N]       float32 input activations
 *   output   [M]       float32 output (caller-allocated)
 *   M        number of output units (rows)
 *   N        number of input units (columns, must be multiple of 4)
 *   alpha    per-layer scaling factor (float32)
 *   bias     [M] float32, or NULL
 *
 * Returns: TERN_OK, TERN_ERR_NULL, TERN_ERR_DIM, or TERN_ERR_ALIGN.
 */
int tern_packed_matvec_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_packed_matmul_f32 — Packed ternary matrix multiply (batched)
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Parameters:
 *   packed   [M * N/4] uint8_t packed weights
 *   input    [B x N]   float32 row-major
 *   output   [B x M]   float32 row-major (caller-allocated)
 *   M, N     matrix dimensions (N must be multiple of 4)
 *   B        batch size
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 */
int tern_packed_matmul_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N, int B,
    float alpha,
    const float *bias);

/* ── Packed operations with bitmap zero-skip ──────────────────────── */

/*
 * tern_packed_matvec_f32_sparse — Packed ternary matvec with bitmap
 *
 * Same computation as tern_packed_matvec_f32, with an additional
 * packed sparsity bitmap for 8-weight block skipping.  When a
 * bitmap byte is 0x00, all 8 weights (2 packed bytes) are skipped
 * without reading the packed weight data.  (Patent 37)
 *
 * For best performance, N should be a multiple of 8 so that bitmap
 * bytes align with row boundaries.  N must be a multiple of 4.
 *
 * Bitmap format: same as ternary_matmul.h (flat, LSB-first, 1 bit
 * per weight, ceil(M*N / 8) bytes total).
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
int tern_packed_matvec_f32_sparse(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_packed_matmul_f32_sparse — Packed ternary matmul with bitmap
 *
 * Parameters:
 *   packed   [M * N/4] uint8_t packed weights
 *   input    [B x N]   float32 row-major
 *   output   [B x M]   float32 row-major (caller-allocated)
 *   bitmap   packed sparsity bitmap, ceil(M*N / 8) bytes
 *   M, N, B  dimensions (N must be multiple of 4)
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 */
int tern_packed_matmul_f32_sparse(
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

#endif /* TERNARY_PACKED_H */
