/*
 * ternary_matmul.h — Scalar ternary matrix multiplication interface
 *
 * Part of tern-core Stage 1B: C extension layer for accelerated
 * ternary neural network inference.
 *
 * Patent 36: Ternary weight encoding, deterministic execution
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#ifndef TERNARY_MATMUL_H
#define TERNARY_MATMUL_H

#include <stdint.h>
#include <stddef.h>

/* ── Error codes ──────────────────────────────────────────────────── */

#define TERN_OK         0   /* Success */
#define TERN_ERR_NULL  -1   /* NULL pointer argument */
#define TERN_ERR_DIM   -2   /* Invalid dimension (M, N, or B <= 0) */

/* ── Ternary weight values ────────────────────────────────────────── */

#define TERN_POS  ((int8_t) 1)   /* +1: add input (pass-through) */
#define TERN_NEG  ((int8_t)-1)   /* -1: subtract input (negate)  */
#define TERN_ZERO ((int8_t) 0)   /*  0: skip (no operation)      */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Dense operations ─────────────────────────────────────────────── */

/*
 * tern_matvec_f32 — Dense ternary matrix-vector multiply
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Multiplication is eliminated: W[i,j] in {-1, 0, +1} maps to
 * subtract, skip, or add.  (Patent 36, Claim 1)
 *
 * Parameters:
 *   weights  [M x N] int8_t, row-major, values in {-1, 0, +1}
 *   input    [N]     float32 input activations
 *   output   [M]     float32 output (caller-allocated)
 *   M        number of output units (rows of weight matrix)
 *   N        number of input units (columns of weight matrix)
 *   alpha    per-layer scaling factor (float32, mean |W| of non-zeros)
 *   bias     [M] float32 bias vector, or NULL if no bias
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 */
int tern_matvec_f32(
    const int8_t *weights,
    const float  *input,
    float        *output,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_matmul_f32 — Dense ternary matrix multiply (batched)
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Equivalent to running tern_matvec_f32 for each of B input vectors.
 *
 * Parameters:
 *   weights  [M x N] int8_t row-major, values in {-1, 0, +1}
 *   input    [B x N] float32 row-major
 *   output   [B x M] float32 row-major (caller-allocated)
 *   M, N     weight matrix dimensions
 *   B        batch size (number of input vectors)
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 */
int tern_matmul_f32(
    const int8_t *weights,
    const float  *input,
    float        *output,
    int M, int N, int B,
    float alpha,
    const float *bias);

/* ── Sparse operations (bitmap zero-skip) ─────────────────────────── */

/*
 * tern_matvec_f32_sparse — Sparse ternary matrix-vector multiply
 *
 * Same computation as tern_matvec_f32, but uses a packed sparsity
 * bitmap to skip zero-weight regions.  With 60-70% sparsity, this
 * skips the majority of operations.  (Patent 37, Claim 1)
 *
 * Bitmap format (flat, LSB-first):
 *   Weight at flat index k = i*N + j is stored as:
 *     byte index  = k / 8
 *     bit position = k % 8   (bit 0 is the LSB)
 *     bit value 1 = non-zero weight, 0 = zero weight
 *
 *   Total bitmap size: ceil(M * N / 8) bytes.
 *
 * Parameters:
 *   weights  [M x N] int8_t row-major (only non-zero entries read)
 *   input    [N]     float32
 *   output   [M]     float32 (caller-allocated)
 *   bitmap   packed sparsity bitmap, ceil(M*N / 8) bytes
 *   M, N     weight matrix dimensions
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 */
int tern_matvec_f32_sparse(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias);

/*
 * tern_matmul_f32_sparse — Sparse ternary matrix multiply (batched)
 *
 * Batched version of tern_matvec_f32_sparse.
 *
 * Parameters:
 *   weights  [M x N] int8_t row-major
 *   input    [B x N] float32 row-major
 *   output   [B x M] float32 row-major (caller-allocated)
 *   bitmap   packed sparsity bitmap, ceil(M*N / 8) bytes
 *   M, N, B  dimensions
 *   alpha    per-layer scaling factor
 *   bias     [M] float32, or NULL
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 */
int tern_matmul_f32_sparse(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_MATMUL_H */
