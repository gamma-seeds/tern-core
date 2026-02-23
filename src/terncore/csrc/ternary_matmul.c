/*
 * ternary_matmul.c — Scalar ternary matrix multiplication kernels
 *
 * Implements the core ternary compute primitive: multiplication is
 * replaced with compare-and-add.
 *
 *   weight == +1  →  accumulator += input   (pass-through)
 *   weight == -1  →  accumulator -= input   (negate)
 *   weight ==  0  →  skip                   (no operation)
 *
 * This eliminates all multiply instructions from the inner loop.
 * Combined with 60-70% zero-weight sparsity, the majority of
 * floating-point operations are skipped entirely.
 *
 * Patent 36: Biological neural mapping → ternary weight encoding.
 *            Deterministic execution for governance and audit.
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic.
 *
 * All functions are deterministic: identical inputs always produce
 * bit-identical outputs (IEEE 754 float32 addition semantics,
 * evaluated left-to-right in column order).
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "ternary_matmul.h"

/* ─────────────────────────────────────────────────────────────────────
 * Bitmap helper: test whether weight at flat index k is non-zero.
 *
 * Bitmap layout — flat, LSB-first packed bits:
 *   byte index   = k >> 3
 *   bit position = k & 7   (bit 0 is the LSB of each byte)
 *   bit value 1  = non-zero weight
 * ─────────────────────────────────────────────────────────────────── */
static inline int bitmap_test(const uint8_t *bitmap, size_t k)
{
    return (bitmap[k >> 3] >> (k & 7)) & 1;
}

/* ══════════════════════════════════════════════════════════════════════
 * Dense ternary matrix-vector multiply
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * The inner loop performs only add and subtract — no multiply.
 * Zero weights are skipped via the branch (implicit zero-skip).
 *
 * Patent 36, Claim 1: ternary weight encoding, add/subtract/skip.
 * Patent 37:          zero weights skip via branch (implicit gating).
 * ═════════════════════════════════════════════════════════════════════*/
int tern_matvec_f32(
    const int8_t *weights,
    const float  *input,
    float        *output,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0) return TERN_ERR_DIM;

    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const int8_t *row = weights + (size_t)i * (size_t)N;

        for (int j = 0; j < N; j++) {
            /*
             * Patent 36, Claim 1 — ternary compute, no multiply:
             *   +1 → add (pass-through)
             *   -1 → subtract (negate)
             *    0 → skip (zero-weight clock-gating, Patent 37)
             */
            int8_t w = row[j];
            if (w == TERN_POS) {
                acc += input[j];
            } else if (w == TERN_NEG) {
                acc -= input[j];
            }
            /* w == 0: no operation — zero-weight skip (Patent 37) */
        }

        /* Per-layer scaling factor (alpha = mean |W| of non-zeros) */
        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Dense ternary matrix multiply (batched)
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Each batch element is processed independently via tern_matvec_f32.
 * Weight matrix is shared across the batch (standard linear layer).
 * ═════════════════════════════════════════════════════════════════════*/
int tern_matmul_f32(
    const int8_t *weights,
    const float  *input,
    float        *output,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0) return TERN_ERR_DIM;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_matvec_f32(weights, in_b, out_b, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Sparse ternary matrix-vector multiply (bitmap zero-skip)
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *               (only where bitmap bit is set)
 *
 * Uses a packed sparsity bitmap (1 bit per weight) to skip zero
 * regions.  The bitmap is processed in aligned 8-weight blocks:
 * when a bitmap byte is 0x00, all 8 weights are zero and the
 * entire block is skipped without touching the weight array.
 *
 * With 60-70% zero weights, many bitmap bytes will be 0x00,
 * giving substantial speedup over per-element branching.
 *
 * The function handles arbitrary N (not required to be a multiple
 * of 8) by processing in three phases:
 *   Phase 1: unaligned head (0-7 weights to reach byte boundary)
 *   Phase 2: aligned body  (8 weights per bitmap byte, block-skip)
 *   Phase 3: remaining tail (0-7 weights)
 *
 * Patent 37, Claim 1: sparsity bitmap for zero-weight clock-gating.
 * Patent 36:          deterministic execution (left-to-right order).
 * ═════════════════════════════════════════════════════════════════════*/
int tern_matvec_f32_sparse(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0) return TERN_ERR_DIM;

    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const size_t row_start = (size_t)i * (size_t)N;
        const int8_t *row = weights + row_start;

        int j = 0;

        /* ── Phase 1: unaligned head ──────────────────────────────
         * If row_start is not byte-aligned in the bitmap, process
         * individual weights until we reach a byte boundary.
         */
        int head_misalign = (int)(row_start & 7);
        if (head_misalign != 0) {
            int head_count = 8 - head_misalign;
            if (head_count > N) head_count = N;

            for (; j < head_count; j++) {
                if (bitmap_test(bitmap, row_start + (size_t)j)) {
                    /* Bitmap says non-zero; weight is +1 or -1. */
                    int8_t w = row[j];
                    if (w == TERN_POS) acc += input[j];
                    else               acc -= input[j];
                }
            }
        }

        /* ── Phase 2: aligned body (8 weights per bitmap byte) ────
         * Each bitmap byte covers exactly 8 consecutive weights.
         * When the byte is 0x00, all 8 weights are zero — skip
         * the entire block without reading the weight array.
         * (Patent 37: zero-weight clock-gating at block level)
         */
        while (j + 8 <= N) {
            size_t byte_idx = (row_start + (size_t)j) >> 3;
            uint8_t bm = bitmap[byte_idx];

            if (bm == 0) {
                /* All 8 weights zero — skip entire block (Patent 37) */
                j += 8;
                continue;
            }

            /* At least one non-zero weight in this block. */
            for (int bit = 0; bit < 8; bit++, j++) {
                if (bm & (1u << bit)) {
                    int8_t w = row[j];
                    if (w == TERN_POS) acc += input[j];
                    else               acc -= input[j];
                }
            }
        }

        /* ── Phase 3: remaining tail ──────────────────────────────
         * Process any leftover weights (< 8) that don't fill a
         * complete bitmap byte.
         */
        for (; j < N; j++) {
            if (bitmap_test(bitmap, row_start + (size_t)j)) {
                int8_t w = row[j];
                if (w == TERN_POS) acc += input[j];
                else               acc -= input[j];
            }
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Sparse ternary matrix multiply (batched, bitmap zero-skip)
 *
 * Each batch element is processed independently via
 * tern_matvec_f32_sparse.  The weight matrix and bitmap are shared
 * across the batch.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_matmul_f32_sparse(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0) return TERN_ERR_DIM;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_matvec_f32_sparse(
            weights, in_b, out_b, bitmap, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}
