/*
 * test_matmul.c — Unit tests for ternary_matmul scalar kernels
 *
 * Validates correctness of dense and sparse ternary matmul against
 * hand-computed reference values.
 */

#include "ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond, msg)                                         \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "FAIL [%s:%d] %s\n",                  \
                    __FILE__, __LINE__, (msg));                    \
            failures++;                                           \
        }                                                         \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                               \
    do {                                                          \
        if (fabsf((a) - (b)) > (tol)) {                           \
            fprintf(stderr, "FAIL [%s:%d] %s: got %f, want %f\n", \
                    __FILE__, __LINE__, (msg), (a), (b));          \
            failures++;                                           \
        }                                                         \
    } while (0)

static int failures = 0;

/* ── Helper: build packed bitmap from int8 weights ───────────────── */
static void build_bitmap(const int8_t *weights, uint8_t *bitmap, int count)
{
    int nbytes = (count + 7) / 8;
    memset(bitmap, 0, (size_t)nbytes);
    for (int k = 0; k < count; k++) {
        if (weights[k] != 0) {
            bitmap[k >> 3] |= (uint8_t)(1u << (k & 7));
        }
    }
}

/* ── Test 1: dense matvec — known values ─────────────────────────── */
static void test_dense_matvec_known(void)
{
    /*
     * W = [+1, -1,  0]    input = [1.0, 2.0, 3.0]    alpha = 0.5
     *     [ 0, +1, +1]    bias  = [0.1, 0.2]
     *
     * row 0: (+1)*1 + (-1)*2 + 0*3 = 1 - 2 = -1    → -1 * 0.5 + 0.1 = -0.4
     * row 1:  0*1   + (+1)*2 + (+1)*3 = 2 + 3 = 5   →  5 * 0.5 + 0.2 =  2.7
     */
    int8_t W[]     = {1, -1, 0,  0, 1, 1};
    float  input[] = {1.0f, 2.0f, 3.0f};
    float  bias[]  = {0.1f, 0.2f};
    float  output[2];

    int rc = tern_matvec_f32(W, input, output, 2, 3, 0.5f, bias);
    ASSERT(rc == TERN_OK, "matvec return code");
    ASSERT_NEAR(output[0], -0.4f, 1e-6f, "matvec row 0");
    ASSERT_NEAR(output[1],  2.7f, 1e-6f, "matvec row 1");
}

/* ── Test 2: dense matvec — no bias ──────────────────────────────── */
static void test_dense_matvec_no_bias(void)
{
    /* W = [+1, +1, +1, +1], input = [1, 2, 3, 4], alpha = 2.0
     * acc = 1+2+3+4 = 10, output = 10 * 2.0 = 20.0
     */
    int8_t W[]     = {1, 1, 1, 1};
    float  input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float  output[1];

    int rc = tern_matvec_f32(W, input, output, 1, 4, 2.0f, NULL);
    ASSERT(rc == TERN_OK, "no-bias return code");
    ASSERT_NEAR(output[0], 20.0f, 1e-6f, "no-bias output");
}

/* ── Test 3: dense matvec — all zeros ────────────────────────────── */
static void test_dense_matvec_all_zeros(void)
{
    int8_t W[]     = {0, 0, 0, 0, 0, 0};
    float  input[] = {1.0f, 2.0f, 3.0f};
    float  bias[]  = {0.5f, -0.5f};
    float  output[2];

    int rc = tern_matvec_f32(W, input, output, 2, 3, 1.0f, bias);
    ASSERT(rc == TERN_OK, "all-zeros return code");
    ASSERT_NEAR(output[0],  0.5f, 1e-6f, "all-zeros row 0 (bias only)");
    ASSERT_NEAR(output[1], -0.5f, 1e-6f, "all-zeros row 1 (bias only)");
}

/* ── Test 4: dense matvec — all negatives ────────────────────────── */
static void test_dense_matvec_all_neg(void)
{
    /* W = [-1, -1, -1], input = [1, 2, 3], alpha = 1.0
     * acc = -1 - 2 - 3 = -6, output = -6 * 1.0 = -6.0
     */
    int8_t W[]     = {-1, -1, -1};
    float  input[] = {1.0f, 2.0f, 3.0f};
    float  output[1];

    int rc = tern_matvec_f32(W, input, output, 1, 3, 1.0f, NULL);
    ASSERT(rc == TERN_OK, "all-neg return code");
    ASSERT_NEAR(output[0], -6.0f, 1e-6f, "all-neg output");
}

/* ── Test 5: batched matmul ──────────────────────────────────────── */
static void test_dense_matmul_batched(void)
{
    /*
     * W = [+1, -1]   alpha = 1.0   bias = NULL
     *     [-1, +1]
     *
     * input batch 0: [3, 5]  → row0: 3-5=-2, row1: -3+5=2
     * input batch 1: [1, 1]  → row0: 1-1= 0, row1: -1+1=0
     */
    int8_t W[]     = {1, -1,  -1, 1};
    float  input[] = {3.0f, 5.0f,  1.0f, 1.0f};
    float  output[4];

    int rc = tern_matmul_f32(W, input, output, 2, 2, 2, 1.0f, NULL);
    ASSERT(rc == TERN_OK, "batched return code");
    ASSERT_NEAR(output[0], -2.0f, 1e-6f, "batch0 row0");
    ASSERT_NEAR(output[1],  2.0f, 1e-6f, "batch0 row1");
    ASSERT_NEAR(output[2],  0.0f, 1e-6f, "batch1 row0");
    ASSERT_NEAR(output[3],  0.0f, 1e-6f, "batch1 row1");
}

/* ── Test 6: sparse matvec matches dense ─────────────────────────── */
static void test_sparse_matches_dense(void)
{
    /* Same as test 1, but via the sparse path */
    int8_t W[]     = {1, -1, 0,  0, 1, 1};
    float  input[] = {1.0f, 2.0f, 3.0f};
    float  bias[]  = {0.1f, 0.2f};
    float  dense_out[2], sparse_out[2];
    uint8_t bitmap[1]; /* ceil(6/8) = 1 byte */

    build_bitmap(W, bitmap, 6);

    tern_matvec_f32(W, input, dense_out, 2, 3, 0.5f, bias);
    int rc = tern_matvec_f32_sparse(W, input, sparse_out, bitmap, 2, 3, 0.5f, bias);

    ASSERT(rc == TERN_OK, "sparse return code");
    ASSERT_NEAR(sparse_out[0], dense_out[0], 1e-6f, "sparse==dense row 0");
    ASSERT_NEAR(sparse_out[1], dense_out[1], 1e-6f, "sparse==dense row 1");
}

/* ── Test 7: sparse matvec — large matrix, unaligned N ───────────── */
static void test_sparse_unaligned(void)
{
    /* N=11 (not a multiple of 8) to exercise head/tail phases */
    const int M = 3, N = 11;
    int8_t W[3 * 11];
    float input[11], dense_out[3], sparse_out[3];
    uint8_t bitmap[(3 * 11 + 7) / 8]; /* 5 bytes */

    /* Fill with a pattern: alternating +1, 0, -1 */
    for (int k = 0; k < M * N; k++) {
        int mod = k % 3;
        W[k] = (mod == 0) ? 1 : (mod == 1) ? 0 : -1;
    }
    for (int j = 0; j < N; j++) {
        input[j] = (float)(j + 1);
    }

    build_bitmap(W, bitmap, M * N);
    tern_matvec_f32(W, input, dense_out, M, N, 1.5f, NULL);
    int rc = tern_matvec_f32_sparse(W, input, sparse_out, bitmap, M, N, 1.5f, NULL);

    ASSERT(rc == TERN_OK, "unaligned sparse return code");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "unaligned sparse row %d", i);
        ASSERT_NEAR(sparse_out[i], dense_out[i], 1e-5f, msg);
    }
}

/* ── Test 8: sparse batched matches dense batched ────────────────── */
static void test_sparse_batched(void)
{
    int8_t W[]     = {1, -1, 0, 1,  0, 0, 1, -1};
    float  input[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f, 8.0f};
    float  bias[]  = {0.1f, 0.2f};
    float  dense_out[4], sparse_out[4];
    uint8_t bitmap[(2 * 4 + 7) / 8]; /* 1 byte */

    build_bitmap(W, bitmap, 8);
    tern_matmul_f32(W, input, dense_out, 2, 4, 2, 0.3f, bias);
    int rc = tern_matmul_f32_sparse(W, input, sparse_out, bitmap, 2, 4, 2, 0.3f, bias);

    ASSERT(rc == TERN_OK, "sparse batched return code");
    for (int k = 0; k < 4; k++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse batched element %d", k);
        ASSERT_NEAR(sparse_out[k], dense_out[k], 1e-6f, msg);
    }
}

/* ── Test 9: error handling ──────────────────────────────────────── */
static void test_error_handling(void)
{
    int8_t W[]     = {1};
    float  input[] = {1.0f};
    float  output[1];

    ASSERT(tern_matvec_f32(NULL, input, output, 1, 1, 1.0f, NULL) == TERN_ERR_NULL,
           "NULL weights");
    ASSERT(tern_matvec_f32(W, NULL, output, 1, 1, 1.0f, NULL) == TERN_ERR_NULL,
           "NULL input");
    ASSERT(tern_matvec_f32(W, input, NULL, 1, 1, 1.0f, NULL) == TERN_ERR_NULL,
           "NULL output");
    ASSERT(tern_matvec_f32(W, input, output, 0, 1, 1.0f, NULL) == TERN_ERR_DIM,
           "M=0");
    ASSERT(tern_matvec_f32(W, input, output, 1, 0, 1.0f, NULL) == TERN_ERR_DIM,
           "N=0");
    ASSERT(tern_matmul_f32(W, input, output, 1, 1, 0, 1.0f, NULL) == TERN_ERR_DIM,
           "B=0");
}

/* ── Test 10: determinism — same result over 100 runs ────────────── */
static void test_determinism(void)
{
    int8_t W[]     = {1, -1, 0, 1, -1, 0, 1, 0, -1};
    float  input[] = {0.123f, 0.456f, 0.789f};
    float  first[3], current[3];

    tern_matvec_f32(W, input, first, 3, 3, 0.42f, NULL);

    for (int run = 0; run < 100; run++) {
        tern_matvec_f32(W, input, current, 3, 3, 0.42f, NULL);
        for (int i = 0; i < 3; i++) {
            /* Bit-identical, not just "near" */
            ASSERT(current[i] == first[i], "determinism: bit-identical");
        }
    }
}

/* ── Test 11: single element ─────────────────────────────────────── */
static void test_single_element(void)
{
    int8_t W[] = {1};
    float input[] = {7.5f};
    float output[1];

    tern_matvec_f32(W, input, output, 1, 1, 2.0f, NULL);
    ASSERT_NEAR(output[0], 15.0f, 1e-6f, "single +1 element");

    W[0] = -1;
    tern_matvec_f32(W, input, output, 1, 1, 2.0f, NULL);
    ASSERT_NEAR(output[0], -15.0f, 1e-6f, "single -1 element");

    W[0] = 0;
    tern_matvec_f32(W, input, output, 1, 1, 2.0f, NULL);
    ASSERT_NEAR(output[0], 0.0f, 1e-6f, "single 0 element");
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void)
{
    printf("Running ternary_matmul scalar kernel tests...\n\n");

    test_dense_matvec_known();
    test_dense_matvec_no_bias();
    test_dense_matvec_all_zeros();
    test_dense_matvec_all_neg();
    test_dense_matmul_batched();
    test_sparse_matches_dense();
    test_sparse_unaligned();
    test_sparse_batched();
    test_error_handling();
    test_determinism();
    test_single_element();

    printf("\n%s (%d failure%s)\n",
           failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
           failures, failures == 1 ? "" : "s");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
