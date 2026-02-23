/*
 * test_packed.c — Unit tests for ternary_packed 2-bit kernels
 *
 * Validates:
 *   1. Packed kernel matches hand-computed reference values
 *   2. Packed kernel produces identical output to unpacked kernel
 *   3. Pack/unpack roundtrip encoding correctness
 *   4. Bitmap sparse variant matches non-bitmap variant
 *   5. Error handling, edge cases, determinism
 */

#include "ternary_matmul.h"
#include "ternary_packed.h"

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

/* ── Helpers ─────────────────────────────────────────────────────── */

/*
 * Pack a single trit value (+1, -1, 0) into 2-bit encoding.
 *   +1 → 0b01, -1 → 0b10, 0 → 0b00
 */
static uint8_t encode_trit(int8_t w)
{
    if (w == 1)  return TRIT_POS;
    if (w == -1) return TRIT_NEG;
    return TRIT_ZERO;
}

/*
 * Pack int8 weight array into 2-bit format (4 trits per byte).
 * count must be a multiple of 4.
 * Output: packed array of count/4 bytes.
 */
static void pack_weights(const int8_t *weights, uint8_t *packed, int count)
{
    for (int i = 0; i < count; i += 4) {
        packed[i >> 2] = (uint8_t)(
              encode_trit(weights[i])
            | (encode_trit(weights[i + 1]) << 2)
            | (encode_trit(weights[i + 2]) << 4)
            | (encode_trit(weights[i + 3]) << 6)
        );
    }
}

/*
 * Build a packed sparsity bitmap from int8 weights.
 * bit set (1) = non-zero weight.
 */
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

/* ── Test 1: encoding roundtrip ──────────────────────────────────── */
static void test_encoding_roundtrip(void)
{
    /*
     * Verify that pack → decode produces the original ternary values.
     * Weights: [+1, -1, 0, +1, -1, 0, 0, +1]
     *
     * Byte 0: trit0=+1(01), trit1=-1(10), trit2=0(00), trit3=+1(01)
     *       = 0b01_00_10_01 = 0x49
     * Byte 1: trit0=-1(10), trit1=0(00), trit2=0(00), trit3=+1(01)
     *       = 0b01_00_00_10 = 0x42
     */
    int8_t W[] = {1, -1, 0, 1,  -1, 0, 0, 1};
    uint8_t packed[2];
    pack_weights(W, packed, 8);

    ASSERT(packed[0] == 0x49, "byte 0 packing");
    ASSERT(packed[1] == 0x42, "byte 1 packing");

    /* Verify decode of byte 0 */
    ASSERT((packed[0] & TRIT_MASK) == TRIT_POS,                "byte0 trit0 = +1");
    ASSERT(((packed[0] >> 2) & TRIT_MASK) == TRIT_NEG,         "byte0 trit1 = -1");
    ASSERT(((packed[0] >> 4) & TRIT_MASK) == TRIT_ZERO,        "byte0 trit2 =  0");
    ASSERT((packed[0] >> 6) == TRIT_POS,                       "byte0 trit3 = +1");
}

/* ── Test 2: packed matvec — known values ────────────────────────── */
static void test_packed_matvec_known(void)
{
    /*
     * W = [+1, -1,  0, +1]    input = [1, 2, 3, 4]    alpha = 0.5
     *     [ 0, +1, -1,  0]    bias  = [0.1, 0.2]
     *
     * row 0: 1 - 2 + 0 + 4 = 3      → 3 * 0.5 + 0.1 = 1.6
     * row 1: 0 + 2 - 3 + 0 = -1     → -1 * 0.5 + 0.2 = -0.3
     */
    int8_t  W[] = {1, -1, 0, 1,  0, 1, -1, 0};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   bias[] = {0.1f, 0.2f};
    uint8_t packed[2];
    float   output[2];

    pack_weights(W, packed, 8);
    int rc = tern_packed_matvec_f32(packed, input, output, 2, 4, 0.5f, bias);

    ASSERT(rc == TERN_OK, "packed matvec return code");
    ASSERT_NEAR(output[0],  1.6f, 1e-6f, "packed matvec row 0");
    ASSERT_NEAR(output[1], -0.3f, 1e-6f, "packed matvec row 1");
}

/* ── Test 3: packed matvec — no bias ─────────────────────────────── */
static void test_packed_matvec_no_bias(void)
{
    /* W = [+1, +1, +1, +1], input = [1, 2, 3, 4], alpha = 2.0
     * acc = 10, output = 20.0
     */
    int8_t  W[] = {1, 1, 1, 1};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t packed[1];
    float   output[1];

    pack_weights(W, packed, 4);
    int rc = tern_packed_matvec_f32(packed, input, output, 1, 4, 2.0f, NULL);

    ASSERT(rc == TERN_OK, "no-bias return code");
    ASSERT_NEAR(output[0], 20.0f, 1e-6f, "no-bias output");
}

/* ── Test 4: packed matches unpacked ─────────────────────────────── */
static void test_packed_matches_unpacked(void)
{
    /* Use a larger matrix to exercise more code paths */
    const int M = 4, N = 16;
    int8_t  W[4 * 16];
    uint8_t packed[4 * 4];  /* M * N/4 */
    float   input[16], bias[4];
    float   unpacked_out[4], packed_out[4];

    /* Fill with a repeating pattern */
    for (int k = 0; k < M * N; k++) {
        int mod = k % 5;
        W[k] = (mod == 0 || mod == 1) ? 1 : (mod == 2) ? -1 : 0;
    }
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1) * 0.1f;
    for (int i = 0; i < M; i++) bias[i] = (float)i * 0.05f;

    pack_weights(W, packed, M * N);

    tern_matvec_f32(W, input, unpacked_out, M, N, 1.5f, bias);
    int rc = tern_packed_matvec_f32(packed, input, packed_out, M, N, 1.5f, bias);

    ASSERT(rc == TERN_OK, "packed matches unpacked rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "packed==unpacked row %d", i);
        ASSERT_NEAR(packed_out[i], unpacked_out[i], 1e-5f, msg);
    }
}

/* ── Test 5: all-zero weights ────────────────────────────────────── */
static void test_packed_all_zeros(void)
{
    int8_t  W[] = {0, 0, 0, 0,  0, 0, 0, 0};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   bias[] = {0.5f, -0.5f};
    uint8_t packed[2];
    float   output[2];

    pack_weights(W, packed, 8);

    /* packed bytes should be 0x00 — exercises the zero-skip path */
    ASSERT(packed[0] == 0x00, "all-zero byte 0");
    ASSERT(packed[1] == 0x00, "all-zero byte 1");

    int rc = tern_packed_matvec_f32(packed, input, output, 2, 4, 1.0f, bias);
    ASSERT(rc == TERN_OK, "all-zero rc");
    ASSERT_NEAR(output[0],  0.5f, 1e-6f, "all-zero row 0 (bias only)");
    ASSERT_NEAR(output[1], -0.5f, 1e-6f, "all-zero row 1 (bias only)");
}

/* ── Test 6: all-negative weights ────────────────────────────────── */
static void test_packed_all_neg(void)
{
    int8_t  W[] = {-1, -1, -1, -1};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t packed[1];
    float   output[1];

    pack_weights(W, packed, 4);

    /* All -1: encoded as 0b10 each = 0b10_10_10_10 = 0xAA */
    ASSERT(packed[0] == 0xAA, "all-neg packing");

    int rc = tern_packed_matvec_f32(packed, input, output, 1, 4, 1.0f, NULL);
    ASSERT(rc == TERN_OK, "all-neg rc");
    ASSERT_NEAR(output[0], -10.0f, 1e-6f, "all-neg output");
}

/* ── Test 7: batched packed matmul ───────────────────────────────── */
static void test_packed_matmul_batched(void)
{
    /*
     * W = [+1, -1, +1, -1]   alpha = 1.0, no bias
     *     [-1, +1, -1, +1]
     *
     * batch 0: [1, 2, 3, 4]
     *   row 0: 1-2+3-4 = -2,  row 1: -1+2-3+4 = 2
     * batch 1: [5, 5, 5, 5]
     *   row 0: 5-5+5-5 = 0,   row 1: -5+5-5+5 = 0
     */
    int8_t  W[] = {1, -1, 1, -1,  -1, 1, -1, 1};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f, 5.0f, 5.0f, 5.0f};
    uint8_t packed[2];
    float   output[4];

    pack_weights(W, packed, 8);
    int rc = tern_packed_matmul_f32(packed, input, output, 2, 4, 2, 1.0f, NULL);

    ASSERT(rc == TERN_OK, "batched rc");
    ASSERT_NEAR(output[0], -2.0f, 1e-6f, "batch0 row0");
    ASSERT_NEAR(output[1],  2.0f, 1e-6f, "batch0 row1");
    ASSERT_NEAR(output[2],  0.0f, 1e-6f, "batch1 row0");
    ASSERT_NEAR(output[3],  0.0f, 1e-6f, "batch1 row1");
}

/* ── Test 8: batched packed matches batched unpacked ─────────────── */
static void test_packed_batched_matches_unpacked(void)
{
    const int M = 3, N = 8, B = 4;
    int8_t  W[3 * 8];
    uint8_t packed[3 * 2];
    float   input[4 * 8], bias[3];
    float   unpacked_out[4 * 3], packed_out[4 * 3];

    for (int k = 0; k < M * N; k++) {
        int mod = k % 3;
        W[k] = (mod == 0) ? 1 : (mod == 1) ? 0 : -1;
    }
    for (int k = 0; k < B * N; k++) input[k] = (float)(k % 7) * 0.3f;
    for (int i = 0; i < M; i++) bias[i] = (float)i * 0.1f;

    pack_weights(W, packed, M * N);

    tern_matmul_f32(W, input, unpacked_out, M, N, B, 0.8f, bias);
    int rc = tern_packed_matmul_f32(packed, input, packed_out, M, N, B, 0.8f, bias);

    ASSERT(rc == TERN_OK, "batched match rc");
    for (int k = 0; k < B * M; k++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "batched packed==unpacked [%d]", k);
        ASSERT_NEAR(packed_out[k], unpacked_out[k], 1e-5f, msg);
    }
}

/* ── Test 9: sparse packed matches non-sparse packed ─────────────── */
static void test_sparse_matches_nonsparse(void)
{
    const int M = 3, N = 8;
    int8_t  W[3 * 8];
    uint8_t packed[3 * 2], bitmap[(3 * 8 + 7) / 8];
    float   input[8], bias[3];
    float   nonsparse_out[3], sparse_out[3];

    for (int k = 0; k < M * N; k++) {
        int mod = k % 4;
        W[k] = (mod == 0) ? 1 : (mod == 3) ? -1 : 0;
    }
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1);
    for (int i = 0; i < M; i++) bias[i] = (float)i * 0.5f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bitmap, M * N);

    tern_packed_matvec_f32(packed, input, nonsparse_out, M, N, 1.0f, bias);
    int rc = tern_packed_matvec_f32_sparse(
        packed, input, sparse_out, bitmap, M, N, 1.0f, bias);

    ASSERT(rc == TERN_OK, "sparse vs nonsparse rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse==nonsparse row %d", i);
        ASSERT_NEAR(sparse_out[i], nonsparse_out[i], 1e-6f, msg);
    }
}

/* ── Test 10: sparse packed, N not multiple of 8 ─────────────────── */
static void test_sparse_n_not_mult_8(void)
{
    /* N=12: multiple of 4 but not 8 — exercises head/tail phases */
    const int M = 2, N = 12;
    int8_t  W[2 * 12];
    uint8_t packed[2 * 3], bitmap[(2 * 12 + 7) / 8];
    float   input[12];
    float   nonsparse_out[2], sparse_out[2];

    for (int k = 0; k < M * N; k++) {
        W[k] = (k % 3 == 0) ? 1 : (k % 3 == 1) ? -1 : 0;
    }
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1) * 0.5f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bitmap, M * N);

    tern_packed_matvec_f32(packed, input, nonsparse_out, M, N, 2.0f, NULL);
    int rc = tern_packed_matvec_f32_sparse(
        packed, input, sparse_out, bitmap, M, N, 2.0f, NULL);

    ASSERT(rc == TERN_OK, "N=12 sparse rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "N=12 sparse row %d", i);
        ASSERT_NEAR(sparse_out[i], nonsparse_out[i], 1e-5f, msg);
    }
}

/* ── Test 11: sparse batched matches non-sparse batched ──────────── */
static void test_sparse_batched(void)
{
    const int M = 2, N = 8, B = 3;
    int8_t  W[2 * 8];
    uint8_t packed[2 * 2], bitmap[(2 * 8 + 7) / 8];
    float   input[3 * 8], bias[2];
    float   nonsparse_out[3 * 2], sparse_out[3 * 2];

    for (int k = 0; k < M * N; k++) W[k] = (k & 1) ? 1 : 0;
    for (int k = 0; k < B * N; k++) input[k] = (float)(k + 1) * 0.1f;
    bias[0] = 0.01f; bias[1] = 0.02f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bitmap, M * N);

    tern_packed_matmul_f32(packed, input, nonsparse_out, M, N, B, 0.5f, bias);
    int rc = tern_packed_matmul_f32_sparse(
        packed, input, sparse_out, bitmap, M, N, B, 0.5f, bias);

    ASSERT(rc == TERN_OK, "sparse batched rc");
    for (int k = 0; k < B * M; k++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse batched [%d]", k);
        ASSERT_NEAR(sparse_out[k], nonsparse_out[k], 1e-6f, msg);
    }
}

/* ── Test 12: error handling ─────────────────────────────────────── */
static void test_error_handling(void)
{
    uint8_t packed[] = {0x01};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   output[1];

    /* NULL pointer checks */
    ASSERT(tern_packed_matvec_f32(NULL, input, output, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL packed");
    ASSERT(tern_packed_matvec_f32(packed, NULL, output, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL input");
    ASSERT(tern_packed_matvec_f32(packed, input, NULL, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL output");

    /* Dimension checks */
    ASSERT(tern_packed_matvec_f32(packed, input, output, 0, 4, 1.0f, NULL)
           == TERN_ERR_DIM, "M=0");
    ASSERT(tern_packed_matvec_f32(packed, input, output, 1, 0, 1.0f, NULL)
           == TERN_ERR_DIM, "N=0");
    ASSERT(tern_packed_matmul_f32(packed, input, output, 1, 4, 0, 1.0f, NULL)
           == TERN_ERR_DIM, "B=0");

    /* Alignment check: N must be multiple of 4 */
    ASSERT(tern_packed_matvec_f32(packed, input, output, 1, 3, 1.0f, NULL)
           == TERN_ERR_ALIGN, "N=3 not aligned");
    ASSERT(tern_packed_matvec_f32(packed, input, output, 1, 5, 1.0f, NULL)
           == TERN_ERR_ALIGN, "N=5 not aligned");

    /* Sparse variant: NULL bitmap */
    ASSERT(tern_packed_matvec_f32_sparse(packed, input, output, NULL, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL bitmap");
}

/* ── Test 13: determinism — bit-identical across 100 runs ────────── */
static void test_determinism(void)
{
    int8_t  W[] = {1, -1, 0, 1, -1, 0, 1, 0, -1, 0, 1, -1};
    float   input[] = {0.123f, 0.456f, 0.789f, 0.321f};
    uint8_t packed[3];
    float   first[3], current[3];

    pack_weights(W, packed, 12);
    tern_packed_matvec_f32(packed, input, first, 3, 4, 0.42f, NULL);

    for (int run = 0; run < 100; run++) {
        tern_packed_matvec_f32(packed, input, current, 3, 4, 0.42f, NULL);
        for (int i = 0; i < 3; i++) {
            ASSERT(current[i] == first[i], "determinism: bit-identical");
        }
    }
}

/* ── Test 14: large matrix — stress test packed vs unpacked ──────── */
static void test_large_matrix(void)
{
    const int M = 64, N = 256;
    int8_t  *W      = malloc((size_t)M * (size_t)N * sizeof(int8_t));
    uint8_t *packed  = malloc((size_t)M * (size_t)(N / 4) * sizeof(uint8_t));
    float   *input   = malloc((size_t)N * sizeof(float));
    float   *bias    = malloc((size_t)M * sizeof(float));
    float   *out_unp = malloc((size_t)M * sizeof(float));
    float   *out_pak = malloc((size_t)M * sizeof(float));

    if (!W || !packed || !input || !bias || !out_unp || !out_pak) {
        fprintf(stderr, "FAIL: allocation failed for large matrix test\n");
        failures++;
        goto cleanup;
    }

    /* Fill with ~65% sparsity pattern */
    for (int k = 0; k < M * N; k++) {
        int mod = k % 10;
        if (mod < 6)      W[k] = 0;    /* 60% zeros */
        else if (mod < 8) W[k] = 1;    /* 20% +1 */
        else              W[k] = -1;   /* 20% -1 */
    }
    for (int j = 0; j < N; j++) input[j] = (float)(j % 17) * 0.1f;
    for (int i = 0; i < M; i++) bias[i] = (float)(i % 5) * 0.01f;

    pack_weights(W, packed, M * N);

    tern_matvec_f32(W, input, out_unp, M, N, 0.73f, bias);
    int rc = tern_packed_matvec_f32(packed, input, out_pak, M, N, 0.73f, bias);

    ASSERT(rc == TERN_OK, "large matrix rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "large matrix row %d", i);
        ASSERT_NEAR(out_pak[i], out_unp[i], 1e-4f, msg);
    }

cleanup:
    free(W); free(packed); free(input); free(bias);
    free(out_unp); free(out_pak);
}

/* ── Test 15: single-element (1x4 minimum) ───────────────────────── */
static void test_single_row(void)
{
    /* Minimum: M=1, N=4 */
    int8_t  W[] = {1, 0, -1, 0};
    float   input[] = {10.0f, 20.0f, 30.0f, 40.0f};
    uint8_t packed[1];
    float   output[1];

    pack_weights(W, packed, 4);

    /* acc = 10 + 0 - 30 + 0 = -20, output = -20 * 0.5 = -10.0 */
    tern_packed_matvec_f32(packed, input, output, 1, 4, 0.5f, NULL);
    ASSERT_NEAR(output[0], -10.0f, 1e-6f, "single row output");
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void)
{
    printf("Running ternary_packed 2-bit kernel tests...\n\n");

    test_encoding_roundtrip();
    test_packed_matvec_known();
    test_packed_matvec_no_bias();
    test_packed_matches_unpacked();
    test_packed_all_zeros();
    test_packed_all_neg();
    test_packed_matmul_batched();
    test_packed_batched_matches_unpacked();
    test_sparse_matches_nonsparse();
    test_sparse_n_not_mult_8();
    test_sparse_batched();
    test_error_handling();
    test_determinism();
    test_large_matrix();
    test_single_row();

    printf("\n%s (%d failure%s)\n",
           failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
           failures, failures == 1 ? "" : "s");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
