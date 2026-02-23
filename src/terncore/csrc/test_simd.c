/*
 * test_simd.c — SIMD kernel tests: bit-identical verification
 *
 * Verifies that SIMD kernels (AVX2/NEON) produce output that is
 * bit-identical to the scalar kernels.  Uses exact float equality
 * (==), not tolerance-based comparison.
 *
 * Patent 36: Deterministic execution — SIMD must match scalar exactly.
 * Patent 38: Configurable precision — SIMD/scalar dual-path.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "ternary_simd.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        failures++; \
    } \
} while (0)

/* ── Helpers ─────────────────────────────────────────────────────── */

static uint8_t encode_trit(int w)
{
    if (w == 1)  return TRIT_POS;
    if (w == -1) return TRIT_NEG;
    return TRIT_ZERO;
}

static void pack_weights(const int8_t *weights, uint8_t *packed, int count)
{
    for (int i = 0; i < count; i += 4) {
        packed[i >> 2] = (uint8_t)(
            encode_trit(weights[i])
            | (encode_trit(weights[i + 1]) << 2)
            | (encode_trit(weights[i + 2]) << 4)
            | (encode_trit(weights[i + 3]) << 6));
    }
}

static void fill_sparse_pattern(int8_t *weights, int count)
{
    /* ~65% sparsity pattern */
    for (int i = 0; i < count; i++) {
        if (i % 3 == 0)      weights[i] = 1;
        else if (i % 7 == 0) weights[i] = -1;
        else                  weights[i] = 0;
    }
}

/* Check that we have a SIMD kernel available on this platform */
static int have_simd_kernel(void)
{
    uint32_t caps = get_simd_support();
#if defined(__x86_64__) || defined(_M_X64)
    return (caps & TERN_SIMD_AVX2) != 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return (caps & TERN_SIMD_NEON) != 0;
#else
    (void)caps;
    return 0;
#endif
}

/* Run SIMD matvec on this platform (dispatches to AVX2 or NEON) */
static int run_simd_matvec(
    const uint8_t *packed, const float *input, float *output,
    int M, int N, float alpha, const float *bias)
{
#if defined(__x86_64__) || defined(_M_X64)
    return tern_packed_matvec_f32_avx2(packed, input, output, M, N, alpha, bias);
#elif defined(__aarch64__) || defined(_M_ARM64)
    return tern_packed_matvec_f32_neon(packed, input, output, M, N, alpha, bias);
#else
    (void)packed; (void)input; (void)output;
    (void)M; (void)N; (void)alpha; (void)bias;
    return TERN_ERR_NULL;
#endif
}

/* Run SIMD matmul on this platform */
static int run_simd_matmul(
    const uint8_t *packed, const float *input, float *output,
    int M, int N, int B, float alpha, const float *bias)
{
#if defined(__x86_64__) || defined(_M_X64)
    return tern_packed_matmul_f32_avx2(packed, input, output, M, N, B, alpha, bias);
#elif defined(__aarch64__) || defined(_M_ARM64)
    return tern_packed_matmul_f32_neon(packed, input, output, M, N, B, alpha, bias);
#else
    (void)packed; (void)input; (void)output;
    (void)M; (void)N; (void)B; (void)alpha; (void)bias;
    return TERN_ERR_NULL;
#endif
}

/* ── Test 1: SIMD detection ──────────────────────────────────────── */
static void test_simd_detection(void)
{
    uint32_t caps = get_simd_support();
    ASSERT(caps & TERN_SIMD_SCALAR, "SCALAR flag always set");

#if defined(__x86_64__) || defined(_M_X64)
    printf("  x86_64: AVX2=%s, AVX512=%s\n",
           (caps & TERN_SIMD_AVX2) ? "yes" : "no",
           (caps & TERN_SIMD_AVX512) ? "yes" : "no");
#elif defined(__aarch64__) || defined(_M_ARM64)
    ASSERT(caps & TERN_SIMD_NEON, "NEON flag set on AArch64");
    printf("  AArch64: NEON=yes\n");
#endif
}

/* ── Test 2: Bit-identical — small 2×8 ───────────────────────────── */
static void test_bit_identical_small(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    int8_t W[16] = {1, -1, 0, 1, -1, 0, 1, 0,
                    0, 1, -1, 0, 1, -1, 0, 1};
    uint8_t packed[4];
    pack_weights(W, packed, 16);

    float input[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float bias[2] = {0.01f, 0.02f};
    float scalar_out[2], simd_out[2];

    tern_packed_matvec_f32(packed, input, scalar_out, 2, 8, 0.73f, bias);
    run_simd_matvec(packed, input, simd_out, 2, 8, 0.73f, bias);

    for (int i = 0; i < 2; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical small");
    }
}

/* ── Test 3: Bit-identical — medium 16×64, sparse pattern ────────── */
static void test_bit_identical_medium(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 16, N = 64;
    int8_t W[M * N];
    fill_sparse_pattern(W, M * N);

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.01f * (float)(j + 1);

    float bias[M];
    for (int i = 0; i < M; i++) bias[i] = 0.001f * (float)(i + 1);

    float scalar_out[M], simd_out[M];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 0.85f, bias);
    run_simd_matvec(packed, input, simd_out, M, N, 0.85f, bias);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical medium");
    }
}

/* ── Test 4: Bit-identical — large 64×256 ────────────────────────── */
static void test_bit_identical_large(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 64, N = 256;
    int8_t W[M * N];
    fill_sparse_pattern(W, M * N);

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.005f * (float)(j - 128);

    float bias[M];
    for (int i = 0; i < M; i++) bias[i] = -0.01f * (float)(i + 1);

    float scalar_out[M], simd_out[M];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 0.42f, bias);
    run_simd_matvec(packed, input, simd_out, M, N, 0.42f, bias);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical large");
    }
}

/* ── Test 5: Bit-identical — N not multiple of 8 (tail path) ─────── */
static void test_bit_identical_tail(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 4, N = 12;  /* N/4=3 packed bytes, 1 pair + 1 tail */
    int8_t W[M * N];
    fill_sparse_pattern(W, M * N);

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.1f * (float)(j + 1);

    float scalar_out[M], simd_out[M];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 1.0f, NULL);
    run_simd_matvec(packed, input, simd_out, M, N, 1.0f, NULL);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical tail");
    }
}

/* ── Test 6: Bit-identical — N=4 (minimum, tail-only) ────────────── */
static void test_bit_identical_n4(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 3, N = 4;
    int8_t W[12] = {1, -1, 1, -1,  0, 1, 0, -1,  1, 0, 0, 1};
    uint8_t packed[3];
    pack_weights(W, packed, 12);

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias[3] = {0.5f, -0.5f, 0.1f};
    float scalar_out[3], simd_out[3];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 0.6f, bias);
    run_simd_matvec(packed, input, simd_out, M, N, 0.6f, bias);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical N=4");
    }
}

/* ── Test 7: Bit-identical — all zeros ───────────────────────────── */
static void test_bit_identical_all_zeros(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    #define AZ_M 4
    #define AZ_N 16
    uint8_t packed[AZ_M * AZ_N / 4];
    memset(packed, 0, sizeof(packed));

    float input[AZ_N];
    for (int j = 0; j < AZ_N; j++) input[j] = (float)(j + 1);

    float bias[AZ_M] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar_out[AZ_M], simd_out[AZ_M];

    tern_packed_matvec_f32(packed, input, scalar_out, AZ_M, AZ_N, 0.5f, bias);
    run_simd_matvec(packed, input, simd_out, AZ_M, AZ_N, 0.5f, bias);

    for (int i = 0; i < AZ_M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical all-zeros");
        ASSERT(simd_out[i] == bias[i], "all-zeros output equals bias");
    }
    #undef AZ_M
    #undef AZ_N
}

/* ── Test 8: Bit-identical — all positive ────────────────────────── */
static void test_bit_identical_all_positive(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 4, N = 16;
    int8_t W[M * N];
    for (int k = 0; k < M * N; k++) W[k] = 1;

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.1f * (float)(j + 1);

    float scalar_out[M], simd_out[M];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 1.0f, NULL);
    run_simd_matvec(packed, input, simd_out, M, N, 1.0f, NULL);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical all-positive");
    }
}

/* ── Test 9: Bit-identical — all negative ────────────────────────── */
static void test_bit_identical_all_negative(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 4, N = 16;
    int8_t W[M * N];
    for (int k = 0; k < M * N; k++) W[k] = -1;

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.1f * (float)(j + 1);

    float scalar_out[M], simd_out[M];

    tern_packed_matvec_f32(packed, input, scalar_out, M, N, 0.9f, NULL);
    run_simd_matvec(packed, input, simd_out, M, N, 0.9f, NULL);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "bit-identical all-negative");
    }
}

/* ── Test 10: Bit-identical — batched ────────────────────────────── */
static void test_bit_identical_batched(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 8, N = 32, B = 4;
    int8_t W[M * N];
    fill_sparse_pattern(W, M * N);

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[B * N];
    for (int k = 0; k < B * N; k++) input[k] = 0.01f * (float)(k - 64);

    float bias[M];
    for (int i = 0; i < M; i++) bias[i] = 0.1f * (float)(i + 1);

    float scalar_out[B * M], simd_out[B * M];

    tern_packed_matmul_f32(packed, input, scalar_out, M, N, B, 0.77f, bias);
    run_simd_matmul(packed, input, simd_out, M, N, B, 0.77f, bias);

    for (int k = 0; k < B * M; k++) {
        ASSERT(simd_out[k] == scalar_out[k], "bit-identical batched");
    }
}

/* ── Test 11: Determinism — 100 runs ─────────────────────────────── */
static void test_simd_determinism(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    const int M = 8, N = 64;
    int8_t W[M * N];
    fill_sparse_pattern(W, M * N);

    uint8_t packed[M * N / 4];
    pack_weights(W, packed, M * N);

    float input[N];
    for (int j = 0; j < N; j++) input[j] = 0.02f * (float)(j + 1);

    float reference[M], output[M];

    run_simd_matvec(packed, input, reference, M, N, 0.65f, NULL);

    for (int run = 0; run < 99; run++) {
        run_simd_matvec(packed, input, output, M, N, 0.65f, NULL);
        for (int i = 0; i < M; i++) {
            ASSERT(output[i] == reference[i], "SIMD deterministic 100 runs");
        }
    }
}

/* ── Test 12: Dispatch via ternary_matmul_f32_simd ───────────────── */
static void test_dispatch_uses_simd(void)
{
    if (!have_simd_kernel()) { printf("  [skipped — no SIMD]\n"); return; }

    #define DU_M 4
    #define DU_N 16
    #define DU_B 2
    int8_t W[DU_M * DU_N];
    fill_sparse_pattern(W, DU_M * DU_N);

    uint8_t packed[DU_M * DU_N / 4];
    pack_weights(W, packed, DU_M * DU_N);

    float input[DU_B * DU_N];
    for (int k = 0; k < DU_B * DU_N; k++) input[k] = 0.05f * (float)(k + 1);

    float bias[DU_M] = {0.1f, 0.2f, 0.3f, 0.4f};
    float dispatch_out[DU_B * DU_M], direct_out[DU_B * DU_M];

    /* Call dispatch (bitmap=NULL → dense SIMD path) */
    ternary_matmul_f32_simd(packed, input, dispatch_out, NULL,
                            0.55f, bias, DU_M, DU_N, DU_B);

    /* Call SIMD kernel directly */
    run_simd_matmul(packed, input, direct_out, DU_M, DU_N, DU_B, 0.55f, bias);

    for (int k = 0; k < DU_B * DU_M; k++) {
        ASSERT(dispatch_out[k] == direct_out[k], "dispatch matches direct SIMD");
    }
    #undef DU_M
    #undef DU_N
    #undef DU_B
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void)
{
    printf("Running SIMD kernel tests...\n\n");

    test_simd_detection();
    test_bit_identical_small();
    test_bit_identical_medium();
    test_bit_identical_large();
    test_bit_identical_tail();
    test_bit_identical_n4();
    test_bit_identical_all_zeros();
    test_bit_identical_all_positive();
    test_bit_identical_all_negative();
    test_bit_identical_batched();
    test_simd_determinism();
    test_dispatch_uses_simd();

    printf("\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED (0 failures)\n");
    } else {
        printf("TESTS FAILED (%d failure%s)\n",
               failures, failures == 1 ? "" : "s");
    }

    return failures > 0 ? 1 : 0;
}
