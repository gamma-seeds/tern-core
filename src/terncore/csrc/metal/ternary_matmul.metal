// ternary_matmul.metal
// Native Metal compute kernel for ternary neural network inference
// Operates directly on packed 2-bit ternary codes — no dequantization
//
// Encoding (per uint32, 16 ternary values):
//   bits [2i+1 : 2i] for value i (i = 0..15)
//   0b00 = zero weight (skip — 43% of all weights)
//   0b01 = +1 (add input)
//   0b10 = -1 (subtract input)
//
// Core trick: sign = float(code & 1) - float(code >> 1)
//   code 00 → 0-0 = 0  (zero: no contribution)
//   code 01 → 1-0 = +1 (add)
//   code 10 → 0-1 = -1 (subtract)
// No branches. No FMA. Pure add/subtract with zero-skip for free.
//
// Terncore · Cubey/Synapticode · Apple Silicon · 2026

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant uint THREADGROUP_SIZE [[function_constant(0)]];  // set at pipeline creation
constant uint ROWS_PER_GROUP  [[function_constant(1)]];   // output rows per threadgroup

// Fallback defaults (overridden by function constants)
// THREADGROUP_SIZE = 256, ROWS_PER_GROUP = 4

// ---------------------------------------------------------------------------
// Kernel: ternary_matvec
// ---------------------------------------------------------------------------
// Computes: output[b, m] = scale[m] * sum_k( ternary(W[m,k]) * input[b,k] )
//
// Grid:     (ceil(M / ROWS_PER_GROUP), B, 1)
// Threads:  (THREADGROUP_SIZE, 1, 1) per threadgroup
//
// Each threadgroup processes ROWS_PER_GROUP output rows.
// Threads cooperate on the K (input) dimension via SIMD + shared memory reduction.
// ---------------------------------------------------------------------------

struct MatmulParams {
    uint M;          // output features
    uint K;          // input features
    uint B;          // batch size
    uint packed_K;   // ceil(K / 16) — uint32s per row
};

kernel void ternary_matvec(
    device const uint32_t*  packed_codes  [[buffer(0)]],  // (M, packed_K)
    device const float*     scales        [[buffer(1)]],  // (M,)
    device const half*      input         [[buffer(2)]],  // (B, K)
    device       half*      output        [[buffer(3)]],  // (B, M)
    constant MatmulParams&  params        [[buffer(4)]],
    uint2  group_id    [[threadgroup_position_in_grid]],
    uint   tid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_idx    [[simdgroup_index_in_threadgroup]])
{
    const uint row_base = group_id.x * ROWS_PER_GROUP;
    const uint batch    = group_id.y;
    const uint M        = params.M;
    const uint K        = params.K;
    const uint pK       = params.packed_K;

    // Pointer to this batch's input vector
    device const half* x = input + batch * K;

    // Shared memory for SIMD group partial sums
    // Max 32 SIMD groups × ROWS_PER_GROUP rows
    threadgroup float partial[32 * 4];  // sized for max ROWS_PER_GROUP=4, 32 simd groups

    const uint num_simd_groups = (THREADGROUP_SIZE + 31) / 32;

    // Accumulate over K for each output row
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        const uint row = row_base + r;
        if (row >= M) break;

        device const uint32_t* row_codes = packed_codes + row * pK;
        float acc = 0.0f;

        // Each thread strides over packed words
        for (uint p = tid; p < pK; p += THREADGROUP_SIZE) {
            uint32_t packed = row_codes[p];
            uint base_k = p * 16;

            // Fast exit: if entire packed word is zero, skip all 16 values
            if (packed == 0) continue;

            // Unroll 16 ternary values from packed uint32
            // Compiler will unroll this constant-bound loop
            uint remaining = min(uint(16), K - base_k);
            for (uint i = 0; i < remaining; i++) {
                uint code = (packed >> (i * 2)) & 0x3;
                // Branch-free ternary decode:
                // code 00 → sign = 0  (zero weight, contributes nothing)
                // code 01 → sign = +1 (add input)
                // code 10 → sign = -1 (subtract input)
                float sign = float(code & 1) - float(code >> 1);
                acc += sign * float(x[base_k + i]);
            }
        }

        // --- Reduction ---
        // Step 1: SIMD-level reduction (hardware-accelerated, 32-wide)
        acc = simd_sum(acc);

        // Step 2: Write SIMD group result to shared memory
        if (simd_lane == 0) {
            partial[simd_idx * ROWS_PER_GROUP + r] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: First thread sums across SIMD groups and writes output
        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < num_simd_groups; s++) {
                total += partial[s * ROWS_PER_GROUP + r];
            }
            output[batch * M + row] = half(total * scales[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ---------------------------------------------------------------------------
// Kernel: ternary_matvec_fast
// ---------------------------------------------------------------------------
// Optimized variant using vectorized half4 loads and aggressive unrolling.
// Requires K % 16 == 0 (true for all transformer layer sizes).
// ---------------------------------------------------------------------------

kernel void ternary_matvec_fast(
    device const uint32_t*  packed_codes  [[buffer(0)]],
    device const float*     scales        [[buffer(1)]],
    device const half*      input         [[buffer(2)]],
    device       half*      output        [[buffer(3)]],
    constant MatmulParams&  params        [[buffer(4)]],
    uint2  group_id    [[threadgroup_position_in_grid]],
    uint   tid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_idx    [[simdgroup_index_in_threadgroup]])
{
    const uint row_base = group_id.x * ROWS_PER_GROUP;
    const uint batch    = group_id.y;
    const uint M        = params.M;
    const uint K        = params.K;
    const uint pK       = params.packed_K;

    device const half* x = input + batch * K;

    threadgroup float partial[32 * 4];
    const uint num_simd_groups = (THREADGROUP_SIZE + 31) / 32;

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        const uint row = row_base + r;
        if (row >= M) break;

        device const uint32_t* row_codes = packed_codes + row * pK;
        float acc = 0.0f;

        for (uint p = tid; p < pK; p += THREADGROUP_SIZE) {
            uint32_t packed = row_codes[p];

            // Skip zero word (pure sparsity win — 43% of weights are zero)
            if (packed == 0) continue;

            uint base_k = p * 16;

            // Vectorized load: 16 half values as 4 × half4
            half4 v0 = *reinterpret_cast<device const half4*>(x + base_k);
            half4 v1 = *reinterpret_cast<device const half4*>(x + base_k + 4);
            half4 v2 = *reinterpret_cast<device const half4*>(x + base_k + 8);
            half4 v3 = *reinterpret_cast<device const half4*>(x + base_k + 12);

            // Decode and accumulate 4 groups of 4 values each
            // Group 0: bits [7:0]
            {
                uint c0 = packed & 0x3; uint c1 = (packed >> 2) & 0x3;
                uint c2 = (packed >> 4) & 0x3; uint c3 = (packed >> 6) & 0x3;
                acc += (float(c0 & 1) - float(c0 >> 1)) * float(v0[0]);
                acc += (float(c1 & 1) - float(c1 >> 1)) * float(v0[1]);
                acc += (float(c2 & 1) - float(c2 >> 1)) * float(v0[2]);
                acc += (float(c3 & 1) - float(c3 >> 1)) * float(v0[3]);
            }
            // Group 1: bits [15:8]
            {
                uint c0 = (packed >> 8) & 0x3; uint c1 = (packed >> 10) & 0x3;
                uint c2 = (packed >> 12) & 0x3; uint c3 = (packed >> 14) & 0x3;
                acc += (float(c0 & 1) - float(c0 >> 1)) * float(v1[0]);
                acc += (float(c1 & 1) - float(c1 >> 1)) * float(v1[1]);
                acc += (float(c2 & 1) - float(c2 >> 1)) * float(v1[2]);
                acc += (float(c3 & 1) - float(c3 >> 1)) * float(v1[3]);
            }
            // Group 2: bits [23:16]
            {
                uint c0 = (packed >> 16) & 0x3; uint c1 = (packed >> 18) & 0x3;
                uint c2 = (packed >> 20) & 0x3; uint c3 = (packed >> 22) & 0x3;
                acc += (float(c0 & 1) - float(c0 >> 1)) * float(v2[0]);
                acc += (float(c1 & 1) - float(c1 >> 1)) * float(v2[1]);
                acc += (float(c2 & 1) - float(c2 >> 1)) * float(v2[2]);
                acc += (float(c3 & 1) - float(c3 >> 1)) * float(v2[3]);
            }
            // Group 3: bits [31:24]
            {
                uint c0 = (packed >> 24) & 0x3; uint c1 = (packed >> 26) & 0x3;
                uint c2 = (packed >> 28) & 0x3; uint c3 = (packed >> 30) & 0x3;
                acc += (float(c0 & 1) - float(c0 >> 1)) * float(v3[0]);
                acc += (float(c1 & 1) - float(c1 >> 1)) * float(v3[1]);
                acc += (float(c2 & 1) - float(c2 >> 1)) * float(v3[2]);
                acc += (float(c3 & 1) - float(c3 >> 1)) * float(v3[3]);
            }
        }

        // Reduction
        acc = simd_sum(acc);
        if (simd_lane == 0) {
            partial[simd_idx * ROWS_PER_GROUP + r] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < num_simd_groups; s++) {
                total += partial[s * ROWS_PER_GROUP + r];
            }
            output[batch * M + row] = half(total * scales[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ---------------------------------------------------------------------------
// Kernel: ternary_matmul_tiled
// ---------------------------------------------------------------------------
// Batched version for prefill (B > 1). Tiles over both M and B.
// Grid: (ceil(M / TILE_M), ceil(B / TILE_B), 1)
// Each thread computes one output element.
// ---------------------------------------------------------------------------

constant uint TILE_M = 8;
constant uint TILE_B = 32;

kernel void ternary_matmul_tiled(
    device const uint32_t*  packed_codes  [[buffer(0)]],
    device const float*     scales        [[buffer(1)]],
    device const half*      input         [[buffer(2)]],
    device       half*      output        [[buffer(3)]],
    constant MatmulParams&  params        [[buffer(4)]],
    uint3  gid  [[thread_position_in_grid]],
    uint3  tid  [[thread_position_in_threadgroup]])
{
    const uint row   = gid.x;  // output feature index
    const uint batch = gid.y;  // batch index
    const uint M     = params.M;
    const uint K     = params.K;
    const uint pK    = params.packed_K;

    if (row >= M || batch >= params.B) return;

    device const uint32_t* row_codes = packed_codes + row * pK;
    device const half* x = input + batch * K;

    float acc = 0.0f;

    for (uint p = 0; p < pK; p++) {
        uint32_t packed = row_codes[p];
        if (packed == 0) continue;

        uint base_k = p * 16;
        uint remaining = min(uint(16), K - base_k);

        for (uint i = 0; i < remaining; i++) {
            uint code = (packed >> (i * 2)) & 0x3;
            float sign = float(code & 1) - float(code >> 1);
            acc += sign * float(x[base_k + i]);
        }
    }

    output[batch * M + row] = half(acc * scales[row]);
}
