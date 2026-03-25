// ternary_engine.h
// C interface for the Metal ternary inference engine
// Terncore · Cubey/Synapticode · 2026

#ifndef TERNARY_ENGINE_H
#define TERNARY_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the Metal engine
typedef struct TernaryEngine TernaryEngine;

// Opaque handle to a GPU buffer
typedef struct TernaryBuffer TernaryBuffer;

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

// Create engine, compile Metal kernels. Returns NULL on failure.
TernaryEngine* tern_engine_create(void);

// Destroy engine and release Metal resources.
void tern_engine_destroy(TernaryEngine* engine);

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

// Allocate GPU buffer and optionally copy data. data=NULL for empty buffer.
TernaryBuffer* tern_buffer_create(TernaryEngine* engine,
                                  const void* data,
                                  size_t bytes);

// Release GPU buffer.
void tern_buffer_destroy(TernaryBuffer* buf);

// Copy GPU buffer contents back to host memory.
void tern_buffer_read(TernaryBuffer* buf, void* dst, size_t bytes);

// Get raw byte size of buffer.
size_t tern_buffer_size(TernaryBuffer* buf);

// ---------------------------------------------------------------------------
// Ternary matrix-vector multiply
// ---------------------------------------------------------------------------
// Computes: output[b, m] = scale[m] * sum_k( ternary(W[m,k]) * input[b,k] )
//
// packed_codes: (M, ceil(K/16)) packed uint32, 2 bits per ternary value
// scales:       (M,) float32 per-channel scale
// input:        (B, K) float16
// output:       (B, M) float16
//
// Returns 0 on success, nonzero on error.
// ---------------------------------------------------------------------------

int tern_matvec(TernaryEngine* engine,
                TernaryBuffer* packed_codes,
                TernaryBuffer* scales,
                TernaryBuffer* input,
                TernaryBuffer* output,
                uint32_t M, uint32_t K, uint32_t B);

// Fast variant — requires K % 16 == 0 (true for all transformer sizes)
int tern_matvec_fast(TernaryEngine* engine,
                     TernaryBuffer* packed_codes,
                     TernaryBuffer* scales,
                     TernaryBuffer* input,
                     TernaryBuffer* output,
                     uint32_t M, uint32_t K, uint32_t B);

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

// Wait for all queued GPU work to complete.
void tern_sync(TernaryEngine* engine);

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// Get GPU name string (caller must not free).
const char* tern_device_name(TernaryEngine* engine);

// Get last error message (caller must not free).
const char* tern_last_error(TernaryEngine* engine);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_ENGINE_H
