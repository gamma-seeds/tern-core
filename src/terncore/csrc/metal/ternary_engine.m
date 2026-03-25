// ternary_engine.m
// Objective-C Metal bridge for ternary inference engine
// Compiles MSL kernels, manages GPU buffers, dispatches compute
//
// Terncore · Cubey/Synapticode · 2026

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ternary_engine.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

struct TernaryEngine {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> pipeline_matvec;
    id<MTLComputePipelineState> pipeline_matvec_fast;
    char                       device_name[256];
    char                       last_error[512];
};

struct TernaryBuffer {
    id<MTLBuffer> buffer;
    size_t        size;
};

// Params struct must match MSL definition
typedef struct {
    uint32_t M;
    uint32_t K;
    uint32_t B;
    uint32_t packed_K;
} MatmulParams;

// ---------------------------------------------------------------------------
// Helper: create pipeline with function constants
// ---------------------------------------------------------------------------

static id<MTLComputePipelineState>
create_pipeline(TernaryEngine* engine,
                NSString* func_name,
                uint32_t threadgroup_size,
                uint32_t rows_per_group)
{
    // Set function constants
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&threadgroup_size type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&rows_per_group  type:MTLDataTypeUInt atIndex:1];

    NSError* error = nil;
    id<MTLFunction> func = [engine->library newFunctionWithName:func_name
                                                 constantValues:constants
                                                          error:&error];
    if (!func) {
        snprintf(engine->last_error, sizeof(engine->last_error),
                 "Failed to create function '%s': %s",
                 [func_name UTF8String],
                 error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [engine->device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        snprintf(engine->last_error, sizeof(engine->last_error),
                 "Failed to create pipeline '%s': %s",
                 [func_name UTF8String],
                 error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
    }

    return pipeline;
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

TernaryEngine* tern_engine_create(void) {
    TernaryEngine* engine = (TernaryEngine*)calloc(1, sizeof(TernaryEngine));
    if (!engine) return NULL;

    @autoreleasepool {
        // Get default GPU
        engine->device = MTLCreateSystemDefaultDevice();
        if (!engine->device) {
            snprintf(engine->last_error, sizeof(engine->last_error),
                     "No Metal device available");
            free(engine);
            return NULL;
        }

        const char* name = [[engine->device name] UTF8String];
        strncpy(engine->device_name, name, sizeof(engine->device_name) - 1);

        // Create command queue
        engine->queue = [engine->device newCommandQueue];
        if (!engine->queue) {
            snprintf(engine->last_error, sizeof(engine->last_error),
                     "Failed to create command queue");
            free(engine);
            return NULL;
        }

        // Load Metal source from adjacent .metal file
        NSString* dir = [[[NSString stringWithUTF8String:__FILE__]
                          stringByDeletingLastPathComponent]
                         stringByAppendingPathComponent:@"ternary_matmul.metal"];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:dir
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            // Try loading from metallib next to the dylib
            NSString* libDir = [[[NSProcessInfo processInfo] arguments][0]
                                stringByDeletingLastPathComponent];
            NSString* metallib = [libDir stringByAppendingPathComponent:@"ternary_matmul.metallib"];
            engine->library = [engine->device newLibraryWithFile:metallib error:&error];

            if (!engine->library) {
                // Try compiling from source embedded at build time
                snprintf(engine->last_error, sizeof(engine->last_error),
                         "Failed to load Metal source: %s",
                         error ? [[error localizedDescription] UTF8String] : "file not found");
                free(engine);
                return NULL;
            }
        } else {
            // Compile from source
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            opts.languageVersion = MTLLanguageVersion3_1;

            engine->library = [engine->device newLibraryWithSource:source
                                                           options:opts
                                                             error:&error];
            if (!engine->library) {
                snprintf(engine->last_error, sizeof(engine->last_error),
                         "Metal compilation failed: %s",
                         [[error localizedDescription] UTF8String]);
                free(engine);
                return NULL;
            }
        }

        // Create compute pipelines with function constants
        uint32_t tg_size = 256;
        uint32_t rows_per_group = 4;

        engine->pipeline_matvec = create_pipeline(
            engine, @"ternary_matvec", tg_size, rows_per_group);
        if (!engine->pipeline_matvec) {
            free(engine);
            return NULL;
        }

        engine->pipeline_matvec_fast = create_pipeline(
            engine, @"ternary_matvec_fast", tg_size, rows_per_group);
        if (!engine->pipeline_matvec_fast) {
            free(engine);
            return NULL;
        }
    }

    return engine;
}

void tern_engine_destroy(TernaryEngine* engine) {
    if (!engine) return;
    // ARC handles Metal object cleanup
    free(engine);
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

TernaryBuffer* tern_buffer_create(TernaryEngine* engine,
                                  const void* data,
                                  size_t bytes) {
    if (!engine || bytes == 0) return NULL;

    TernaryBuffer* buf = (TernaryBuffer*)calloc(1, sizeof(TernaryBuffer));
    if (!buf) return NULL;

    @autoreleasepool {
        if (data) {
            buf->buffer = [engine->device newBufferWithBytes:data
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared];
        } else {
            buf->buffer = [engine->device newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
        }
    }

    if (!buf->buffer) {
        free(buf);
        return NULL;
    }
    buf->size = bytes;
    return buf;
}

void tern_buffer_destroy(TernaryBuffer* buf) {
    if (!buf) return;
    // ARC releases the MTLBuffer
    free(buf);
}

void tern_buffer_read(TernaryBuffer* buf, void* dst, size_t bytes) {
    if (!buf || !dst) return;
    size_t copy_size = bytes < buf->size ? bytes : buf->size;
    memcpy(dst, [buf->buffer contents], copy_size);
}

size_t tern_buffer_size(TernaryBuffer* buf) {
    return buf ? buf->size : 0;
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

static int dispatch_matvec(TernaryEngine* engine,
                           id<MTLComputePipelineState> pipeline,
                           TernaryBuffer* packed_codes,
                           TernaryBuffer* scales,
                           TernaryBuffer* input,
                           TernaryBuffer* output,
                           uint32_t M, uint32_t K, uint32_t B)
{
    if (!engine || !packed_codes || !scales || !input || !output) {
        snprintf(engine->last_error, sizeof(engine->last_error), "NULL argument");
        return -1;
    }

    @autoreleasepool {
        MatmulParams params;
        params.M = M;
        params.K = K;
        params.B = B;
        params.packed_K = (K + 15) / 16;

        uint32_t rows_per_group = 4;
        uint32_t threadgroup_size = 256;

        id<MTLCommandBuffer> cmd = [engine->queue commandBuffer];
        if (!cmd) {
            snprintf(engine->last_error, sizeof(engine->last_error),
                     "Failed to create command buffer");
            return -1;
        }

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:packed_codes->buffer offset:0 atIndex:0];
        [enc setBuffer:scales->buffer        offset:0 atIndex:1];
        [enc setBuffer:input->buffer         offset:0 atIndex:2];
        [enc setBuffer:output->buffer        offset:0 atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];

        MTLSize grid = MTLSizeMake((M + rows_per_group - 1) / rows_per_group, B, 1);
        MTLSize tg   = MTLSizeMake(threadgroup_size, 1, 1);

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] == MTLCommandBufferStatusError) {
            NSError* error = [cmd error];
            snprintf(engine->last_error, sizeof(engine->last_error),
                     "GPU execution error: %s",
                     error ? [[error localizedDescription] UTF8String] : "unknown");
            return -1;
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Public dispatch functions
// ---------------------------------------------------------------------------

int tern_matvec(TernaryEngine* engine,
                TernaryBuffer* packed_codes,
                TernaryBuffer* scales,
                TernaryBuffer* input,
                TernaryBuffer* output,
                uint32_t M, uint32_t K, uint32_t B)
{
    return dispatch_matvec(engine, engine->pipeline_matvec,
                           packed_codes, scales, input, output, M, K, B);
}

int tern_matvec_fast(TernaryEngine* engine,
                     TernaryBuffer* packed_codes,
                     TernaryBuffer* scales,
                     TernaryBuffer* input,
                     TernaryBuffer* output,
                     uint32_t M, uint32_t K, uint32_t B)
{
    return dispatch_matvec(engine, engine->pipeline_matvec_fast,
                           packed_codes, scales, input, output, M, K, B);
}

// ---------------------------------------------------------------------------
// Synchronization & diagnostics
// ---------------------------------------------------------------------------

void tern_sync(TernaryEngine* engine) {
    if (!engine) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [engine->queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

const char* tern_device_name(TernaryEngine* engine) {
    return engine ? engine->device_name : "unknown";
}

const char* tern_last_error(TernaryEngine* engine) {
    return engine ? engine->last_error : "no engine";
}
