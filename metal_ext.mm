#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <tuple>
#include <unordered_map>

namespace py = pybind11;

struct MetalBufferHandle {
    id<MTLBuffer> buffer;
    NSUInteger offset;
};

struct AdamWParams {
    float decay_factor;
    float lr_t;
    float b1;
    float b2;
    float eps_sq;
};

class MetalContext {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    // rows, columns, matrices, rowBytes, matrixBytes
    using MatrixDescriptorKey = std::tuple<NSUInteger, NSUInteger, NSUInteger, NSUInteger, NSUInteger, NSUInteger>;
    // buffer_id, buffer offset
    using MatrixViewKey = std::tuple<uintptr_t, NSUInteger, MatrixDescriptorKey>;
    // batch count, M, N, K, transpose A, transpose B, alpha bits, beta bits
    using GemmKey = std::tuple<NSUInteger, NSUInteger, NSUInteger, NSUInteger, bool, bool, uint32_t, uint32_t, NSUInteger>;
    std::map<MatrixDescriptorKey, MPSMatrixDescriptor*> matrix_descriptor_cache;
    std::map<MatrixViewKey, MPSMatrix*> matrix_view_cache;
    std::map<GemmKey, MPSMatrixMultiplication*> matrix_multiplication_cache;

    std::chrono::steady_clock::time_point profile_begin_time;
    bool profile_active = false;

    id<MTLCounterSampleBuffer> active_counter_buffer = nil;
    id<MTLCounterSet> active_counter_set = nil;

    enum class CounterSamplingMode {
        None,
        DispatchBoundary,
        StageBoundary,
    };
    CounterSamplingMode active_counter_mode = CounterSamplingMode::None;

public:
    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        
        std::ifstream file("metal_ops.metal");
        if (!file.is_open()) throw std::runtime_error("Failed to open metal_ops.metal");
        std::stringstream buffer;
        buffer << file.rdbuf();
        NSString *source = [NSString stringWithUTF8String:buffer.str().c_str()];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion3_0;
        NSError *compile_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&compile_error];
        if (library == nil) {
            NSLog(@"Error compiling metal library: %@", compile_error);
            throw std::runtime_error("Failed to compile metal library");
        }

        NSArray<NSString*> *kernel_names = @[
            @"add_residual_forward_vec4", @"add_residual_forward", @"add_pos_forward", @"add_bias_forward",
            @"add_residual_backward_vec4", @"add_residual_backward", @"contiguous_backward_kernel", @"pos_emb_grad",
            @"mul_forward_kernel", @"mul_backward_kernel", @"bias_grad_reduce", @"bias_grad_reduce_bf16",
            @"mul_backward_scalar_dx", @"mul_backward_scalar_dy",
            @"gather_forward_kernel", @"gather_backward_kernel", @"gather_backward_bf16_kernel",
            @"strided_add_backward_kernel", 
            @"contiguous_forward_kernel", 
            @"rmsnorm_forward", @"layernorm_forward",
            @"mlp_backward_dx",
            @"layernorm_backward_kernel",
            @"gelu_add_bias_kernel", @"gelu_backward_bias_kernel", 
            @"gelu_forward_kernel", @"gelu_backward_kernel",
            @"swiglu_forward_kernel", @"swiglu_backward_inplace_kernel",
            @"swiglu_forward_tile_kernel",
            @"fused_rms_fa_forward_block64", @"fused_rms_fa_forward_block128",
            @"fused_linear_add_forward_kernel", 
            @"fused_rmsnorm_mlp_forward_kernel_bm16", @"fused_rmsnorm_mlp_forward_kernel_bm32",
            @"fill_scalar_kernel", @"fill_scalar_kernel2",
            @"rope_qk_inplace",
            @"fused_rms_fa_project_kv",
            @"rmsnorm_backward",
            @"rmsnorm_weighted_forward", @"rmsnorm_weighted_backward", @"rmsnorm_weighted_dgamma",
            @"fused_rms_fa_qkv", 
            @"fused_fa_from_qkv",
            @"fused_fa_from_qkv_d64", @"fused_fa_from_qkv_d128",
            @"fused_fa_bwd_prepare",  
            @"fused_rms_recompute",
            @"fused_fa_bwd_dq_block64", @"fused_fa_bwd_dq_block128", 
            @"fused_fa_bwd_dkv_block128", 
            @"fused_fa_bwd_dkv_mma64",
            @"fused_qkv_bias_grad", @"fused_qkv_bias_grad_bf16",
            @"fused_rmsnorm_bwd_dx",
            @"gqa_rms_qkv", @"gqa_from_qkv_d64", @"gqa_from_qkv_d128", @"gqa_bwd_dq", @"gqa_bwd_dkv", @"gqa_rope_inplace", 
            @"gqa_rms_recompute", @"gqa_rms_dgamma", @"gqa_rms_dx",
            @"kv_cache_rope_store", @"kv_cache_fa_forward_d64", @"kv_cache_fa_forward_d128", @"kv_cache_rope_store_broadcast",
            @"kv_cache_rope_store_rows", @"kv_cache_fa_forward_rows_d64", @"kv_cache_fa_forward_rows_d128",
            @"fused_linear_ce_logits_to_dlogits", @"fused_linear_ce_logits_to_dlogits_row_scaled",
            @"fused_linear_ce_forward_reduce",
            @"fused_linear_ce_chunk_logits", 
            @"ce_tile_logits_to_dlogits", @"fp32_to_bf16",
            @"fused_rmsnorm_linear_ce_dx",
            @"fused_rmsnorm_linear_ce_restore_target", 
            @"fused_rmsnorm_linear_ce_subtract_target",
            @"fused_rmsnorm_linear_ce_dw_tail",
            @"fused_rmsnorm_linear_ce_dw_bf16", @"gqa_rms_dgamma_bf16",
            @"rollout_sample_from_topk", @"rollout_topk_select", @"rollout_decode_state", @"rollout_decode_finished",
            @"grad_norm_sq",
            @"adamw_step", @"adamw8bit_step",
            @"mini_update_v", @"adamw_mini8bit_step",
        ];
        for (NSString* name in kernel_names) {
            id<MTLFunction> func = [library newFunctionWithName:name];
            if (func == nil) {
                NSLog(@"Error: Failed to find function %@", name);
                throw std::runtime_error("Failed to find function in metal library");
            }
            NSError *error = nil;
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
            if (error != nil) {
                NSLog(@"Error creating pipeline state for %@: %@", name, error);
                throw std::runtime_error("Failed to create pipeline state");
            }
            pipelines[[name UTF8String]] = pso;
        }
    }

    size_t max_buffer_length() const {
        return static_cast<size_t>(device.maxBufferLength);
    }

    uintptr_t alloc(size_t bytes) {
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        return (uintptr_t)(__bridge_retained void*)buf;
    }

    void free(uintptr_t buffer_id) {
        auto iterator = matrix_view_cache.begin();
        while (iterator != matrix_view_cache.end()) {
            if (std::get<0>(iterator->first) == buffer_id) {
                iterator = matrix_view_cache.erase(iterator);
            } else {
                ++iterator;
            }
        }

        (__bridge_transfer id<MTLBuffer>)(void*)buffer_id;
    }

    void begin_pass() {
        cmd = [queue commandBuffer];
        enc = [cmd computeCommandEncoder];
    }

    void end_pass() {
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        cmd = nil;
        enc = nil;
    }

    void begin_pass_profiled() {
        profile_begin_time = std::chrono::steady_clock::now();
        profile_active = true;
        active_counter_mode = CounterSamplingMode::None;

        cmd = [queue commandBuffer];
        enc = [cmd computeCommandEncoder];
    }

    py::dict end_pass_profiled() {
        if (active_counter_buffer != nil && active_counter_mode == CounterSamplingMode::DispatchBoundary) {
            [enc sampleCountersInBuffer:active_counter_buffer atSampleIndex:1 withBarrier:YES];
        }
        [enc endEncoding];
        auto encode_done = std::chrono::steady_clock::now();

        id<MTLCommandBuffer> completed_cmd = cmd;
        auto commit_begin = std::chrono::steady_clock::now();
        [completed_cmd commit];
        auto commit_done = std::chrono::steady_clock::now();

        [completed_cmd waitUntilCompleted];
        auto wait_done = std::chrono::steady_clock::now();
        auto cpu_ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };

        auto gpu_interval_ms = [](CFTimeInterval begin, CFTimeInterval end) {
            if (begin <= 0 || end < begin) return -1.0;
            return (end - begin) * 1000.0;
        };

        py::dict result;
        result["cpu_total_ms"] = cpu_ms(profile_begin_time, wait_done);
        result["encode_ms"] = cpu_ms(profile_begin_time, encode_done);
        result["commit_ms"] = cpu_ms(commit_begin, commit_done);
        result["wait_ms"] = cpu_ms(commit_done, wait_done);
        result["gpu_ms"] = gpu_interval_ms(completed_cmd.GPUStartTime, completed_cmd.GPUEndTime);
        result["host_kernel_ms"] = gpu_interval_ms(completed_cmd.kernelStartTime, completed_cmd.kernelEndTime);
        result["status"] = static_cast<int>(completed_cmd.status);
        NSString *error_string = completed_cmd.error ? completed_cmd.error.localizedDescription : @"";
        result["error"] = std::string([error_string UTF8String]);

        switch(active_counter_mode) {
            case CounterSamplingMode::DispatchBoundary: 
                result["counter_sampling_mode"] = "dispatch_boundary";
                break;
            case CounterSamplingMode::StageBoundary:
                result["counter_sampling_mode"] = "stage_boundary";
                break;
            default:
                result["counter_sampling_mode"] = py::none();
                break;
        }

        if (active_counter_buffer != nil) {
            result["counter_set"] = std::string([active_counter_set.name UTF8String]);
            result["counter_deltas"] = resolve_counter_deltas();
        } else {
            result["counter_set"] = py::none();
            result["counter_deltas"] = py::dict();
        }
        active_counter_buffer = nil;
        active_counter_set = nil;
        active_counter_mode = CounterSamplingMode::None;

        cmd = nil;
        enc = nil;
        profile_active = false;

        return result;
    }

    py::dict counter_capacities() {
        py::dict result;

        bool dispatch_boundary = false;
        bool stage_boundary = false;

        if (@available(macOS 11.0, *)) {
            dispatch_boundary = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
            stage_boundary = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        }

        result["device"] = std::string([device.name UTF8String]);
        result["dispatch_boundary"] = dispatch_boundary;
        result["stage_boundary"] = stage_boundary;

        py::list sets;
        if (@available(macOS 10.15, *)) {
            for (id<MTLCounterSet> counter_set in device.counterSets) {
                py::dict set_info;
                set_info["name"] = std::string([counter_set.name UTF8String]);
                py::list counters;
                for (id<MTLCounter> counter in counter_set.counters) {
                    counters.append(std::string([counter.name UTF8String]));
                }

                set_info["counters"] = counters;
                sets.append(set_info);
            }
        }

        result["sets"] = sets;
        return result;
    }

    void begin_pass_counter_profiled(const std::string& requested_set_name) {
        if (!@available(macOS 11.0, *)) {
            throw std::runtime_error("Metal counter sampling requires macOS 11 or newer");
        }

        if (cmd != nil || enc != nil) {
            throw std::runtime_error("A Metal pass is already active");
        }

        const bool dispatch_supported = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        const bool stage_supported = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        if (!dispatch_supported && !stage_supported) {
            throw std::runtime_error("Device supports neither dispatch-boundary nor stage-boundary counter sampling");
        }

        NSString *requested_name = [NSString stringWithUTF8String:requested_set_name.c_str()];
        id<MTLCounterSet> selected_set = nil;
        for (id<MTLCounterSet> counter_set in device.counterSets) {
            if ([counter_set.name caseInsensitiveCompare:requested_name] == NSOrderedSame) {
                selected_set = counter_set;
                break;
            }
        }

        if (selected_set == nil) {
            throw std::runtime_error("Requested Metal Counter set was not found: " + requested_set_name);
        }

        MTLCounterSampleBufferDescriptor *descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.counterSet = selected_set;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 2;
        descriptor.label = @"autograd kernel counter samples";

        NSError *error = nil;
        id<MTLCounterSampleBuffer> sample_buffer = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if (sample_buffer == nil) {
            std::string message = "Failed to create MTLCounterSampleBuffer";
            if (error != nil) {
                const char *error_text = [error.localizedDescription UTF8String];
                message += ": ";
                message += error_text != nullptr ? error_text : "unknown error";
            }

            throw std::runtime_error(message);
        }

        active_counter_set = selected_set;
        active_counter_buffer = sample_buffer;

        profile_begin_time = std::chrono::steady_clock::now();
        profile_active = true;

        cmd = [queue commandBuffer];
        if (dispatch_supported) {
            active_counter_mode = CounterSamplingMode::DispatchBoundary;
            enc = [cmd computeCommandEncoder];
            if (enc == nil) {
                throw std::runtime_error("Failed to create compute encoder");
            }
            [enc sampleCountersInBuffer:active_counter_buffer atSampleIndex:0 withBarrier:YES];
        } else {
            active_counter_mode = CounterSamplingMode::StageBoundary;
            MTLComputePassDescriptor *pass_descriptor = [MTLComputePassDescriptor computePassDescriptor];
            MTLComputePassSampleBufferAttachmentDescriptor *attachment = [pass_descriptor.sampleBufferAttachments objectAtIndexedSubscript:0];
            attachment.sampleBuffer = active_counter_buffer;
            attachment.startOfEncoderSampleIndex = 0;
            attachment.endOfEncoderSampleIndex = 1;
            enc = [cmd computeCommandEncoderWithDescriptor:pass_descriptor];

            if (enc == nil) {
                active_counter_buffer = nil;
                active_counter_set = nil;
                active_counter_mode = CounterSamplingMode::None;
                cmd = nil;
                profile_active = false;

                throw std::runtime_error("Failed to create stage-boundary counter compute encoder");
            }
        }
    }

    py::dict resolve_counter_deltas() {
        py::dict result;

        if (active_counter_buffer == nil || active_counter_set == nil) return result;

        NSData *resolved = [active_counter_buffer resolveCounterRange:NSMakeRange(0,2)];
        if (resolved == nil) {
            result["_error"] = "resolveCounterRange returned nil";
            return result;
        }

        NSArray<id<MTLCounter>> *counters = active_counter_set.counters;
        const NSUInteger counter_count = counters.count;
        const NSUInteger expected_sample_bytes = counter_count * sizeof(uint64_t);
        if (resolved.length != expected_sample_bytes * 2) {
            result["_error"] = "Unknown counter result layout";
            result["_resolved_bytes"] = static_cast<uint64_t>(resolved.length);
            result["_expected_bytes"] = static_cast<uint64_t>(expected_sample_bytes * 2);
            return result;
        }

        const uint8_t *bytes = static_cast<const uint8_t*>(resolved.bytes);
        const uint8_t *before_bytes = bytes;
        const uint8_t *after_bytes = bytes + expected_sample_bytes;
        for (NSUInteger i = 0; i < counter_count; i++) {
            uint64_t before = 0;
            uint64_t after = 0;
            std::memcpy(&before, before_bytes + i*sizeof(uint64_t), sizeof(uint64_t));
            std::memcpy(&after, after_bytes + i*sizeof(uint64_t), sizeof(uint64_t));
            std::string name([counters[i].name UTF8String]);
            if (before == MTLCounterErrorValue || after == MTLCounterErrorValue || after < before) {
                result[name.c_str()] = py::none();
            } else {
                result[name.c_str()] = after - before;
            }
        }

        return result;
    }

    void fill_scalar(const std::string& kernel_name,
                  const std::vector<uintptr_t>& buffer_ids,
                  const std::vector<NSUInteger>& offsets,
                  float scalar_val,
                  size_t grid_size) {
        bool owns_enc = false;
        id<MTLCommandBuffer> current_cmd = cmd;
        id<MTLComputeCommandEncoder> current_enc = enc;
        
        if (current_enc == nil) {
            current_cmd = [queue commandBuffer];
            current_enc = [current_cmd computeCommandEncoder];
            owns_enc = true;
        }

        id<MTLComputePipelineState> pso = pipelines[kernel_name];
        [current_enc setComputePipelineState:pso];

        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_ids[0];
        [current_enc setBuffer:buf offset:offsets[0] atIndex:0];
        [current_enc setBytes:&scalar_val length:sizeof(float) atIndex:1];

        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        NSUInteger tgSize = std::min((NSUInteger)grid_size, pso.maxTotalThreadsPerThreadgroup);
        [current_enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

        if (owns_enc) {
            [current_enc endEncoding];
            [current_cmd commit];
            [current_cmd waitUntilCompleted];
        }
    }

    void fill_scalar2(const std::string& kernel_name,
                  const std::vector<uintptr_t>& buffer_ids,
                  const std::vector<NSUInteger>& offsets,
                  uint32_t value_bits, 
                  uint32_t element_size,
                  size_t grid_size) {
        bool owns_enc = false;
        id<MTLCommandBuffer> current_cmd = cmd;
        id<MTLComputeCommandEncoder> current_enc = enc;
        
        if (current_enc == nil) {
            current_cmd = [queue commandBuffer];
            current_enc = [current_cmd computeCommandEncoder];
            owns_enc = true;
        }

        id<MTLComputePipelineState> pso = pipelines[kernel_name];
        [current_enc setComputePipelineState:pso];

        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_ids[0];
        [current_enc setBuffer:buf offset:offsets[0] atIndex:0];
        [current_enc setBytes:&value_bits length:sizeof(value_bits) atIndex:1];
        [current_enc setBytes:&element_size length:sizeof(element_size) atIndex:2];

        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        NSUInteger tgSize = std::min((NSUInteger)grid_size, pso.maxTotalThreadsPerThreadgroup);
        [current_enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

        if (owns_enc) {
            [current_enc endEncoding];
            [current_cmd commit];
            [current_cmd waitUntilCompleted];
        }
    }

    void dispatch_strided(const std::string& kernel_name,
                          uintptr_t grad_out_id, NSUInteger grad_out_offset,
                          uintptr_t grad_x_id, NSUInteger grad_x_offset,
                          const std::vector<uint32_t>& shape,
                          const std::vector<uint32_t>& strides,
                          size_t grid_size) {
        id<MTLComputePipelineState> pso = pipelines[kernel_name];
        [enc setComputePipelineState:pso];
        id<MTLBuffer> buf_out = (__bridge id<MTLBuffer>)(void*)grad_out_id;
        id<MTLBuffer> buf_x = (__bridge id<MTLBuffer>)(void*)grad_x_id;
        
        [enc setBuffer:buf_out offset:grad_out_offset atIndex:0];
        [enc setBuffer:buf_x offset:grad_x_offset atIndex:1];
        
        [enc setBytes:shape.data() length:shape.size() * sizeof(uint32_t) atIndex:2];
        [enc setBytes:strides.data() length:strides.size() * sizeof(uint32_t) atIndex:3];
        
        uint32_t ndim = shape.size();
        [enc setBytes:&ndim length:sizeof(uint32_t) atIndex:4];
        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        NSUInteger tgSize = std::min((NSUInteger)grid_size, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    }

    void dispatch_contiguous(const std::string& kernel_name,
                             uintptr_t x_id, NSUInteger x_offset,
                             uintptr_t out_id, NSUInteger out_offset,
                             const std::vector<uint32_t>& shape,
                             const std::vector<uint32_t>& strides,
                             size_t grid_size) {
        id<MTLComputePipelineState> pso = pipelines[kernel_name];
        [enc setComputePipelineState:pso];
        
        [enc setBuffer:(__bridge id<MTLBuffer>)(void*)x_id offset:x_offset atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(void*)out_id offset:out_offset atIndex:1];
        
        [enc setBytes:shape.data() length:shape.size() * sizeof(uint32_t) atIndex:2];
        [enc setBytes:strides.data() length:strides.size() * sizeof(uint32_t) atIndex:3];
        
        uint32_t ndim = shape.size();
        [enc setBytes:&ndim length:sizeof(uint32_t) atIndex:4];
        
        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        NSUInteger tgSize = std::min((NSUInteger)grid_size, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    }

    void dispatch_with_args_3d(const std::string& kernel_name,
                           const std::vector<uintptr_t>& buffer_ids,
                           const std::vector<NSUInteger>& offsets,
                           const std::vector<uint32_t>& args,
                           size_t grid_size_x,
                           size_t grid_size_y,
                           size_t grid_size_z,
                           size_t tg_size) {
        id<MTLComputePipelineState> pso = pipelines[kernel_name];
        [enc setComputePipelineState:pso];

        for (size_t i = 0; i < buffer_ids.size(); i++) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_ids[i];
            [enc setBuffer:buf offset:offsets[i] atIndex:i];
        }

        NSUInteger base_idx = buffer_ids.size();
        for (size_t i = 0; i < args.size(); i++) {
            [enc setBytes:&args[i] length:sizeof(uint32_t) atIndex:base_idx + i];
        }

        MTLSize grid = MTLSizeMake(grid_size_x, grid_size_y, grid_size_z);
        MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
    }

    static uint32_t float_bits(float value) {
        uint32_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    static constexpr uint32_t GEMM_DTYPE_FP32 = 0;
    static constexpr uint32_t GEMM_DTYPE_FP16 = 1;
    static NSUInteger dtype_size(NSUInteger data_type) {
        switch(data_type) {
            case GEMM_DTYPE_FP32:
                return sizeof(float);
            case GEMM_DTYPE_FP16:
                return sizeof(uint16_t);
            default:
                throw std::runtime_error("unsupported GEMM dtype");
        }
    }
    static MPSDataType mps_data_type(NSUInteger data_type) {
        switch(data_type) {
            case GEMM_DTYPE_FP32:
                return MPSDataTypeFloat32;
            case GEMM_DTYPE_FP16:
                return MPSDataTypeFloat16;
            default:
                throw std::runtime_error("unsupported GEMM dtype");
        }
    }

    MPSMatrixDescriptor* get_matrix_descriptor(NSUInteger rows, NSUInteger columns, NSUInteger matrices, NSUInteger matrix_bytes, NSUInteger data_type,
                                               NSUInteger row_bytes_override=0) {
        const NSUInteger element_size = dtype_size(data_type);
        const NSUInteger row_bytes = row_bytes_override != 0 ? row_bytes_override : columns * element_size;
        const MatrixDescriptorKey key = std::make_tuple(rows, columns, matrices, row_bytes, matrix_bytes, data_type);
        auto found = matrix_descriptor_cache.find(key);
        if (found != matrix_descriptor_cache.end()) {
            return found->second;
        }
        const MPSDataType mps_type = mps_data_type(data_type);
        MPSMatrixDescriptor* descriptor = nil;
        if (matrices == 1) {
            descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows 
                columns:columns rowBytes:row_bytes dataType:mps_type];
        } else {
            descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows 
                columns:columns matrices:matrices rowBytes:row_bytes matrixBytes:matrix_bytes dataType:mps_type];
        }
        matrix_descriptor_cache.emplace(key, descriptor);
        return descriptor;
    }

    MPSMatrix* get_matrix_view(uintptr_t buffer_id, NSUInteger offset, const MatrixDescriptorKey& descriptor_key) {
        const MatrixViewKey key = std::make_tuple(buffer_id, offset, descriptor_key);
        auto found = matrix_view_cache.find(key);
        if (found != matrix_view_cache.end()) {
            return found->second;
        }
        auto descriptor_found = matrix_descriptor_cache.find(descriptor_key);
        if (descriptor_found == matrix_descriptor_cache.end()) {
            throw std::runtime_error("MPS matrix descriptor is not cached");
        }
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)buffer_id;
        MPSMatrix* matrix = [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:descriptor_found->second];
        matrix_view_cache.emplace(key, matrix);
        return matrix;
    }

    MPSMatrixMultiplication* get_matrix_multiplication(NSUInteger batch_count, NSUInteger M, NSUInteger N, NSUInteger K,
        bool transpose_a, bool transpose_b, float alpha, float beta, NSUInteger data_type) {
        auto key = std::make_tuple(batch_count, M, N, K, transpose_a, transpose_b, float_bits(alpha), float_bits(beta), data_type);
        auto found = matrix_multiplication_cache.find(key);
        if (found != matrix_multiplication_cache.end()) {
            return found->second;
        }
        MPSMatrixMultiplication* operation = [[MPSMatrixMultiplication alloc] initWithDevice:device 
            transposeLeft : transpose_a ? YES : NO 
            transposeRight : transpose_b ? YES : NO 
            resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:beta];
        matrix_multiplication_cache.emplace(key, operation);
        return operation;
    }

    void dispatch_gemm_mps(const std::vector<uintptr_t>& buffer_ids,
                           const std::vector<NSUInteger>& offsets,
                           const std::vector<NSUInteger>& args,
                           float alpha, float beta) {
        // C[M, N] = alpha * op(A)[M, K] * op(B)[K,N] + beta * C[M, N]
        const NSUInteger batch_count = args[0];
        const NSUInteger M = args[1];
        const NSUInteger N = args[2];
        const NSUInteger K = args[3];
        const bool transpose_a = args[4] != 0;
        const bool transpose_b = args[5] != 0;
        const bool a_is_batched = args[6] != 0;
        const bool b_is_batched = args[7] != 0;
        const NSUInteger data_type = args.size() >= 9 ? args[8] : GEMM_DTYPE_FP32;

        const NSUInteger element_size = dtype_size(data_type);

        const NSUInteger a_rows = transpose_a ? K : M;
        const NSUInteger a_cols = transpose_a ? M : K;
        const NSUInteger b_rows = transpose_b ? N : K;
        const NSUInteger b_cols = transpose_b ? K : N;
        const NSUInteger c_rows = M;
        const NSUInteger c_cols = N;

        const NSUInteger a_row_bytes_override = args.size() >= 10 ? args[9] : 0;

        const NSUInteger a_matrix_bytes = a_is_batched ? a_rows * a_cols * element_size : 0;
        const NSUInteger b_matrix_bytes = b_is_batched ? b_rows * b_cols * element_size : 0;
        const NSUInteger c_matrix_bytes = c_rows * c_cols * element_size;

        const NSUInteger a_row_bytes = a_row_bytes_override != 0 ? a_row_bytes_override : a_cols * element_size;
        const NSUInteger b_row_bytes = b_cols * element_size;
        const NSUInteger c_row_bytes = c_cols * element_size;
        const MatrixDescriptorKey a_key = std::make_tuple(a_rows, a_cols, batch_count, a_row_bytes, a_matrix_bytes, data_type);
        const MatrixDescriptorKey b_key = std::make_tuple(b_rows, b_cols, batch_count, b_row_bytes, b_matrix_bytes, data_type);
        const MatrixDescriptorKey c_key = std::make_tuple(c_rows, c_cols, batch_count, c_row_bytes, c_matrix_bytes, data_type);
        get_matrix_descriptor(a_rows, a_cols, batch_count, a_matrix_bytes, data_type, a_row_bytes);
        get_matrix_descriptor(b_rows, b_cols, batch_count, b_matrix_bytes, data_type);
        get_matrix_descriptor(c_rows, c_cols, batch_count, c_matrix_bytes, data_type);
        MPSMatrix* A = get_matrix_view(buffer_ids[0], offsets[0], a_key);
        MPSMatrix* B = get_matrix_view(buffer_ids[1], offsets[1], b_key);
        MPSMatrix* C = get_matrix_view(buffer_ids[2], offsets[2], c_key);
        MPSMatrixMultiplication* gemm = get_matrix_multiplication(batch_count, M, N, K, transpose_a, transpose_b, alpha, beta, data_type);

        [enc endEncoding];
        [gemm encodeToCommandBuffer:cmd leftMatrix:A rightMatrix:B resultMatrix:C];
        enc = [cmd computeCommandEncoder];
    }

    template <typename T>
    py::array_t<T> numpy_view_typed(uintptr_t buffer_id, NSUInteger offset, py::tuple shape) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_id;
        void* src_ptr = (char*)[buf contents] + offset;
        std::vector<size_t> shape_vec;
        for (auto dim: shape) {
            shape_vec.push_back(dim.cast<size_t>());
        }
        return py::array_t<T>(shape_vec, (T*)src_ptr, py::capsule(src_ptr, [](void*){} ));
    }

    void copy_from_host(uintptr_t buffer_id, NSUInteger offset, py::array_t<float> host_array) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_id;
        void* dst_ptr = (char*)[buf contents] + offset;

        py::buffer_info info = host_array.request();
        void* src_ptr = info.ptr;
        size_t bytes = info.size * info.itemsize;
        memcpy(dst_ptr, src_ptr, bytes);
    }

    py::array_t<float> copy_to_host(uintptr_t buffer_id, NSUInteger offset, py::tuple shape) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer_id;
        float* src_ptr = (float*)((char*)[buf contents] + offset);

        size_t num_elements = 1;
        std::vector<size_t> shape_vec;
        for (auto dim: shape) {
            size_t d = dim.cast<size_t>();
            num_elements *= d;
            shape_vec.push_back(d);
        }

        py::array_t<float> result(shape_vec);
        py::buffer_info info = result.request();
        memcpy(info.ptr, src_ptr, num_elements * sizeof(float));

        return result;
    }

    void synchronize() {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
};

PYBIND11_MODULE(metal_ext, m) {
    py::class_<MetalContext>(m, "MetalContext")
        .def(py::init<>())
        .def("max_buffer_length", &MetalContext::max_buffer_length)
        .def("alloc", &MetalContext::alloc)
        .def("free", &MetalContext::free)
        .def("begin_pass", &MetalContext::begin_pass) 
        .def("end_pass", &MetalContext::end_pass) 
        .def("begin_pass_profiled", &MetalContext::begin_pass_profiled)
        .def("end_pass_profiled", &MetalContext::end_pass_profiled)
        .def("counter_capacities", &MetalContext::counter_capacities)
        .def("begin_pass_counter_profiled", &MetalContext::begin_pass_counter_profiled)
        .def("dispatch_strided", &MetalContext::dispatch_strided)
        .def("dispatch_contiguous", &MetalContext::dispatch_contiguous)
        .def("dispatch_with_args_3d", &MetalContext::dispatch_with_args_3d)
        .def("dispatch_gemm_mps", &MetalContext::dispatch_gemm_mps)
        .def("fill_scalar", &MetalContext::fill_scalar)
        .def("fill_scalar2", &MetalContext::fill_scalar2)
        .def("copy_from_host", &MetalContext::copy_from_host)
        .def("copy_to_host", &MetalContext::copy_to_host)
        .def("numpy_view_float32", &MetalContext::numpy_view_typed<float>)
        .def("numpy_view_float16", &MetalContext::numpy_view_typed<__fp16>)
        .def("numpy_view_int32", &MetalContext::numpy_view_typed<int32_t>)
        .def("numpy_view_uint8", &MetalContext::numpy_view_typed<uint8_t>)
        .def("numpy_view_uint32", &MetalContext::numpy_view_typed<uint32_t>)
        .def("synchronize", &MetalContext::synchronize);
}