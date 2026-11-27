#pragma once
#include <cstdint>

__device__ __forceinline__ float bf16_to_float(uint16_t b) {
	return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_rne(float v) {
	uint32_t bits = __float_as_uint(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return uint16_t((bits >> 16) | ((bits & 0x007fffffu) ? 0x0040u : 0u));
    }
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, mask);
    }
    return v;
}

__global__ void fill_scalar_kernel2(uint8_t* out, uint32_t count, uint32_t value_bits, uint32_t itemsize) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    if (itemsize == 4u) {
        reinterpret_cast<uint32_t*>(out)[id] = value_bits;
    } else if (itemsize == 2u) {
        reinterpret_cast<uint16_t*>(out)[id] = static_cast<uint16_t>(value_bits);
    } else if (itemsize == 1u) {
        out[id] = static_cast<uint8_t>(value_bits);
    } else {
        size_t base = size_t(id) * size_t(itemsize);
        for (uint32_t b = 0; b < itemsize; ++b) {
            out[base + b] = static_cast<uint8_t>((value_bits >> (b * 8u)) & 0xffu);
        }
    }
}

__global__ void gather_forward_kernel(
		const int32_t* __restrict__ indices,
		const float* __restrict__ weight,
		float* __restrict__ out,
		uint32_t rows, uint32_t C, uint32_t vec4) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec4) {
        uint32_t c4 = C >> 2;
        if (id >= rows * c4) return;
        uint32_t row = id / c4;
        uint32_t col4 = id - row * c4;
        uint32_t token = static_cast<uint32_t>(indices[row]);
        reinterpret_cast<float4*>(out)[id] =
            reinterpret_cast<const float4*>(weight)[size_t(token) * c4 + col4];
    } else {
        if (id >= rows * C) return;
        uint32_t row = id / C;
        uint32_t col = id - row * C;
        uint32_t token = static_cast<uint32_t>(indices[row]);
        out[id] = weight[size_t(token) * C + col];
    }
}

__global__ void gather_backward_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ grad_out,
    float* grad_weight,
    uint32_t total, uint32_t C
) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total || C == 0u) return;
    uint32_t row = id / C;
    uint32_t col = id - row * C;
    uint32_t token = static_cast<uint32_t>(indices[row]);
    atomicAdd(grad_weight + size_t(token) * C + col, grad_out[id]);
}

__global__ void gather_backward_bf16_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ grad_out,
    uint16_t* grad_weight,
    uint32_t rows, uint32_t C
) {
    uint32_t lane = threadIdx.x;
    uint32_t source_row = blockIdx.y;
    if (source_row >= rows) return;
    int32_t token = indices[source_row];

    bool appeared_before = false;
    for (uint32_t r = lane; r < source_row; r += 32u) {
        if (indices[r] == token) {
            appeared_before = true;
            break;
        }
    }
    if (__any_sync(0xffffffffu, appeared_before)) return;

    uint32_t channel = blockIdx.x * 128u + lane * 4u;
    if (channel >= C) return;
    uint32_t valid = min(4u, C - channel);

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t r = source_row; r < rows; ++r) {
        if (indices[r] != token) continue;
        size_t src_base = size_t(r) * C + channel;
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            if (i < valid) sum[i] += grad_out[src_base + i];
        }
    }

    size_t dst_base = size_t(static_cast<uint32_t>(token)) * C + channel;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        if (i < valid) {
            float old_val = bf16_to_float(grad_weight[dst_base + i]);
            grad_weight[dst_base + i] = float_to_bf16_rne(old_val + sum[i]);
        }
    }
}

__global__ void add_residual_forward_vec4(const float4* x, const float4* y, float4* out, uint32_t n4) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n4) return;
    float4 a = x[id];
    float4 b = y[id];
    out[id] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void add_residual_forward(const float* x, const float* y, float* out, uint32_t n) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    out[id] = x[id] + y[id];
}

__global__ void add_residual_backward_vec4(const float4* grad_out, float4* grad_x, float4* grad_y, uint32_t n4) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n4) return;
    float4 g = grad_out[id];
    if (grad_x) grad_x[id] = g;
    if (grad_y) grad_y[id] = g;
}

__global__ void add_residual_backward(const float* grad_out, float* grad_x, float* grad_y, uint32_t n) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    float g = grad_out[id];
    if (grad_x) grad_x[id] = g;
    if (grad_y) grad_y[id] = g;
}

__global__ void contiguous_backward_kernel(const float* grad_out, float* grad_x, uint32_t n) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n && grad_x) grad_x[id] = grad_out[id];
}

__global__ void add_pos_forward(const float* x, const float* pos, float* out, uint32_t n, uint32_t tc) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    out[id] = x[id] + pos[id % tc];
}

__global__ void pos_emb_grad(const float* grad_out, float* grad_pos, uint32_t B, uint32_t tc, uint32_t accumulate) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tc) return;
    float sum = 0.0f;
    for (uint32_t b = 0; b < B; ++b) {
        sum += grad_out[size_t(b) * tc + id];
    }
    grad_pos[id] = accumulate ? (grad_pos[id] + sum) : sum;
}

__global__ void add_bias_forward(const float* x, const float* bias, float* out, uint32_t n, uint32_t C) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    out[id] = x[id] + bias[id % C];
}

template <bool kBF16>
__device__ __forceinline__ void bias_grad_reduce_impl(
    const float* __restrict__ grad_out,
    void* __restrict__ grad_bias,
    uint32_t rows, uint32_t C, uint32_t accumulate
) {
    constexpr uint32_t kThreads = 256u;
    constexpr uint32_t kWarps = kThreads / 32u;
    constexpr uint32_t kBN = 16u;

    uint32_t col0 = blockIdx.x * kBN;
    if (col0 >= C) return;
    uint32_t valid_cols = min(kBN, C - col0);
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid & 31u;
    uint32_t warp_id = tid >> 5;

    float acc[16] = {0.0f};
    bool aligned4 = ((C & 3u) == 0u) && (((uintptr_t)grad_out & 15u) == 0u) && (valid_cols == kBN);

    for (uint32_t row = tid; row < rows; row += kThreads) {
        const float* src = grad_out + size_t(row) * C + col0;
        if (aligned4) {
            const float4* src4 = reinterpret_cast<const float4*>(src);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                float4 val = src4[v];
                acc[v * 4 + 0] += val.x;
                acc[v * 4 + 1] += val.y;
                acc[v * 4 + 2] += val.z;
                acc[v * 4 + 3] += val.w;
            }
        } else {
            for (uint32_t c = 0; c < valid_cols; ++c) {
                acc[c] += src[c];
            }
        }
    }

    __shared__ float smem[kWarps][kBN];
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        float s = warp_reduce_sum(acc[c]);
        if (lane == 0u) smem[warp_id][c] = s;
    }
    __syncthreads();

    if (tid < valid_cols) {
        float total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0; w < kWarps; ++w) {
            total += smem[w][tid];
        }
        if constexpr (kBF16) {
            uint16_t* dst = static_cast<uint16_t*>(grad_bias) + col0;
            if (accumulate) total += bf16_to_float(dst[tid]);
            dst[tid] = float_to_bf16_rne(total);
        } else {
            float* dst = static_cast<float*>(grad_bias) + col0;
            dst[tid] = accumulate ? (dst[tid] + total) : total;
        }
    }
}

__global__ void bias_grad_reduce(const float* grad_out, float* grad_bias, uint32_t rows, uint32_t C, uint32_t accumulate) {
    bias_grad_reduce_impl<false>(grad_out, grad_bias, rows, C, accumulate);
}

__global__ void bias_grad_reduce_bf16(const float* grad_out, uint16_t* grad_bias, uint32_t rows, uint32_t C, uint32_t accumulate) {
    bias_grad_reduce_impl<true>(grad_out, grad_bias, rows, C, accumulate);
}