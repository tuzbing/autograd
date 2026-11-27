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

__device__ __forceinline__ void atomic_add_bf16(uint16_t* addr, float val) {
    uintptr_t byte_addr = reinterpret_cast<uintptr_t>(addr);
    uint32_t* word_addr = reinterpret_cast<uint32_t*>(byte_addr & ~uintptr_t(3));
    bool is_high = (byte_addr & 2u) != 0u;
    uint32_t old_word = *word_addr;
    uint32_t assumed;
    do {
        assumed = old_word;
        uint16_t old_bf16 = is_high ? uint16_t(assumed >> 16) : uint16_t(assumed & 0xffffu);
        uint16_t new_bf16 = float_to_bf16_rne(bf16_to_float(old_bf16) + val);
        uint32_t new_word = is_high
            ? ((assumed & 0x0000ffffu) | (uint32_t(new_bf16) << 16))
            : ((assumed & 0xffff0000u) | uint32_t(new_bf16));
        old_word = atomicCAS(word_addr, assumed, new_word);
    } while (assumed != old_word);
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
    uint32_t rows, uint32_t C, uint32_t vec4
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec4) {
        uint32_t c4 = C >> 2;
        if (col >= c4) return;
        for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
            uint32_t token = static_cast<uint32_t>(__ldg(&indices[row]));
            reinterpret_cast<float4*>(out)[size_t(row) * c4 + col] =
                reinterpret_cast<const float4*>(weight)[size_t(token) * c4 + col];
        }
    } else {
        if (col >= C) return;
        for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
            uint32_t token = static_cast<uint32_t>(__ldg(&indices[row]));
            out[size_t(row) * C + col] = weight[size_t(token) * C + col];
        }
    }
}

__global__ void gather_backward_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ grad_out,
    void* grad_weight,
    uint32_t rows, uint32_t C, uint32_t out_bf16
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) return;
    if (out_bf16) {
        uint16_t* gw = static_cast<uint16_t*>(grad_weight);
        for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
            uint32_t token = static_cast<uint32_t>(__ldg(&indices[row]));
            atomic_add_bf16(gw + size_t(token) * C + col, grad_out[size_t(row) * C + col]);
        }
    } else {
        float* gw = static_cast<float*>(grad_weight);
        for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
            uint32_t token = static_cast<uint32_t>(__ldg(&indices[row]));
            atomicAdd(gw + size_t(token) * C + col, grad_out[size_t(row) * C + col]);
        }
    }
}

template <bool kTransA, bool kTransB, bool kFusedLinearAdd>
__device__ __forceinline__ void cuda_sgemm_64x64x16_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* res,
    const float* __restrict__ bias,
    void* C_out,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t accumulate, uint32_t out_bf16
) {
    constexpr uint32_t BM = 64u;
    constexpr uint32_t BN = 64u;
    constexpr uint32_t BK = 16u;
    constexpr int TM = 4;
    constexpr int TN = 4;

    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4;
    const uint32_t block_n = blockIdx.x * BN;
    const uint32_t block_m = blockIdx.y * BM;
    if (block_m >= M || block_n >= N) return;

    __shared__ float As[BK][BM + 2];
    __shared__ float Bs[BK][BN + 2];

    float acc[TM][TN] = {0.0f};

    for (uint32_t k0 = 0; k0 < K; k0 += BK) {
        if constexpr (!kTransA) {
            uint32_t k_sub = tid & 15u;
            uint32_t g_k = k0 + k_sub;
            #pragma unroll
            for (uint32_t m = (tid >> 4); m < BM; m += 16u) {
                uint32_t g_m = block_m + m;
                As[k_sub][m] = (g_m < M && g_k < K) ? A[size_t(g_m) * K + g_k] : 0.0f;
            }
        } else {
            uint32_t m = tid & 63u;
            uint32_t g_m = block_m + m;
            #pragma unroll
            for (uint32_t k_sub = (tid >> 6); k_sub < BK; k_sub += 4u) {
                uint32_t g_k = k0 + k_sub;
                As[k_sub][m] = (g_k < K && g_m < M) ? A[size_t(g_k) * M + g_m] : 0.0f;
            }
        }

        if constexpr (!kTransB) {
            uint32_t n = tid & 63u;
            uint32_t g_n = block_n + n;
            #pragma unroll
            for (uint32_t k_sub = (tid >> 6); k_sub < BK; k_sub += 4u) {
                uint32_t g_k = k0 + k_sub;
                Bs[k_sub][n] = (g_k < K && g_n < N) ? B[size_t(g_k) * N + g_n] : 0.0f;
            }
        } else {
            uint32_t k_sub = tid & 15u;
            uint32_t g_k = k0 + k_sub;
            #pragma unroll
            for (uint32_t n = (tid >> 4); n < BN; n += 16u) {
                uint32_t g_n = block_n + n;
                Bs[k_sub][n] = (g_n < N && g_k < K) ? B[size_t(g_n) * K + g_k] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < int(BK); ++kk) {
            float a_reg[TM];
            float b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[kk][i * 16 + ty];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[kk][j * 16 + tx];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    float bias_reg[TN];
    if constexpr (kFusedLinearAdd) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint32_t col = block_n + uint32_t(j * 16) + tx;
            bias_reg[j] = (bias && col < N) ? __ldg(&bias[col]) : 0.0f;
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        uint32_t row = block_m + uint32_t(i * 16) + ty;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint32_t col = block_n + uint32_t(j * 16) + tx;
            if (col >= N) continue;
            size_t idx = size_t(row) * N + col;
            float val = acc[i][j];
            if constexpr (kFusedLinearAdd) {
                static_cast<float*>(C_out)[idx] = val + res[idx] + bias_reg[j];
            } else if (out_bf16) {
                uint16_t* dst = static_cast<uint16_t*>(C_out);
                if (accumulate) val += bf16_to_float(dst[idx]);
                dst[idx] = float_to_bf16_rne(val);
            } else {
                float* dst = static_cast<float*>(C_out);
                dst[idx] = accumulate ? (dst[idx] + val) : val;
            }
        }
    }
}

__global__ void cuda_sgemm_nt_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_64x64x16_impl<false, true, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__global__ void fused_linear_add_forward_kernel(
    const float* x, const float* __restrict__ act, const float* __restrict__ w,
    const float* __restrict__ b, float* out, uint32_t M, uint32_t N, uint32_t K
) {
    cuda_sgemm_64x64x16_impl<false, false, true>(act, w, x, b, out, M, N, K, 0u, 0u);
}

__global__ void cuda_sgemm_tn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, void* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate, uint32_t out_bf16
) {
    cuda_sgemm_64x64x16_impl<true, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, out_bf16);
}

__global__ void linear_add_dx_backward_kernel(
    const float* __restrict__ grad_out,
    float* grad_x,
    uint32_t n, uint32_t contiguous, uint32_t vec4, uint32_t accumulate,
    uint32_t d1, uint32_t d2, uint32_t d3,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3
) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (contiguous) {
        if (vec4) {
            uint32_t n4 = n >> 2;
            if (id >= n4) return;
            float4 g = reinterpret_cast<const float4*>(grad_out)[id];
            float4* dst4 = reinterpret_cast<float4*>(grad_x);
            if (accumulate) {
                float4 prev = dst4[id];
                dst4[id] = make_float4(prev.x + g.x, prev.y + g.y, prev.z + g.z, prev.w + g.w);
            } else {
                dst4[id] = g;
            }
        } else {
            if (id >= n) return;
            grad_x[id] = accumulate ? (grad_x[id] + grad_out[id]) : grad_out[id];
        }
        return;
    }
    if (id >= n) return;
    uint32_t rem = id;
    uint32_t i3 = rem % d3; rem /= d3;
    uint32_t i2 = rem % d2; rem /= d2;
    uint32_t i1 = rem % d1;
    uint32_t i0 = rem / d1;
    size_t dst_idx = size_t(i0) * s0 + size_t(i1) * s1 + size_t(i2) * s2 + size_t(i3) * s3;
    grad_x[dst_idx] = accumulate ? (grad_x[dst_idx] + grad_out[id]) : grad_out[id];
}

__global__ void bias_grad_reduce_kernel(
    const float* __restrict__ grad_out,
    void* __restrict__ grad_bias,
    uint32_t rows, uint32_t C, uint32_t accumulate, uint32_t out_bf16
) {
    constexpr uint32_t kCols = 8u;
    constexpr uint32_t kRowStride = 32u;

    uint32_t tid = threadIdx.x;
    uint32_t tx = tid & 7u;
    uint32_t ty = tid >> 3;
    uint32_t col = blockIdx.x * kCols + tx;

    float acc = 0.0f;
    if (col < C) {
        for (uint32_t row = ty; row < rows; row += kRowStride) {
            acc += grad_out[size_t(row) * C + col];
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);

    __shared__ float smem[8][kCols];
    uint32_t warp_id = tid >> 5;
    uint32_t lane = tid & 31u;
    if (lane < kCols) smem[warp_id][lane] = acc;
    __syncthreads();

    uint32_t out_col = blockIdx.x * kCols + tid;
    if (tid < kCols && out_col < C) {
        float total = 0.0f;
        #pragma unroll
        for (uint32_t w = 0; w < 8u; ++w) total += smem[w][tid];
        if (out_bf16) {
            uint16_t* dst = static_cast<uint16_t*>(grad_bias);
            if (accumulate) total += bf16_to_float(dst[out_col]);
            dst[out_col] = float_to_bf16_rne(total);
        } else {
            float* dst = static_cast<float*>(grad_bias);
            dst[out_col] = accumulate ? (dst[out_col] + total) : total;
        }
    }
}