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

__global__ void cuda_sgemm_nn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_64x64x16_impl<false, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__device__ __forceinline__ float warp_allreduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, mask);
    }
    return val;
}

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    uint32_t N, uint32_t C, uint32_t eps_bits
) {
    uint32_t warp_id = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_id;
    if (row >= N || C == 0u) return;

    const float* x_row = x + size_t(row) * C;
    float* out_row = out + size_t(row) * C;
    const float eps = __uint_as_float(eps_bits);
    const bool vec4 = ((C & 3u) == 0u) &&
                      ((reinterpret_cast<uintptr_t>(x_row) & 15u) == 0u) &&
                      ((reinterpret_cast<uintptr_t>(out_row) & 15u) == 0u) &&
                      (!gamma || (reinterpret_cast<uintptr_t>(gamma) & 15u) == 0u);

    float sum_sq = 0.0f;
    if (vec4) {
        uint32_t c4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (uint32_t i = lane; i < c4; i += 32u) {
            float4 v = x4[i];
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        }
    } else {
        for (uint32_t c = lane; c < C; c += 32u) {
            float v = x_row[c];
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    const float inv_rms = rsqrtf(warp_allreduce_sum(sum_sq) / float(C) + eps);

    if (vec4) {
        uint32_t c4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        float4* out4 = reinterpret_cast<float4*>(out_row);
        if (gamma) {
            const float4* g4 = reinterpret_cast<const float4*>(gamma);
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                float4 g = __ldg(&g4[i]);
                out4[i] = make_float4(v.x * inv_rms * g.x, v.y * inv_rms * g.y,
                                      v.z * inv_rms * g.z, v.w * inv_rms * g.w);
            }
        } else {
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                out4[i] = make_float4(v.x * inv_rms, v.y * inv_rms,
                                      v.z * inv_rms, v.w * inv_rms);
            }
        }
    } else if (gamma) {
        for (uint32_t c = lane; c < C; c += 32u) {
            out_row[c] = x_row[c] * inv_rms * __ldg(&gamma[c]);
        }
    } else {
        for (uint32_t c = lane; c < C; c += 32u) {
            out_row[c] = x_row[c] * inv_rms;
        }
    }
}

__global__ void fused_wgu_swiglu_forward_kernel(
    const float* __restrict__ x_norm,
    const float* __restrict__ wgu,
    float* __restrict__ hidden,
    uint32_t M, uint32_t H, uint32_t K
) {
    constexpr uint32_t BM = 64u;
    constexpr uint32_t BH = 32u;
    constexpr uint32_t BK = 16u;

    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4;
    const uint32_t block_h = blockIdx.x * BH;
    const uint32_t block_m = blockIdx.y * BM;
    if (block_m >= M || block_h >= H) return;

    const size_t H2 = size_t(H) * 2u;
    __shared__ float As[BK][BM + 2];
    __shared__ float Bs[BK][66];

    float acc[4][4] = {0.0f};

    for (uint32_t k0 = 0; k0 < K; k0 += BK) {
        uint32_t k_sub_a = tid & 15u;
        uint32_t g_k_a = k0 + k_sub_a;
        #pragma unroll
        for (uint32_t m = (tid >> 4); m < BM; m += 16u) {
            uint32_t g_m = block_m + m;
            As[k_sub_a][m] = (g_m < M && g_k_a < K) ? x_norm[size_t(g_m) * K + g_k_a] : 0.0f;
        }

        uint32_t n = tid & 63u;
        uint32_t col_h = block_h + (n & 31u);
        size_t w_col = (n < 32u) ? size_t(col_h) : (size_t(H) + size_t(col_h));
        #pragma unroll
        for (uint32_t k_sub_b = (tid >> 6); k_sub_b < BK; k_sub_b += 4u) {
            uint32_t g_k_b = k0 + k_sub_b;
            Bs[k_sub_b][n] = (g_k_b < K && col_h < H) ? wgu[size_t(g_k_b) * H2 + w_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < int(BK); ++kk) {
            float a_reg[4];
            float b_reg[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a_reg[i] = As[kk][i * 16 + ty];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b_reg[j] = Bs[kk][j * 16 + tx];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t row = block_m + uint32_t(i * 16) + ty;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t col = block_h + uint32_t(j * 16) + tx;
            if (col >= H) continue;
            float g = acc[i][j];
            float u = acc[i][j + 2];
            float silu = g / (1.0f + expf(-g));
            hidden[size_t(row) * H + col] = silu * u;
        }
    }
}

__global__ void swiglu_forward_kernel(
    const float* __restrict__ gu,
    float* __restrict__ hidden,
    uint32_t N, uint32_t H
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= H) return;
    const size_t H2 = size_t(H) * 2u;
    for (uint32_t row = blockIdx.y; row < N; row += gridDim.y) {
        size_t gu_base = size_t(row) * H2;
        float g = gu[gu_base + col];
        float u = gu[gu_base + H + col];
        float sig = 1.0f / (1.0f + expf(-g));
        hidden[size_t(row) * H + col] = (g * sig) * u;
    }
}

__global__ void swiglu_backward_inplace_kernel(
    float* __restrict__ gu,
    const float* __restrict__ dhidden,
    uint32_t N, uint32_t H
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= H) return;
    const size_t H2 = size_t(H) * 2u;
    for (uint32_t row = blockIdx.y; row < N; row += gridDim.y) {
        size_t gu_base = size_t(row) * H2;
        float g = gu[gu_base + col];
        float u = gu[gu_base + H + col];
        float dh = dhidden[size_t(row) * H + col];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        float dsilu = sig * (1.0f + g * (1.0f - sig));
        gu[gu_base + col] = dh * u * dsilu;
        gu[gu_base + H + col] = dh * silu;
    }
}

__global__ void rmsnorm_backward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ d_xnorm,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ inv_rms_out,
    uint32_t N, uint32_t C, uint32_t eps_bits, uint32_t accumulate
) {
    uint32_t warp_id = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_id;
    if (row >= N || C == 0u) return;

    const float* x_row = x + size_t(row) * C;
    const float* dy_row = d_xnorm + size_t(row) * C;
    const float eps = __uint_as_float(eps_bits);
    const bool vec4 = ((C & 3u) == 0u) &&
                      ((reinterpret_cast<uintptr_t>(x_row) & 15u) == 0u) &&
                      ((reinterpret_cast<uintptr_t>(dy_row) & 15u) == 0u) &&
                      (!gamma || (reinterpret_cast<uintptr_t>(gamma) & 15u) == 0u) &&
                      (!dx || (reinterpret_cast<uintptr_t>(dx + size_t(row) * C) & 15u) == 0u);

    float sum_sq = 0.0f;
    if (vec4) {
        uint32_t c4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (uint32_t i = lane; i < c4; i += 32u) {
            float4 v = x4[i];
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        }
    } else {
        for (uint32_t c = lane; c < C; c += 32u) {
            float v = x_row[c];
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    const float inv_rms = rsqrtf(warp_allreduce_sum(sum_sq) / float(C) + eps);
    if (inv_rms_out && lane == 0u) {
        inv_rms_out[row] = inv_rms;
    }
    if (!dx) return;

    float sum_dot = 0.0f;
    if (vec4) {
        uint32_t c4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        const float4* dy4 = reinterpret_cast<const float4*>(dy_row);
        if (gamma) {
            const float4* g4 = reinterpret_cast<const float4*>(gamma);
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                float4 dy = dy4[i];
                float4 g = __ldg(&g4[i]);
                sum_dot = fmaf(dy.x * g.x, v.x * inv_rms, sum_dot);
                sum_dot = fmaf(dy.y * g.y, v.y * inv_rms, sum_dot);
                sum_dot = fmaf(dy.z * g.z, v.z * inv_rms, sum_dot);
                sum_dot = fmaf(dy.w * g.w, v.w * inv_rms, sum_dot);
            }
        } else {
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                float4 dy = dy4[i];
                sum_dot = fmaf(dy.x, v.x * inv_rms, sum_dot);
                sum_dot = fmaf(dy.y, v.y * inv_rms, sum_dot);
                sum_dot = fmaf(dy.z, v.z * inv_rms, sum_dot);
                sum_dot = fmaf(dy.w, v.w * inv_rms, sum_dot);
            }
        }
    } else if (gamma) {
        for (uint32_t c = lane; c < C; c += 32u) {
            sum_dot = fmaf(dy_row[c] * __ldg(&gamma[c]), x_row[c] * inv_rms, sum_dot);
        }
    } else {
        for (uint32_t c = lane; c < C; c += 32u) {
            sum_dot = fmaf(dy_row[c], x_row[c] * inv_rms, sum_dot);
        }
    }

    const float mean_dot = warp_allreduce_sum(sum_dot) / float(C);
    float* dx_row = dx + size_t(row) * C;

    if (vec4) {
        uint32_t c4 = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        const float4* dy4 = reinterpret_cast<const float4*>(dy_row);
        float4* dx4 = reinterpret_cast<float4*>(dx_row);
        if (gamma) {
            const float4* g4 = reinterpret_cast<const float4*>(gamma);
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                float4 dy = dy4[i];
                float4 g = __ldg(&g4[i]);
                float4 res = make_float4(
                    inv_rms * (dy.x * g.x - (v.x * inv_rms) * mean_dot),
                    inv_rms * (dy.y * g.y - (v.y * inv_rms) * mean_dot),
                    inv_rms * (dy.z * g.z - (v.z * inv_rms) * mean_dot),
                    inv_rms * (dy.w * g.w - (v.w * inv_rms) * mean_dot));
                if (accumulate) {
                    float4 prev = dx4[i];
                    res = make_float4(prev.x + res.x, prev.y + res.y, prev.z + res.z, prev.w + res.w);
                }
                dx4[i] = res;
            }
        } else {
            for (uint32_t i = lane; i < c4; i += 32u) {
                float4 v = x4[i];
                float4 dy = dy4[i];
                float4 res = make_float4(
                    inv_rms * (dy.x - (v.x * inv_rms) * mean_dot),
                    inv_rms * (dy.y - (v.y * inv_rms) * mean_dot),
                    inv_rms * (dy.z - (v.z * inv_rms) * mean_dot),
                    inv_rms * (dy.w - (v.w * inv_rms) * mean_dot));
                if (accumulate) {
                    float4 prev = dx4[i];
                    res = make_float4(prev.x + res.x, prev.y + res.y, prev.z + res.z, prev.w + res.w);
                }
                dx4[i] = res;
            }
        }
    } else {
        for (uint32_t c = lane; c < C; c += 32u) {
            float dy_val = gamma ? (dy_row[c] * __ldg(&gamma[c])) : dy_row[c];
            float x_hat = x_row[c] * inv_rms;
            float val = inv_rms * (dy_val - x_hat * mean_dot);
            dx_row[c] = accumulate ? (dx_row[c] + val) : val;
        }
    }
}

__global__ void rmsnorm_dgamma_kernel(
    const float* __restrict__ x,
    const float* __restrict__ d_xnorm,
    const float* __restrict__ inv_rms,
    void* __restrict__ dgamma,
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
            size_t idx = size_t(row) * C + col;
            acc = fmaf(d_xnorm[idx], x[idx] * __ldg(&inv_rms[row]), acc);
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
            uint16_t* dst = static_cast<uint16_t*>(dgamma);
            if (accumulate) total += bf16_to_float(dst[out_col]);
            dst[out_col] = float_to_bf16_rne(total);
        } else {
            float* dst = static_cast<float*>(dgamma);
            dst[out_col] = accumulate ? (dst[out_col] + total) : total;
        }
    }
}