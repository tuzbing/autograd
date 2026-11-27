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

__global__ void fused_rms_fa_qkv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wq, const float* __restrict__ bq,
    const float* __restrict__ wk, const float* __restrict__ bk,
    const float* __restrict__ wv, const float* __restrict__ bv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    uint32_t B, uint32_t T, uint32_t C, uint32_t H,
    uint32_t eps_bits, uint32_t q_bthd
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t block_n = blockIdx.x * 64u;
    const uint32_t block_m = blockIdx.y * 64u;
    const uint32_t proj_id = blockIdx.z;
    const uint32_t N = B * T;
    if (block_m >= N || block_n >= C || proj_id >= 3u || H == 0u || (C % H) != 0u) return;

    const uint32_t D = C / H;
    const float eps = __uint_as_float(eps_bits);
    const float* W = (proj_id == 0u) ? wq : ((proj_id == 1u) ? wk : wv);
    const float* bias = (proj_id == 0u) ? bq : ((proj_id == 1u) ? bk : bv);
    float* dst = (proj_id == 0u) ? q_out : ((proj_id == 1u) ? k_out : v_out);
    const bool write_bhtd = (proj_id != 0u) || (q_bthd == 0u);

    __shared__ float As[16][66];
    __shared__ float Bs[16][66];
    __shared__ float inv_rms_s[64];

    for (uint32_t r = warp_id; r < 64u; r += 8u) {
        uint32_t g_m = block_m + r;
        float sum_sq = 0.0f;
        if (g_m < N) {
            const float* x_row = x + size_t(g_m) * C;
            for (uint32_t c = lane; c < C; c += 32u) {
                float v = x_row[c];
                sum_sq = fmaf(v, v, sum_sq);
            }
            sum_sq = warp_allreduce_sum(sum_sq);
        }
        if (lane == 0u) {
            inv_rms_s[r] = (g_m < N) ? rsqrtf(sum_sq / float(C) + eps) : 0.0f;
        }
    }
    __syncthreads();

    float acc[4][4] = {0.0f};
    for (uint32_t k0 = 0u; k0 < C; k0 += 16u) {
        uint32_t k_sub_a = tid & 15u;
        uint32_t g_k_a = k0 + k_sub_a;
        #pragma unroll
        for (uint32_t m = (tid >> 4); m < 64u; m += 16u) {
            uint32_t g_m = block_m + m;
            As[k_sub_a][m] = (g_m < N && g_k_a < C) ? (x[size_t(g_m) * C + g_k_a] * inv_rms_s[m]) : 0.0f;
        }

        uint32_t n = tid & 63u;
        uint32_t g_n = block_n + n;
        #pragma unroll
        for (uint32_t k_sub_b = (tid >> 6); k_sub_b < 16u; k_sub_b += 4u) {
            uint32_t g_k_b = k0 + k_sub_b;
            Bs[k_sub_b][n] = (g_k_b < C && g_n < C) ? W[size_t(g_k_b) * C + g_n] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a_reg[4], b_reg[4];
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

    float bias_reg[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t col = block_n + uint32_t(j * 16) + tx;
        bias_reg[j] = (bias && col < C) ? __ldg(&bias[col]) : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t row = block_m + uint32_t(i * 16) + ty;
        if (row >= N) continue;
        uint32_t b_idx = row / T;
        uint32_t t_idx = row - b_idx * T;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t col = block_n + uint32_t(j * 16) + tx;
            if (col >= C) continue;
            float val = acc[i][j] + bias_reg[j];
            if (write_bhtd) {
                uint32_t h_idx = col / D;
                uint32_t d_idx = col - h_idx * D;
                dst[((size_t(b_idx) * H + h_idx) * T + t_idx) * D + d_idx] = val;
            } else {
                dst[size_t(row) * C + col] = val;
            }
        }
    }
}

__global__ void rope_qk_inplace_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    uint32_t B, uint32_t H, uint32_t T, uint32_t D, uint32_t mode
) {
    uint32_t half_D = D >> 1;
    uint32_t total = B * H * T * half_D;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (half_D == 0u || idx >= total) return;

    uint32_t j = idx % half_D;
    uint32_t slice = idx / half_D;
    uint32_t b, h, t;
    if (mode == 1u) {
        uint32_t bt = slice / H;
        h = slice - bt * H;
        b = bt / T;
        t = bt - b * T;
    } else {
        uint32_t bh = slice / T;
        t = slice - bh * T;
        b = bh / H;
        h = bh - b * H;
    }

    size_t bhtd_base = ((size_t(b) * H + h) * T + t) * D;
    size_t bthd_base = ((size_t(b) * T + t) * H + h) * D;
    size_t q_base = (mode == 0u) ? bhtd_base : bthd_base;
    size_t k_base = (mode == 1u) ? bthd_base : bhtd_base;

    size_t rope_idx = size_t(t) * half_D + j;
    float c = __ldg(&cos_table[rope_idx]);
    float s = (mode == 1u) ? -__ldg(&sin_table[rope_idx]) : __ldg(&sin_table[rope_idx]);

    float q0 = q[q_base + j], q1 = q[q_base + half_D + j];
    float k0 = k[k_base + j], k1 = k[k_base + half_D + j];
    q[q_base + j]          = fmaf(q0, c, q1 * s);
    q[q_base + half_D + j] = fmaf(-q0, s, q1 * c);
    k[k_base + j]          = fmaf(k0, c, k1 * s);
    k[k_base + half_D + j] = fmaf(-k0, s, k1 * c);
}

__global__ void flash_attn_fwd_kernel(
    const float* q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* out,
    float* __restrict__ lse_out,
    uint32_t B, uint32_t H, uint32_t T, uint32_t C,
    uint32_t is_causal, uint32_t write_lse, uint32_t q_bthd
) {
    if (B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) return;
    const uint32_t D = C / H;
    if (D == 0u || D > 128u) return;

    const uint32_t query_base = blockIdx.x * 32u;
    const uint32_t bh = blockIdx.y;
    if (query_base >= T || bh >= B * H) return;

    const uint32_t b = bh / H;
    const uint32_t h = bh - b * H;
    const uint32_t tid = threadIdx.x;
    const uint32_t load_row = tid >> 3;
    const uint32_t load_col = tid & 7u;
    const uint32_t s_ty = tid >> 4;
    const uint32_t s_tx = tid & 15u;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane = tid & 31u;

    constexpr uint32_t kStride = 129u;
    __shared__ float Qs[32][kStride];
    __shared__ float KVs[32][kStride];
    __shared__ float Ps[32][33];
    __shared__ float m_s[32];
    __shared__ float l_s[32];
    __shared__ float alpha_s[32];

    if (tid < 32u) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
        alpha_s[tid] = 0.0f;
    }

    const uint32_t q_load_tok = query_base + load_row;
    for (uint32_t d = load_col; d < D; d += 8u) {
        if (q_load_tok < T) {
            size_t q_off = q_bthd
                ? (((size_t(b) * T + q_load_tok) * H + h) * D + d)
                : ((size_t(bh) * T + q_load_tok) * D + d);
            Qs[load_row][d] = q[q_off];
        } else {
            Qs[load_row][d] = 0.0f;
        }
    }
    __syncthreads();

    float o_acc[16] = {0.0f};
    const float scale = rsqrtf(float(D));
    const uint32_t key_limit = is_causal ? min(T, query_base + 32u) : T;

    for (uint32_t key_base = 0u; key_base < key_limit; key_base += 32u) {
        uint32_t kv_tok = key_base + load_row;
        for (uint32_t d = load_col; d < D; d += 8u) {
            KVs[load_row][d] = (kv_tok < T) ? k[(size_t(bh) * T + kv_tok) * D + d] : 0.0f;
        }
        __syncthreads();

        float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f;
        uint32_t q0 = s_ty, q1 = s_ty + 16u;
        uint32_t k0 = s_tx, k1 = s_tx + 16u;
        for (uint32_t d = 0u; d < D; ++d) {
            float qv0 = Qs[q0][d], qv1 = Qs[q1][d];
            float kv0 = KVs[k0][d], kv1 = KVs[k1][d];
            s00 = fmaf(qv0, kv0, s00);
            s01 = fmaf(qv0, kv1, s01);
            s10 = fmaf(qv1, kv0, s10);
            s11 = fmaf(qv1, kv1, s11);
        }
        uint32_t gq0 = query_base + q0, gq1 = query_base + q1;
        uint32_t gk0 = key_base + k0,   gk1 = key_base + k1;
        Ps[q0][k0] = (gq0 < T && gk0 < T && (!is_causal || gk0 <= gq0)) ? (s00 * scale) : -INFINITY;
        Ps[q0][k1] = (gq0 < T && gk1 < T && (!is_causal || gk1 <= gq0)) ? (s01 * scale) : -INFINITY;
        Ps[q1][k0] = (gq1 < T && gk0 < T && (!is_causal || gk0 <= gq1)) ? (s10 * scale) : -INFINITY;
        Ps[q1][k1] = (gq1 < T && gk1 < T && (!is_causal || gk1 <= gq1)) ? (s11 * scale) : -INFINITY;
        __syncthreads();

        #pragma unroll
        for (uint32_t r_off = 0u; r_off < 32u; r_off += 8u) {
            uint32_t r = warp_id + r_off;
            float score = Ps[r][lane];
            float tmax = score;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, mask));
            }
            float old_max = m_s[r];
            float new_max = fmaxf(old_max, tmax);
            float alpha = isfinite(old_max) ? expf(old_max - new_max) : 0.0f;
            float p = isfinite(score) ? expf(score - new_max) : 0.0f;
            float tsum = warp_allreduce_sum(p);
            Ps[r][lane] = p;
            if (lane == 0u) {
                m_s[r] = new_max;
                l_s[r] = l_s[r] * alpha + tsum;
                alpha_s[r] = alpha;
            }
        }
        for (uint32_t d = load_col; d < D; d += 8u) {
            KVs[load_row][d] = (kv_tok < T) ? v[(size_t(bh) * T + kv_tok) * D + d] : 0.0f;
        }
        __syncthreads();

        float alpha = alpha_s[load_row];
        #pragma unroll
        for (int m = 0; m < 16; ++m) o_acc[m] *= alpha;

        #pragma unroll
        for (uint32_t kk = 0u; kk < 32u; ++kk) {
            float p = Ps[load_row][kk];
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                uint32_t d = uint32_t(m * 8) + load_col;
                if (d < D) o_acc[m] = fmaf(p, KVs[kk][d], o_acc[m]);
            }
        }
        __syncthreads();
    }

    if (q_load_tok < T) {
        float denom = l_s[load_row];
        float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        if (write_lse && load_col == 0u) {
            lse_out[size_t(bh) * T + q_load_tok] = (denom > 0.0f) ? (m_s[load_row] + logf(denom)) : 0.0f;
        }
        size_t out_base = ((size_t(b) * T + q_load_tok) * H + h) * D;
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            uint32_t d = uint32_t(m * 8) + load_col;
            if (d < D) out[out_base + d] = o_acc[m] * inv_sum;
        }
    }
}

__global__ void flash_attn_bwd_prepare_kernel(
    const float* __restrict__ o_btc,
    const float* __restrict__ do_btc,
    float* __restrict__ d_val,
    uint32_t B, uint32_t H, uint32_t T, uint32_t C
) {
    if (B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) return;
    const uint32_t D = C / H;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * 4u + warp_id;
    if (idx >= B * H * T) return;

    const uint32_t bh = idx / T;
    const uint32_t t = idx - bh * T;
    const uint32_t b = bh / H;
    const uint32_t h = bh - b * H;
    const size_t base = ((size_t(b) * T + t) * H + h) * D;

    float sum = 0.0f;
    for (uint32_t d = lane; d < D; d += 32u) {
        sum = fmaf(o_btc[base + d], do_btc[base + d], sum);
    }
    sum = warp_allreduce_sum(sum);
    if (lane == 0u) d_val[idx] = sum;
}

__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const float* __restrict__ lse,
    const float* __restrict__ d_val,
    uint32_t B, uint32_t H, uint32_t T, uint32_t C, uint32_t is_causal
) {
    if (B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) return;
    const uint32_t D = C / H;
    if (D == 0u || D > 128u) return;

    const uint32_t block_tok = blockIdx.x * 16u;
    const uint32_t bh = blockIdx.y;
    const bool mode_dkv = (blockIdx.z != 0u);
    if (block_tok >= T || bh >= B * H) return;

    const uint32_t b = bh / H;
    const uint32_t h = bh - b * H;
    const uint32_t tid = threadIdx.x;
    const uint32_t row16 = tid >> 4;
    const uint32_t col16 = tid & 15u;
    const float scale = rsqrtf(float(D));

    constexpr uint32_t kStride = 129u;
    __shared__ float Qs[16][kStride];
    __shared__ float dOs[16][kStride];
    __shared__ float Ks[16][kStride];
    __shared__ float Vs[16][kStride];
    __shared__ float Ps[16][17];
    __shared__ float dSs[16][17];
    __shared__ float lse_s[16];
    __shared__ float dval_s[16];

    float acc0[8] = {0.0f};
    float acc1[8] = {0.0f};

    if (!mode_dkv) {
        uint32_t q_tok_init = block_tok + row16;
        for (uint32_t d = col16; d < D; d += 16u) {
            if (q_tok_init < T) {
                Qs[row16][d]  = q[(size_t(bh) * T + q_tok_init) * D + d];
                dOs[row16][d] = dO[((size_t(b) * T + q_tok_init) * H + h) * D + d];
            } else {
                Qs[row16][d]  = 0.0f;
                dOs[row16][d] = 0.0f;
            }
        }
        if (tid < 16u) {
            uint32_t qt = block_tok + tid;
            lse_s[tid]  = (qt < T) ? lse[size_t(bh) * T + qt] : 0.0f;
            dval_s[tid] = (qt < T) ? d_val[size_t(bh) * T + qt] : 0.0f;
        }
        __syncthreads();

        uint32_t k_limit = is_causal ? min(T, block_tok + 16u) : T;
        for (uint32_t k_base = 0u; k_base < k_limit; k_base += 16u) {
            uint32_t k_tok_load = k_base + row16;
            for (uint32_t d = col16; d < D; d += 16u) {
                Ks[row16][d] = (k_tok_load < T) ? k[(size_t(bh) * T + k_tok_load) * D + d] : 0.0f;
                Vs[row16][d] = (k_tok_load < T) ? v[(size_t(bh) * T + k_tok_load) * D + d] : 0.0f;
            }
            __syncthreads();

            uint32_t q_tok = block_tok + row16;
            uint32_t k_tok = k_base + col16;
            float score = 0.0f, dp = 0.0f;
            for (uint32_t d = 0u; d < D; ++d) {
                score = fmaf(Qs[row16][d], Ks[col16][d], score);
                dp    = fmaf(dOs[row16][d], Vs[col16][d], dp);
            }
            bool valid = (q_tok < T) && (k_tok < T) && (!is_causal || k_tok <= q_tok);
            float p = valid ? expf(score * scale - lse_s[row16]) : 0.0f;
            dSs[row16][col16] = valid ? (scale * p * (dp - dval_s[row16])) : 0.0f;
            __syncthreads();

            #pragma unroll
            for (uint32_t kk = 0u; kk < 16u; ++kk) {
                float ds = dSs[row16][kk];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    uint32_t d = uint32_t(m * 16) + col16;
                    if (d < D) acc0[m] = fmaf(ds, Ks[kk][d], acc0[m]);
                }
            }
            __syncthreads();
        }

        if (q_tok_init < T) {
            size_t out_base = ((size_t(b) * T + q_tok_init) * H + h) * D;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                uint32_t d = uint32_t(m * 16) + col16;
                if (d < D) dQ[out_base + d] = acc0[m];
            }
        }
    } else {
        uint32_t k_tok_init = block_tok + row16;
        for (uint32_t d = col16; d < D; d += 16u) {
            Ks[row16][d] = (k_tok_init < T) ? k[(size_t(bh) * T + k_tok_init) * D + d] : 0.0f;
            Vs[row16][d] = (k_tok_init < T) ? v[(size_t(bh) * T + k_tok_init) * D + d] : 0.0f;
        }
        __syncthreads();

        uint32_t q_start = is_causal ? block_tok : 0u;
        for (uint32_t q_base = q_start; q_base < T; q_base += 16u) {
            uint32_t q_tok_load = q_base + row16;
            for (uint32_t d = col16; d < D; d += 16u) {
                if (q_tok_load < T) {
                    Qs[row16][d]  = q[(size_t(bh) * T + q_tok_load) * D + d];
                    dOs[row16][d] = dO[((size_t(b) * T + q_tok_load) * H + h) * D + d];
                } else {
                    Qs[row16][d]  = 0.0f;
                    dOs[row16][d] = 0.0f;
                }
            }
            if (tid < 16u) {
                uint32_t qt = q_base + tid;
                lse_s[tid]  = (qt < T) ? lse[size_t(bh) * T + qt] : 0.0f;
                dval_s[tid] = (qt < T) ? d_val[size_t(bh) * T + qt] : 0.0f;
            }
            __syncthreads();

            uint32_t q_tok = q_base + row16;
            uint32_t k_tok = block_tok + col16;
            float score = 0.0f, dp = 0.0f;
            for (uint32_t d = 0u; d < D; ++d) {
                score = fmaf(Qs[row16][d], Ks[col16][d], score);
                dp    = fmaf(dOs[row16][d], Vs[col16][d], dp);
            }
            bool valid = (q_tok < T) && (k_tok < T) && (!is_causal || k_tok <= q_tok);
            float p = valid ? expf(score * scale - lse_s[row16]) : 0.0f;
            Ps[col16][row16]  = p;
            dSs[col16][row16] = valid ? (scale * p * (dp - dval_s[row16])) : 0.0f;
            __syncthreads();

            #pragma unroll
            for (uint32_t qq = 0u; qq < 16u; ++qq) {
                float ds = dSs[row16][qq];
                float p  = Ps[row16][qq];
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    uint32_t d = uint32_t(m * 16) + col16;
                    if (d < D) {
                        acc0[m] = fmaf(ds, Qs[qq][d], acc0[m]);
                        acc1[m] = fmaf(p, dOs[qq][d], acc1[m]);
                    }
                }
            }
            __syncthreads();
        }

        if (k_tok_init < T) {
            size_t out_base = ((size_t(b) * T + k_tok_init) * H + h) * D;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                uint32_t d = uint32_t(m * 16) + col16;
                if (d < D) {
                    dK[out_base + d] = acc0[m];
                    dV[out_base + d] = acc1[m];
                }
            }
        }
    }
}