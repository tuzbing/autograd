#pragma once
#include <cstdint>

template <uint32_t GROUP_M = 8u>
__device__ __forceinline__ void gemm_swizzle_cta(
    uint32_t bx, uint32_t by, uint32_t grid_x, uint32_t grid_y,
    uint32_t& pid_m, uint32_t& pid_n
) {
    const uint32_t num_pid_in_group = GROUP_M * grid_x;
    const uint32_t pid = by * grid_x + bx;
    const uint32_t group_id = pid / num_pid_in_group;
    const uint32_t first_pid_m = group_id * GROUP_M;
    const uint32_t group_size_m = min(grid_y - first_pid_m, GROUP_M);
    const uint32_t in_group = pid % num_pid_in_group;
    pid_m = first_pid_m + (in_group % group_size_m);
    pid_n = in_group / group_size_m;
}

__device__ __forceinline__ void gemm_fma_8x8(
    float acc[8][8],
    const float4& fa0, const float4& fa1,
    const float4& fb0, const float4& fb1
) {
    const float av[8] = {fa0.x, fa0.y, fa0.z, fa0.w, fa1.x, fa1.y, fa1.z, fa1.w};
    const float bv[8] = {fb0.x, fb0.y, fb0.z, fb0.w, fb1.x, fb1.y, fb1.z, fb1.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }
}

template <bool kTransA, bool kTransB, bool kFusedLinearAdd>
__device__ __forceinline__ void cuda_sgemm_128x128x8_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* res,
    const float* __restrict__ bias,
    void* C_out,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t accumulate, uint32_t out_bf16
) {
    constexpr uint32_t BM = 128u;
    constexpr uint32_t BN = 128u;
    constexpr uint32_t BK = 8u;
    constexpr uint32_t PAD = 4u; // 132 stride: 16B 对齐且转置散射写 (4*132)%32=16 零 Bank 冲突

    uint32_t pid_m, pid_n;
    gemm_swizzle_cta<8u>(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, pid_m, pid_n);
    const uint32_t block_m = pid_m * BM;
    const uint32_t block_n = pid_n * BN;
    if (block_m >= M || block_n >= N) return;

    const uint32_t total_kt = (K + BK - 1u) / BK;
    uint32_t kt_start = 0u;
    uint32_t kt_end = total_kt;
    if (gridDim.z > 1u) {
        const uint32_t per_split = (total_kt + gridDim.z - 1u) / gridDim.z;
        kt_start = blockIdx.z * per_split;
        kt_end = min(total_kt, kt_start + per_split);
    }
    if (kt_start >= kt_end) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t warp_m = (warp_id >> 2) * 64u;
    const uint32_t warp_n = (warp_id & 3u) * 32u;
    const uint32_t m0 = warp_m + (lane >> 2) * 4u;
    const uint32_t m1 = m0 + 32u;
    const uint32_t n0 = warp_n + (lane & 3u) * 4u;
    const uint32_t n1 = n0 + 16u;

    __shared__ alignas(16) float As[2][BK][BM + PAD];
    __shared__ alignas(16) float Bs[2][BK][BN + PAD];

    const bool vec_a = kTransA
        ? (((M & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u))
        : (((K & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u));
    const bool vec_b = kTransB
        ? (((K & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u))
        : (((N & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u));

    auto load_a = [&](uint32_t k0) __device__ __forceinline__ -> float4 {
        if constexpr (!kTransA) {
            const uint32_t r = tid >> 1;
            const uint32_t c = (tid & 1u) * 4u;
            const uint32_t gm = block_m + r;
            const uint32_t gk = k0 + c;
            if (gm >= M) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const size_t base = size_t(gm) * K;
            if (vec_a && gk + 4u <= K) return *reinterpret_cast<const float4*>(A + base + gk);
            return make_float4(
                (gk + 0u < K) ? A[base + gk + 0u] : 0.0f,
                (gk + 1u < K) ? A[base + gk + 1u] : 0.0f,
                (gk + 2u < K) ? A[base + gk + 2u] : 0.0f,
                (gk + 3u < K) ? A[base + gk + 3u] : 0.0f);
        } else {
            const uint32_t r = tid >> 5;
            const uint32_t c = (tid & 31u) * 4u;
            const uint32_t gk = k0 + r;
            const uint32_t gm = block_m + c;
            if (gk >= K) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const size_t base = size_t(gk) * M;
            if (vec_a && gm + 4u <= M) return *reinterpret_cast<const float4*>(A + base + gm);
            return make_float4(
                (gm + 0u < M) ? A[base + gm + 0u] : 0.0f,
                (gm + 1u < M) ? A[base + gm + 1u] : 0.0f,
                (gm + 2u < M) ? A[base + gm + 2u] : 0.0f,
                (gm + 3u < M) ? A[base + gm + 3u] : 0.0f);
        }
    };

    auto store_a = [&](uint32_t stage, const float4& v) __device__ __forceinline__ {
        if constexpr (!kTransA) {
            const uint32_t r = tid >> 1;
            const uint32_t c = (tid & 1u) * 4u;
            As[stage][c + 0u][r] = v.x;
            As[stage][c + 1u][r] = v.y;
            As[stage][c + 2u][r] = v.z;
            As[stage][c + 3u][r] = v.w;
        } else {
            *reinterpret_cast<float4*>(&As[stage][tid >> 5][(tid & 31u) * 4u]) = v;
        }
    };

    auto load_b = [&](uint32_t k0) __device__ __forceinline__ -> float4 {
        if constexpr (!kTransB) {
            const uint32_t r = tid >> 5;
            const uint32_t c = (tid & 31u) * 4u;
            const uint32_t gk = k0 + r;
            const uint32_t gn = block_n + c;
            if (gk >= K) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const size_t base = size_t(gk) * N;
            if (vec_b && gn + 4u <= N) return *reinterpret_cast<const float4*>(B + base + gn);
            return make_float4(
                (gn + 0u < N) ? B[base + gn + 0u] : 0.0f,
                (gn + 1u < N) ? B[base + gn + 1u] : 0.0f,
                (gn + 2u < N) ? B[base + gn + 2u] : 0.0f,
                (gn + 3u < N) ? B[base + gn + 3u] : 0.0f);
        } else {
            const uint32_t r = tid >> 1;
            const uint32_t c = (tid & 1u) * 4u;
            const uint32_t gn = block_n + r;
            const uint32_t gk = k0 + c;
            if (gn >= N) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const size_t base = size_t(gn) * K;
            if (vec_b && gk + 4u <= K) return *reinterpret_cast<const float4*>(B + base + gk);
            return make_float4(
                (gk + 0u < K) ? B[base + gk + 0u] : 0.0f,
                (gk + 1u < K) ? B[base + gk + 1u] : 0.0f,
                (gk + 2u < K) ? B[base + gk + 2u] : 0.0f,
                (gk + 3u < K) ? B[base + gk + 3u] : 0.0f);
        }
    };

    auto store_b = [&](uint32_t stage, const float4& v) __device__ __forceinline__ {
        if constexpr (!kTransB) {
            *reinterpret_cast<float4*>(&Bs[stage][tid >> 5][(tid & 31u) * 4u]) = v;
        } else {
            const uint32_t r = tid >> 1;
            const uint32_t c = (tid & 1u) * 4u;
            Bs[stage][c + 0u][r] = v.x;
            Bs[stage][c + 1u][r] = v.y;
            Bs[stage][c + 2u][r] = v.z;
            Bs[stage][c + 3u][r] = v.w;
        }
    };

    float acc[8][8] = {0.0f};

    float4 ldg_a = load_a(kt_start * BK);
    float4 ldg_b = load_b(kt_start * BK);
    store_a(0u, ldg_a);
    store_b(0u, ldg_b);
    __syncthreads();

    uint32_t read_stage = 0u;
    for (uint32_t kt = kt_start; kt < kt_end; ++kt) {
        const uint32_t next_kt = kt + 1u;
        if (next_kt < kt_end) {
            ldg_a = load_a(next_kt * BK);
            ldg_b = load_b(next_kt * BK);
        }

        float4 fa0[2], fa1[2], fb0[2], fb1[2];
        fa0[0] = *reinterpret_cast<const float4*>(&As[read_stage][0][m0]);
        fa1[0] = *reinterpret_cast<const float4*>(&As[read_stage][0][m1]);
        fb0[0] = *reinterpret_cast<const float4*>(&Bs[read_stage][0][n0]);
        fb1[0] = *reinterpret_cast<const float4*>(&Bs[read_stage][0][n1]);

        #pragma unroll
        for (int kk = 0; kk < int(BK); ++kk) {
            if (kk + 1 < int(BK)) {
                const int rn = (kk + 1) & 1;
                fa0[rn] = *reinterpret_cast<const float4*>(&As[read_stage][kk + 1][m0]);
                fa1[rn] = *reinterpret_cast<const float4*>(&As[read_stage][kk + 1][m1]);
                fb0[rn] = *reinterpret_cast<const float4*>(&Bs[read_stage][kk + 1][n0]);
                fb1[rn] = *reinterpret_cast<const float4*>(&Bs[read_stage][kk + 1][n1]);
            }
            const int rc = kk & 1;
            gemm_fma_8x8(acc, fa0[rc], fa1[rc], fb0[rc], fb1[rc]);
        }

        if (next_kt < kt_end) {
            const uint32_t write_stage = read_stage ^ 1u;
            store_a(write_stage, ldg_a);
            store_b(write_stage, ldg_b);
            __syncthreads();
            read_stage = write_stage;
        }
    }

    float bias_reg[8] = {0.0f};
    if constexpr (kFusedLinearAdd) {
        if (bias) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t c0 = block_n + n0 + uint32_t(j);
                uint32_t c1 = block_n + n1 + uint32_t(j);
                bias_reg[j]     = (c0 < N) ? __ldg(&bias[c0]) : 0.0f;
                bias_reg[j + 4] = (c1 < N) ? __ldg(&bias[c1]) : 0.0f;
            }
        }
    }

    const bool split_k_atomic = (gridDim.z > 1u);
    const bool vec_c = !out_bf16 && !split_k_atomic && ((N & 3u) == 0u) &&
                       ((reinterpret_cast<uintptr_t>(C_out) & 15u) == 0u) &&
                       (!kFusedLinearAdd || ((reinterpret_cast<uintptr_t>(res) & 15u) == 0u));

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t row = block_m + ((i < 4) ? (m0 + uint32_t(i)) : (m1 + uint32_t(i - 4)));
        if (row >= M) continue;
        const size_t row_base = size_t(row) * N;

        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint32_t col_base = block_n + ((q == 0) ? n0 : n1);
            const int jb = q * 4;
            if (vec_c && col_base + 4u <= N) {
                float4 v4 = make_float4(acc[i][jb], acc[i][jb + 1], acc[i][jb + 2], acc[i][jb + 3]);
                float* dst = static_cast<float*>(C_out) + row_base + col_base;
                if constexpr (kFusedLinearAdd) {
                    float4 r4 = *reinterpret_cast<const float4*>(res + row_base + col_base);
                    v4 = make_float4(
                        v4.x + r4.x + bias_reg[jb],     v4.y + r4.y + bias_reg[jb + 1],
                        v4.z + r4.z + bias_reg[jb + 2], v4.w + r4.w + bias_reg[jb + 3]);
                } else if (accumulate) {
                    float4 p4 = *reinterpret_cast<const float4*>(dst);
                    v4 = make_float4(v4.x + p4.x, v4.y + p4.y, v4.z + p4.z, v4.w + p4.w);
                }
                *reinterpret_cast<float4*>(dst) = v4;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t col = col_base + uint32_t(j);
                    if (col >= N) continue;
                    const size_t idx = row_base + col;
                    float val = acc[i][jb + j];
                    if constexpr (kFusedLinearAdd) {
                        static_cast<float*>(C_out)[idx] = val + res[idx] + bias_reg[jb + j];
                    } else if (split_k_atomic) {
                        atomicAdd(static_cast<float*>(C_out) + idx, val);
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
    }
}

template <bool kTransA, bool kTransB, bool kFusedLinearAdd>
__device__ __forceinline__ void cuda_sgemm_64x64x8_impl(
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
    constexpr uint32_t BK = 8u;
    constexpr uint32_t PAD = 4u; // 68 stride: (4*68)%32=16 零 Bank 冲突

    uint32_t pid_m, pid_n;
    gemm_swizzle_cta<8u>(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, pid_m, pid_n);
    const uint32_t block_m = pid_m * BM;
    const uint32_t block_n = pid_n * BN;
    if (block_m >= M || block_n >= N) return;

    const uint32_t total_kt = (K + BK - 1u) / BK;
    uint32_t kt_start = 0u;
    uint32_t kt_end = total_kt;
    if (gridDim.z > 1u) {
        const uint32_t per_split = (total_kt + gridDim.z - 1u) / gridDim.z;
        kt_start = blockIdx.z * per_split;
        kt_end = min(total_kt, kt_start + per_split);
    }
    if (kt_start >= kt_end) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t m0 = (warp_id >> 2) * 32u + (lane >> 2) * 4u;
    const uint32_t n0 = (warp_id & 3u) * 16u + (lane & 3u) * 4u;

    __shared__ alignas(16) float As[2][BK][BM + PAD];
    __shared__ alignas(16) float Bs[2][BK][BN + PAD];

    const bool vec_a = kTransA
        ? (((M & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u))
        : (((K & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u));
    const bool vec_b = kTransB
        ? (((K & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u))
        : (((N & 3u) == 0u) && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u));

    auto load_ab = [&](uint32_t k0) __device__ __forceinline__ -> float4 {
        if (tid < 128u) {
            if constexpr (!kTransA) {
                const uint32_t r = tid >> 1;
                const uint32_t c = (tid & 1u) * 4u;
                const uint32_t gm = block_m + r;
                const uint32_t gk = k0 + c;
                if (gm >= M) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const size_t base = size_t(gm) * K;
                if (vec_a && gk + 4u <= K) return *reinterpret_cast<const float4*>(A + base + gk);
                return make_float4(
                    (gk + 0u < K) ? A[base + gk + 0u] : 0.0f,
                    (gk + 1u < K) ? A[base + gk + 1u] : 0.0f,
                    (gk + 2u < K) ? A[base + gk + 2u] : 0.0f,
                    (gk + 3u < K) ? A[base + gk + 3u] : 0.0f);
            } else {
                const uint32_t r = tid >> 4;
                const uint32_t c = (tid & 15u) * 4u;
                const uint32_t gk = k0 + r;
                const uint32_t gm = block_m + c;
                if (gk >= K) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const size_t base = size_t(gk) * M;
                if (vec_a && gm + 4u <= M) return *reinterpret_cast<const float4*>(A + base + gm);
                return make_float4(
                    (gm + 0u < M) ? A[base + gm + 0u] : 0.0f,
                    (gm + 1u < M) ? A[base + gm + 1u] : 0.0f,
                    (gm + 2u < M) ? A[base + gm + 2u] : 0.0f,
                    (gm + 3u < M) ? A[base + gm + 3u] : 0.0f);
            }
        } else {
            const uint32_t ltid = tid - 128u;
            if constexpr (!kTransB) {
                const uint32_t r = ltid >> 4;
                const uint32_t c = (ltid & 15u) * 4u;
                const uint32_t gk = k0 + r;
                const uint32_t gn = block_n + c;
                if (gk >= K) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const size_t base = size_t(gk) * N;
                if (vec_b && gn + 4u <= N) return *reinterpret_cast<const float4*>(B + base + gn);
                return make_float4(
                    (gn + 0u < N) ? B[base + gn + 0u] : 0.0f,
                    (gn + 1u < N) ? B[base + gn + 1u] : 0.0f,
                    (gn + 2u < N) ? B[base + gn + 2u] : 0.0f,
                    (gn + 3u < N) ? B[base + gn + 3u] : 0.0f);
            } else {
                const uint32_t r = ltid >> 1;
                const uint32_t c = (ltid & 1u) * 4u;
                const uint32_t gn = block_n + r;
                const uint32_t gk = k0 + c;
                if (gn >= N) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const size_t base = size_t(gn) * K;
                if (vec_b && gk + 4u <= K) return *reinterpret_cast<const float4*>(B + base + gk);
                return make_float4(
                    (gk + 0u < K) ? B[base + gk + 0u] : 0.0f,
                    (gk + 1u < K) ? B[base + gk + 1u] : 0.0f,
                    (gk + 2u < K) ? B[base + gk + 2u] : 0.0f,
                    (gk + 3u < K) ? B[base + gk + 3u] : 0.0f);
            }
        }
    };

    auto store_ab = [&](uint32_t stage, const float4& v) __device__ __forceinline__ {
        if (tid < 128u) {
            if constexpr (!kTransA) {
                const uint32_t r = tid >> 1;
                const uint32_t c = (tid & 1u) * 4u;
                As[stage][c + 0u][r] = v.x;
                As[stage][c + 1u][r] = v.y;
                As[stage][c + 2u][r] = v.z;
                As[stage][c + 3u][r] = v.w;
            } else {
                *reinterpret_cast<float4*>(&As[stage][tid >> 4][(tid & 15u) * 4u]) = v;
            }
        } else {
            const uint32_t ltid = tid - 128u;
            if constexpr (!kTransB) {
                *reinterpret_cast<float4*>(&Bs[stage][ltid >> 4][(ltid & 15u) * 4u]) = v;
            } else {
                const uint32_t r = ltid >> 1;
                const uint32_t c = (ltid & 1u) * 4u;
                Bs[stage][c + 0u][r] = v.x;
                Bs[stage][c + 1u][r] = v.y;
                Bs[stage][c + 2u][r] = v.z;
                Bs[stage][c + 3u][r] = v.w;
            }
        }
    };

    float acc[4][4] = {0.0f};
    float4 ldg_v = load_ab(kt_start * BK);
    store_ab(0u, ldg_v);
    __syncthreads();

    uint32_t read_stage = 0u;
    for (uint32_t kt = kt_start; kt < kt_end; ++kt) {
        const uint32_t next_kt = kt + 1u;
        if (next_kt < kt_end) {
            ldg_v = load_ab(next_kt * BK);
        }

        float4 fa[2], fb[2];
        fa[0] = *reinterpret_cast<const float4*>(&As[read_stage][0][m0]);
        fb[0] = *reinterpret_cast<const float4*>(&Bs[read_stage][0][n0]);

        #pragma unroll
        for (int kk = 0; kk < int(BK); ++kk) {
            if (kk + 1 < int(BK)) {
                const int rn = (kk + 1) & 1;
                fa[rn] = *reinterpret_cast<const float4*>(&As[read_stage][kk + 1][m0]);
                fb[rn] = *reinterpret_cast<const float4*>(&Bs[read_stage][kk + 1][n0]);
            }
            const int rc = kk & 1;
            const float av[4] = {fa[rc].x, fa[rc].y, fa[rc].z, fa[rc].w};
            const float bv[4] = {fb[rc].x, fb[rc].y, fb[rc].z, fb[rc].w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }

        if (next_kt < kt_end) {
            const uint32_t write_stage = read_stage ^ 1u;
            store_ab(write_stage, ldg_v);
            __syncthreads();
            read_stage = write_stage;
        }
    }

    float bias_reg[4] = {0.0f};
    if constexpr (kFusedLinearAdd) {
        if (bias) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t col = block_n + n0 + uint32_t(j);
                bias_reg[j] = (col < N) ? __ldg(&bias[col]) : 0.0f;
            }
        }
    }

    const bool split_k_atomic = (gridDim.z > 1u);
    const bool vec_c = !out_bf16 && !split_k_atomic && ((N & 3u) == 0u) &&
                       ((reinterpret_cast<uintptr_t>(C_out) & 15u) == 0u) &&
                       (!kFusedLinearAdd || ((reinterpret_cast<uintptr_t>(res) & 15u) == 0u));

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t row = block_m + m0 + uint32_t(i);
        if (row >= M) continue;
        const size_t row_base = size_t(row) * N;
        const uint32_t col_base = block_n + n0;

        if (vec_c && col_base + 4u <= N) {
            float4 v4 = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            float* dst = static_cast<float*>(C_out) + row_base + col_base;
            if constexpr (kFusedLinearAdd) {
                float4 r4 = *reinterpret_cast<const float4*>(res + row_base + col_base);
                v4 = make_float4(
                    v4.x + r4.x + bias_reg[0], v4.y + r4.y + bias_reg[1],
                    v4.z + r4.z + bias_reg[2], v4.w + r4.w + bias_reg[3]);
            } else if (accumulate) {
                float4 p4 = *reinterpret_cast<const float4*>(dst);
                v4 = make_float4(v4.x + p4.x, v4.y + p4.y, v4.z + p4.z, v4.w + p4.w);
            }
            *reinterpret_cast<float4*>(dst) = v4;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t col = col_base + uint32_t(j);
                if (col >= N) continue;
                const size_t idx = row_base + col;
                float val = acc[i][j];
                if constexpr (kFusedLinearAdd) {
                    static_cast<float*>(C_out)[idx] = val + res[idx] + bias_reg[j];
                } else if (split_k_atomic) {
                    atomicAdd(static_cast<float*>(C_out) + idx, val);
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
}

__global__ __launch_bounds__(256, 2) void cuda_sgemm_nn_128_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_128x128x8_impl<false, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__global__ __launch_bounds__(256, 2) void cuda_sgemm_nt_128_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_128x128x8_impl<false, true, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__global__ __launch_bounds__(256, 2) void cuda_sgemm_tn_128_kernel(
    const float* __restrict__ A, const float* __restrict__ B, void* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate, uint32_t out_bf16
) {
    cuda_sgemm_128x128x8_impl<true, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, out_bf16);
}

__global__ __launch_bounds__(256, 3) void cuda_sgemm_nn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_64x64x8_impl<false, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__global__ __launch_bounds__(256, 3) void cuda_sgemm_nt_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate
) {
    cuda_sgemm_64x64x8_impl<false, true, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, 0u);
}

__global__ __launch_bounds__(256, 3) void cuda_sgemm_tn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, void* C_out,
    uint32_t M, uint32_t N, uint32_t K, uint32_t accumulate, uint32_t out_bf16
) {
    cuda_sgemm_64x64x8_impl<true, false, false>(A, B, nullptr, nullptr, C_out, M, N, K, accumulate, out_bf16);
}