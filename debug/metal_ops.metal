#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// 解决 Shared Memory Bank Conflict 的 Morton 映射
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
    ushort lane_id = thread_index_in_simdgroup;
    ushort quad_id = lane_id / 4;
    ushort M_floor_of_quadrant = (quad_id / 4) * 4;
    ushort M_in_quadrant = (lane_id / 2) % 4;
    ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;
    ushort N_floor_of_quadrant = (quad_id & 2) * 2;
    ushort N_in_quadrant = (lane_id % 2) * 2;
    ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;
    return ushort2(N_in_simd, M_in_simd);
}

METAL_FUNC static float2 load_device_8x8_pair(
    device const float* src,
    uint row_stride,
    uint row_base,
    uint col_base,
    uint extent,
    ushort lane) {
    const ushort2 offset = morton_order(lane);
    const uint row = row_base + uint(offset.y);
    const uint col = col_base + uint(offset.x);
    if (row < extent && (col + 1u) < extent) {
        return float2(*reinterpret_cast<device const packed_float2*>(src + row * row_stride + col));
    }
    const float first = (row < extent && col < extent) ? src[row * row_stride + col] : 0.0f;
    const float second = (row < extent && (col + 1u) < extent) ? src[row * row_stride + col + 1u] : 0.0f;
    return float2(first, second);
}

METAL_FUNC static float fused_mlp_gelu_scalar(float value) {
    const float scale = sqrt(2.0f / M_PI_F);
    const float cube = value * value * value;
    float inner = scale * (value + 0.044715f * cube);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * value * (1.0f + fast::tanh(inner));
}

METAL_FUNC static float4 fused_mlp_gelu_float4_exact(float4 value) {
    return float4(
        fused_mlp_gelu_scalar(value.x),
        fused_mlp_gelu_scalar(value.y),
        fused_mlp_gelu_scalar(value.z),
        fused_mlp_gelu_scalar(value.w)
    );
}

#pragma METAL internals : enable

namespace lowreg_matrix {

template <typename T>
METAL_FUNC vec<T, 2> load_tg_8x8(
    threadgroup const T* src,
    uint stride,
    ushort2 origin,
    ushort lane,
    bool transpose = false) {
    const ushort2 offset = morton_order(lane);
    const uint row = uint(origin.x) + uint(offset.y);
    const uint col = uint(origin.y) + uint(offset.x);
    if (transpose) {
        return vec<T, 2>(src[col * stride + row], src[(col + 1u) * stride + row]);
    }
    return vec<T, 2>(src[row * stride + col], src[row * stride + col + 1u]);
}

template <typename T>
METAL_FUNC void store_tg_8x8(
    threadgroup T* dst,
    uint stride,
    ushort2 origin,
    ushort lane,
    vec<T, 2> value,
    bool transpose = false) {
    const ushort2 offset = morton_order(lane);
    if (transpose) {
        const uint row = uint(origin.x) + uint(offset.x);
        const uint col = uint(origin.y) + uint(offset.y);
        dst[col * stride + row] = value[0];
        dst[(col + 1u) * stride + row] = value[1];
        return;
    }

    const uint row = uint(origin.x) + uint(offset.y);
    const uint col = uint(origin.y) + uint(offset.x);
    dst[row * stride + col] = value[0];
    dst[row * stride + col + 1u] = value[1];
}

template <typename T, typename U, typename V>
METAL_FUNC vec<T, 2> mma_8x8(
    vec<U, 2> a_value,
    vec<V, 2> b_value,
    vec<T, 2> accumulator) {
    typedef vec<U, 64> native_a_type;
    typedef vec<V, 64> native_b_type;
    typedef vec<T, 64> native_c_type;
    native_a_type a_native;
    native_b_type b_native;
    native_c_type c_native;

    *reinterpret_cast<thread vec<U, 2>*>(&a_native) = a_value;
    *reinterpret_cast<thread vec<V, 2>*>(&b_native) = b_value;
    *reinterpret_cast<thread vec<T, 2>*>(&c_native) = accumulator;
    c_native = __metal_simdgroup_matrix_8x8_multiply_accumulate(a_native, b_native, c_native, native_c_type());

    return *reinterpret_cast<thread vec<T, 2>*>(&c_native);
}

template <typename T, typename U, typename V>
METAL_FUNC void mma_8x8_accumulate_native(
    vec<U, 2> a_value,
    vec<V, 2> b_value,
    thread vec<T, 64>& c_native) {
    typedef vec<U, 64> native_a_type;
    typedef vec<V, 64> native_b_type;
    typedef vec<T, 64> native_c_type;

    native_a_type a_native;
    native_b_type b_native;

    *reinterpret_cast<thread vec<U, 2>*>(&a_native) = a_value;
    *reinterpret_cast<thread vec<V, 2>*>(&b_native) = b_value;

    c_native = __metal_simdgroup_matrix_8x8_multiply_accumulate(
            a_native, b_native, c_native, native_c_type());
}

} // namespace lowreg_matrix

#pragma METAL internals : disable

kernel void fill_scalar_kernel(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) {
    if (id < threads_per_grid) out[id] = value;
}

kernel void fill_scalar_kernel2(
    device uchar* out         [[buffer(0)]],
    constant uint& value_bits [[buffer(1)]],
    constant uint& itemsize   [[buffer(2)]],
    uint id                   [[thread_position_in_grid]],
    uint count                [[threads_per_grid]]
) {
    if (id >= count) {
        return;
    }
    const ulong base = ulong(id) * ulong(itemsize);
    for (uint byte = 0; byte < itemsize; ++byte) {
        out[base + byte] = uchar((value_bits >> (byte * 8u)) & 0xffu);
    }
}

inline void atomic_add_float(device atomic_uint* target, float value) {
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    uint desired;
    do {
        float current_float = as_type<float>(expected);
        if (isnan(current_float) || isnan(value)) {
            desired = as_type<uint>(NAN);
            atomic_compare_exchange_weak_explicit(target, &expected, desired, memory_order_relaxed, memory_order_relaxed);
            break; 
        }
        float desired_float = current_float + value;
        desired = as_type<uint>(desired_float);
    } while (!atomic_compare_exchange_weak_explicit(target, &expected, desired, memory_order_relaxed, memory_order_relaxed));
}

kernel void add_residual_forward_vec4(
    device const float4* x [[buffer(0)]],
    device const float4* y [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    out[gid] = x[gid] + y[gid];
}

kernel void add_residual_backward_vec4(
    device const float4* grad_out [[buffer(0)]],
    device float4* grad_x [[buffer(1)]],
    device float4* grad_y [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const float4 g = grad_out[gid];
    if (grad_x) {
        grad_x[gid] = g;
    }
    if (grad_y) {
        grad_y[gid] = g;
    }
}

kernel void add_residual_forward(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = x[gid] + y[gid];
}

kernel void add_residual_backward(
    device const float* grad_out [[buffer(0)]],
    device float* grad_x [[buffer(1)]],
    device float* grad_y [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float g = grad_out[gid];

    grad_x[gid] = g;
    grad_y[gid] = g;
}

kernel void add_pos_forward(
    device const float* x [[buffer(0)]],
    device const float* pos [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint tc = gid % (T*C);
    out[gid] = x[gid] + pos[tc];
}

kernel void pos_emb_grad(
    device const float* grad_out [[buffer(0)]],
    device float* grad_pos [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint tc = gid;
    uint t = tc / C;
    uint c = tc % C;

    float sum = 0;
    for(uint b=0;b<B;b++) {
        sum += grad_out[(b*T+t)*C+c];
    }
    grad_pos[tc] = sum;
}

kernel void contiguous_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device float* grad_x [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) 
{
    if (id < threads_per_grid) {
        if (grad_x) grad_x[id] = grad_out[id];
    }
}

kernel void add_bias_forward(
    device const float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint c = gid % C;
    out[gid] = x[gid] + bias[c];
}

kernel void mul_forward_kernel(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& n0 [[buffer(3)]],
    constant uint& n1 [[buffer(4)]],
    constant uint& n2 [[buffer(5)]],
    constant uint& n3 [[buffer(6)]],
    constant uint& sx0 [[buffer(7)]],
    constant uint& sx1 [[buffer(8)]],
    constant uint& sx2 [[buffer(9)]],
    constant uint& sx3 [[buffer(10)]],
    constant uint& sy0 [[buffer(11)]],
    constant uint& sy1 [[buffer(12)]],
    constant uint& sy2 [[buffer(13)]],
    constant uint& sy3 [[buffer(14)]],
    constant uint& ndim [[buffer(15)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    if (id >= threads_per_grid) return;
    uint shape[4] = {n0, n1, n2, n3};
    uint sx[4] = {sx0, sx1, sx2, sx3};
    uint sy[4] = {sy0, sy1, sy2, sy3};
    uint linear = id, idx_x = 0, idx_y = 0;
    for (int i = int(ndim) - 1; i >= 0; --i) {
        uint coord = linear % shape[i];
        idx_x += coord * sx[i];
        idx_y += coord * sy[i];
        linear /= shape[i];
    }
    out[id] = x[idx_x] * y[idx_y];
}

kernel void mul_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device const float* y        [[buffer(2)]],
    device float* grad_x         [[buffer(3)]],
    device float* grad_y         [[buffer(4)]],
    constant uint& n0  [[buffer(5)]],
    constant uint& n1  [[buffer(6)]],
    constant uint& n2  [[buffer(7)]],
    constant uint& n3  [[buffer(8)]],
    constant uint& sx0 [[buffer(9)]],
    constant uint& sx1 [[buffer(10)]],
    constant uint& sx2 [[buffer(11)]],
    constant uint& sx3 [[buffer(12)]],
    constant uint& sy0 [[buffer(13)]],
    constant uint& sy1 [[buffer(14)]],
    constant uint& sy2 [[buffer(15)]],
    constant uint& sy3 [[buffer(16)]],
    constant uint& ndim [[buffer(17)]],
    constant uint& do_x [[buffer(18)]],
    constant uint& do_y [[buffer(19)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    if (id >= threads_per_grid) return;
    uint shape[4] = {n0, n1, n2, n3};
    uint sx[4] = {sx0, sx1, sx2, sx3};
    uint sy[4] = {sy0, sy1, sy2, sy3};
    uint linear = id;
    uint idx_x = 0;
    uint idx_y = 0;
    for (int i = int(ndim) - 1; i >= 0; --i) {
        uint coord = linear % shape[i];
        idx_x += coord * sx[i];
        idx_y += coord * sy[i];
        linear /= shape[i];
    }
    const float g = grad_out[id];
    // dx: 训练里 x 通常与 out 同形；用 atomic 兼容广播
    if (do_x != 0u) {
        atomic_add_float(reinterpret_cast<device atomic_uint*>(&grad_x[idx_x]), g * y[idx_y]);
    }
    // dy: scalar / row 广播会多线程写同一地址，必须 atomic
    if (do_y != 0u) {
        atomic_add_float(reinterpret_cast<device atomic_uint*>(&grad_y[idx_y]), g * x[idx_x]);
    }
}

kernel void mul_backward_scalar_dx(
    device const float* grad_out [[buffer(0)]],
    device const float* y        [[buffer(1)]],
    device float* grad_x         [[buffer(2)]],
    constant uint& n             [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }

    grad_x[id] += grad_out[id] * y[0];
}

kernel void mul_backward_scalar_dy(
    device const float* grad_out [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* grad_y         [[buffer(2)]],
    constant uint& n             [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    // Python 端固定使用 256 个线程。
    threadgroup float partial_sum[256];

    float local_sum = 0.0f;

    // 256 个线程分别处理 0,256,512... 等位置。
    for (uint i = tid; i < n; i += tpg) {
        local_sum += grad_out[i] * x[i];
    }

    partial_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 固定顺序的树形归约。
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 只有一个线程写 grad_y，因此不需要 atomic。
    if (tid == 0u) {
        grad_y[0] += partial_sum[0];
    }
}

kernel void gather_forward_kernel(
    device const int* indices [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) 
{
    if (id < threads_per_grid) {
        uint row = id / C;
        uint col = id % C;
        int idx = indices[row];
        out[id] = weight[idx * C + col];
    }
}

kernel void gather_backward_kernel(
    device const int* indices        [[buffer(0)]],
    device const float* grad_out     [[buffer(1)]],
    device atomic_uint* grad_weight  [[buffer(2)]],
    constant uint& C                 [[buffer(3)]],
    uint id                          [[thread_position_in_grid]],
    uint threads_per_grid            [[threads_per_grid]]
) {
    if (id >= threads_per_grid || C == 0u) {
        return;
    }

    const uint row = id / C;
    const uint col = id - row * C;
    const uint token = uint(indices[row]);
    const uint destination = token * C + col;

    atomic_add_float(
        grad_weight + destination,
        grad_out[id]
    );
}

kernel void contiguous_forward_kernel(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* strides [[buffer(3)]],
    constant uint* p_ndim [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    if (id >= threads_per_grid) return;
    const uint ndim = p_ndim[0];
    uint linear_idx = id;
    uint strided_idx = 0;
    for (int i = int(ndim) - 1; i >= 0; --i) {
        const uint dim_size = shape[i];
        const uint coord = linear_idx % dim_size;
        strided_idx += coord * strides[i];
        linear_idx /= dim_size;
    }
    out[id] = x[strided_idx];
}

kernel void fused_linear_add_forward_kernel(
    device const float* x           [[buffer(0)]],
    device const float* act         [[buffer(1)]],
    device const float* w           [[buffer(2)]],
    device const float* b           [[buffer(3)]],
    device float* out               [[buffer(4)]],
    constant uint& M                [[buffer(5)]],
    constant uint& K                [[buffer(6)]],
    constant uint& N                [[buffer(7)]],
    constant uint& alias_x_out      [[buffer(8)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]])
{
    constexpr uint kThreads = 256u;
    constexpr uint kBM = 32u;
    constexpr uint kBN = 64u;
    constexpr uint kBK = 32u;
    constexpr uint kFrag = 8u;

    constexpr uint kAStride = kBK + 1u;
    constexpr uint kWStride = kBN + 1u;
    constexpr uint kK4PerTile = kBK / 4u;
    constexpr uint kN4PerTile = kBN / 4u;

    constexpr uint kAFloat4Count = kBM * kK4PerTile;
    constexpr uint kWFloat4Count = kBK * kN4PerTile;

    threadgroup float A_tg[kBM][kAStride];
    threadgroup float W_tg[kBK][kWStride];

    const uint tid = thread_pos.x;

    if (tid >= kThreads ||
        simd_lane_count != 32u ||
        tg_pos.x != 0u ||
        M == 0u ||
        K == 0u ||
        N == 0u) {
        return;
    }

    (void)alias_x_out;

    const uint output_base = tg_pos.y * kBN;
    const uint row_base0 = tg_pos.z * (2u * kBM);
    const uint row_base1 = row_base0 + kBM;

    if (output_base >= N || row_base0 >= M) {
        return;
    }

    const bool have_tile1 = row_base1 < M;

    const uint matrix_row_group = simd_id >> 1u;
    const uint matrix_col_group = simd_id & 1u;
    const uint simd_row_base = matrix_row_group * kFrag;
    const uint simd_col_base = matrix_col_group * 32u;

    float2 accumulator0 = float2(0.0f);
    float2 accumulator1 = float2(0.0f);
    float2 accumulator2 = float2(0.0f);
    float2 accumulator3 = float2(0.0f);

    float2 accumulator0b = float2(0.0f);
    float2 accumulator1b = float2(0.0f);
    float2 accumulator2b = float2(0.0f);
    float2 accumulator3b = float2(0.0f);

    for (uint input_base = 0u;
         input_base < K;
         input_base += kBK) {

        // W[K,N] -> W_tg[32,64].
        // 每个 K-tile 只从 device 加载一次，供两个 M-tile 共用。
        for (uint index = tid;
             index < kWFloat4Count;
             index += kThreads) {

            const uint local_k = index / kN4PerTile;
            const uint local_n4 =
                index - local_k * kN4PerTile;

            const uint global_k = input_base + local_k;
            const uint global_n =
                output_base + local_n4 * 4u;

            float4 value = float4(0.0f);

            if (global_k < K && global_n + 3u < N) {
                value = float4(
                    *reinterpret_cast<
                        device const packed_float4*>(
                            w + global_k * N + global_n
                        )
                );
            } else if (global_k < K && global_n < N) {
                #pragma unroll
                for (uint component = 0u;
                     component < 4u;
                     ++component) {

                    const uint n = global_n + component;
                    if (n < N) {
                        value[component] =
                            w[global_k * N + n];
                    }
                }
            }

            const uint local_n = local_n4 * 4u;

            *reinterpret_cast<
                threadgroup packed_float4*>(
                    &W_tg[local_k][local_n]
                ) = packed_float4(value);
        }

        // 第一块 act[row_base0:row_base0+32, input_base:input_base+32]
        // -> A_tg[32,32]。
        for (uint index = tid;
             index < kAFloat4Count;
             index += kThreads) {

            const uint local_row = index / kK4PerTile;
            const uint local_k4 =
                index - local_row * kK4PerTile;

            const uint global_row = row_base0 + local_row;
            const uint global_k =
                input_base + local_k4 * 4u;

            float4 value = float4(0.0f);

            if (global_row < M && global_k + 3u < K) {
                value = float4(
                    *reinterpret_cast<
                        device const packed_float4*>(
                            act + global_row * K + global_k
                        )
                );
            } else if (global_row < M && global_k < K) {
                #pragma unroll
                for (uint component = 0u;
                     component < 4u;
                     ++component) {

                    const uint k = global_k + component;
                    if (k < K) {
                        value[component] =
                            act[global_row * K + k];
                    }
                }
            }

            const uint local_k = local_k4 * 4u;

            *reinterpret_cast<
                threadgroup packed_float4*>(
                    &A_tg[local_row][local_k]
                ) = packed_float4(value);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 第一块 32x64 M-tile。
        for (uint k_sub = 0u;
             k_sub < kBK;
             k_sub += kFrag) {

            const float2 a_fragment =
                lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<
                        threadgroup const float*>(A_tg),
                    kAStride,
                    ushort2(
                        ushort(simd_row_base),
                        ushort(k_sub)
                    ),
                    ushort(lane)
                );

            const float2 w_fragment0 =
                lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<
                        threadgroup const float*>(W_tg),
                    kWStride,
                    ushort2(
                        ushort(k_sub),
                        ushort(simd_col_base)
                    ),
                    ushort(lane)
                );

            const float2 w_fragment1 =
                lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<
                        threadgroup const float*>(W_tg),
                    kWStride,
                    ushort2(
                        ushort(k_sub),
                        ushort(simd_col_base + 8u)
                    ),
                    ushort(lane)
                );

            const float2 w_fragment2 =
                lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<
                        threadgroup const float*>(W_tg),
                    kWStride,
                    ushort2(
                        ushort(k_sub),
                        ushort(simd_col_base + 16u)
                    ),
                    ushort(lane)
                );

            const float2 w_fragment3 =
                lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<
                        threadgroup const float*>(W_tg),
                    kWStride,
                    ushort2(
                        ushort(k_sub),
                        ushort(simd_col_base + 24u)
                    ),
                    ushort(lane)
                );

            accumulator0 =
                lowreg_matrix::mma_8x8(
                    a_fragment,
                    w_fragment0,
                    accumulator0
                );

            accumulator1 =
                lowreg_matrix::mma_8x8(
                    a_fragment,
                    w_fragment1,
                    accumulator1
                );

            accumulator2 =
                lowreg_matrix::mma_8x8(
                    a_fragment,
                    w_fragment2,
                    accumulator2
                );

            accumulator3 =
                lowreg_matrix::mma_8x8(
                    a_fragment,
                    w_fragment3,
                    accumulator3
                );
        }

        // 所有 SG 用完第一块 A_tg 后，才能覆盖 A_tg。
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (have_tile1) {
            // 第二块 act[row_base1:row_base1+32, ...] 覆盖 A_tg。
            // W_tg 保持不变。
            for (uint index = tid;
                 index < kAFloat4Count;
                 index += kThreads) {

                const uint local_row = index / kK4PerTile;
                const uint local_k4 =
                    index - local_row * kK4PerTile;

                const uint global_row =
                    row_base1 + local_row;
                const uint global_k =
                    input_base + local_k4 * 4u;

                float4 value = float4(0.0f);

                if (global_row < M && global_k + 3u < K) {
                    value = float4(
                        *reinterpret_cast<
                            device const packed_float4*>(
                                act + global_row * K + global_k
                            )
                    );
                } else if (global_row < M &&
                           global_k < K) {

                    #pragma unroll
                    for (uint component = 0u;
                         component < 4u;
                         ++component) {

                        const uint k =
                            global_k + component;

                        if (k < K) {
                            value[component] =
                                act[global_row * K + k];
                        }
                    }
                }

                const uint local_k = local_k4 * 4u;

                *reinterpret_cast<
                    threadgroup packed_float4*>(
                        &A_tg[local_row][local_k]
                    ) = packed_float4(value);
            }

            threadgroup_barrier(
                mem_flags::mem_threadgroup
            );

            // 第二块 32x64 M-tile，复用同一块 W_tg。
            for (uint k_sub = 0u;
                 k_sub < kBK;
                 k_sub += kFrag) {

                const float2 a_fragment =
                    lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<
                            threadgroup const float*>(A_tg),
                        kAStride,
                        ushort2(
                            ushort(simd_row_base),
                            ushort(k_sub)
                        ),
                        ushort(lane)
                    );

                const float2 w_fragment0 =
                    lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<
                            threadgroup const float*>(W_tg),
                        kWStride,
                        ushort2(
                            ushort(k_sub),
                            ushort(simd_col_base)
                        ),
                        ushort(lane)
                    );

                const float2 w_fragment1 =
                    lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<
                            threadgroup const float*>(W_tg),
                        kWStride,
                        ushort2(
                            ushort(k_sub),
                            ushort(simd_col_base + 8u)
                        ),
                        ushort(lane)
                    );

                const float2 w_fragment2 =
                    lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<
                            threadgroup const float*>(W_tg),
                        kWStride,
                        ushort2(
                            ushort(k_sub),
                            ushort(simd_col_base + 16u)
                        ),
                        ushort(lane)
                    );

                const float2 w_fragment3 =
                    lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<
                            threadgroup const float*>(W_tg),
                        kWStride,
                        ushort2(
                            ushort(k_sub),
                            ushort(simd_col_base + 24u)
                        ),
                        ushort(lane)
                    );

                accumulator0b =
                    lowreg_matrix::mma_8x8(
                        a_fragment,
                        w_fragment0,
                        accumulator0b
                    );

                accumulator1b =
                    lowreg_matrix::mma_8x8(
                        a_fragment,
                        w_fragment1,
                        accumulator1b
                    );

                accumulator2b =
                    lowreg_matrix::mma_8x8(
                        a_fragment,
                        w_fragment2,
                        accumulator2b
                    );

                accumulator3b =
                    lowreg_matrix::mma_8x8(
                        a_fragment,
                        w_fragment3,
                        accumulator3b
                    );
            }

            // 保护 W_tg/A_tg，防止下一轮 K-tile 提前覆盖。
            threadgroup_barrier(
                mem_flags::mem_threadgroup
            );
        } else {
            threadgroup_barrier(
                mem_flags::mem_threadgroup
            );
        }
    }

    // K 循环结束，A_tg 可复用。
    // 平坦使用前 64 个 float 缓存当前 N 条带的 bias。
    threadgroup float* bias_tg =
        reinterpret_cast<threadgroup float*>(
            &A_tg[0][0]
        );

    if (tid < kBN) {
        const uint global_n = output_base + tid;
        bias_tg[tid] =
            global_n < N ? b[global_n] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort2 matrix_offset =
        morton_order(ushort(lane));

    const uint local_row =
        simd_row_base + uint(matrix_offset.y);

    // 第一块 32x64 的 Morton 标量 epilogue。
    const uint global_row0 = row_base0 + local_row;

    if (global_row0 < M) {
        #pragma unroll
        for (uint fragment_index = 0u;
             fragment_index < 4u;
             ++fragment_index) {

            float2 accumulator = float2(0.0f);

            if (fragment_index == 0u) {
                accumulator = accumulator0;
            } else if (fragment_index == 1u) {
                accumulator = accumulator1;
            } else if (fragment_index == 2u) {
                accumulator = accumulator2;
            } else {
                accumulator = accumulator3;
            }

            const uint fragment_column =
                simd_col_base +
                uint(matrix_offset.x) +
                fragment_index * 8u;

            #pragma unroll
            for (uint component = 0u;
                 component < 2u;
                 ++component) {

                const uint local_n =
                    fragment_column + component;
                const uint global_n =
                    output_base + local_n;

                if (global_n >= N) {
                    continue;
                }

                const uint output_index =
                    global_row0 * N + global_n;

                out[output_index] =
                    accumulator[component] +
                    x[output_index] +
                    bias_tg[local_n];
            }
        }
    }

    // 第二块复用相同 Morton 坐标及 bias_tg。
    if (have_tile1) {
        const uint global_row1 =
            row_base1 + local_row;

        if (global_row1 < M) {
            #pragma unroll
            for (uint fragment_index = 0u;
                 fragment_index < 4u;
                 ++fragment_index) {

                float2 accumulator = float2(0.0f);

                if (fragment_index == 0u) {
                    accumulator = accumulator0b;
                } else if (fragment_index == 1u) {
                    accumulator = accumulator1b;
                } else if (fragment_index == 2u) {
                    accumulator = accumulator2b;
                } else {
                    accumulator = accumulator3b;
                }

                const uint fragment_column =
                    simd_col_base +
                    uint(matrix_offset.x) +
                    fragment_index * 8u;

                #pragma unroll
                for (uint component = 0u;
                     component < 2u;
                     ++component) {

                    const uint local_n =
                        fragment_column + component;
                    const uint global_n =
                        output_base + local_n;

                    if (global_n >= N) {
                        continue;
                    }

                    const uint output_index =
                        global_row1 * N + global_n;

                    out[output_index] =
                        accumulator[component] +
                        x[output_index] +
                        bias_tg[local_n];
                }
            }
        }
    }
}

kernel void strided_add_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device float* grad_x [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* strides [[buffer(3)]],
    constant uint* p_ndim [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) 
{
    if (id < threads_per_grid) {
        uint linear_idx = id;
        uint strided_idx = 0;
        uint ndim = p_ndim[0];
        
        for (int i = ndim - 1; i >= 0; --i) {
            uint dim_size = shape[i];
            uint coord = linear_idx % dim_size;
            strided_idx += coord * strides[i];
            linear_idx /= dim_size;
        }
        if (grad_x) grad_x[id] = grad_out[strided_idx];
    }
}

kernel void bias_grad_reduce(
    device const float* grad_out [[buffer(0)]],
    device float* grad_bias      [[buffer(1)]],
    constant uint& rows          [[buffer(2)]],
    constant uint& C             [[buffer(3)]],
    constant uint& accumulate    [[buffer(4)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kSimdGroups = kThreads / kSimdWidth;

    constexpr uint kBN = 16u;
    constexpr uint kVecWidth = 4u;
    constexpr uint kVecsPerBlock = kBN / kVecWidth;

    const uint col0 = tg_pos.y * kBN;
    const uint valid_cols = min(kBN, C - col0);
    const uint full_vecs = valid_cols / kVecWidth;
    const uint tail_cols = valid_cols % kVecWidth;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    for (uint row = tid; row < rows; row += kThreads) {
        const device float* src =
            grad_out + row * C + col0;

        if (full_vecs >= 1u) {
            acc0 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 0u));
        }

        if (full_vecs >= 2u) {
            acc1 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 4u));
        }

        if (full_vecs >= 3u) {
            acc2 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 8u));
        }

        if (full_vecs >= 4u) {
            acc3 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 12u));
        }

        if (tail_cols != 0u) {
            const device float* tail_src =
                src + full_vecs * kVecWidth;

            float4 tail = float4(0.0f);

            if (tail_cols >= 1u) {
                tail.x = tail_src[0];
            }
            if (tail_cols >= 2u) {
                tail.y = tail_src[1];
            }
            if (tail_cols >= 3u) {
                tail.z = tail_src[2];
            }

            if (full_vecs == 0u) {
                acc0 += tail;
            } else if (full_vecs == 1u) {
                acc1 += tail;
            } else if (full_vecs == 2u) {
                acc2 += tail;
            } else {
                acc3 += tail;
            }
        }
    }

    const float4 simd_sum0 = float4(
        simd_sum(acc0.x),
        simd_sum(acc0.y),
        simd_sum(acc0.z),
        simd_sum(acc0.w));

    const float4 simd_sum1 = float4(
        simd_sum(acc1.x),
        simd_sum(acc1.y),
        simd_sum(acc1.z),
        simd_sum(acc1.w));

    const float4 simd_sum2 = float4(
        simd_sum(acc2.x),
        simd_sum(acc2.y),
        simd_sum(acc2.z),
        simd_sum(acc2.w));

    const float4 simd_sum3 = float4(
        simd_sum(acc3.x),
        simd_sum(acc3.y),
        simd_sum(acc3.z),
        simd_sum(acc3.w));

    threadgroup float4 tg_partial[kSimdGroups][kVecsPerBlock];

    if (lane == 0u) {
        tg_partial[simd_id][0] = simd_sum0;
        tg_partial[simd_id][1] = simd_sum1;
        tg_partial[simd_id][2] = simd_sum2;
        tg_partial[simd_id][3] = simd_sum3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float4 sums[kVecsPerBlock];

        sums[0] = float4(0.0f);
        sums[1] = float4(0.0f);
        sums[2] = float4(0.0f);
        sums[3] = float4(0.0f);

        for (uint group = 0u; group < kSimdGroups; ++group) {
            sums[0] += tg_partial[group][0];
            sums[1] += tg_partial[group][1];
            sums[2] += tg_partial[group][2];
            sums[3] += tg_partial[group][3];
        }

        device float* dst = grad_bias + col0;

        // 完整 float4：packed 读写，允许非 16 字节对齐。
        for (uint vec = 0u; vec < full_vecs; ++vec) {
            device packed_float4* dst4 =
                reinterpret_cast<device packed_float4*>(
                    dst + vec * kVecWidth);

            float4 value = sums[vec];

            if (accumulate != 0u) {
                value += float4(*dst4);
            }

            *dst4 = packed_float4(value);
        }

        // 不完整尾部：只访问实际存在的 1–3 列。
        if (tail_cols != 0u) {
            const uint tail_base = full_vecs * kVecWidth;

            for (uint i = 0u; i < tail_cols; ++i) {
                const float value = sums[full_vecs][i];

                if (accumulate != 0u) {
                    dst[tail_base + i] += value;
                } else {
                    dst[tail_base + i] = value;
                }
            }
        }
    }
}

kernel void fused_rmsnorm_mlp_forward_kernel_bm16(
    device const float* x        [[buffer(0)]],
    device const float* w1       [[buffer(1)]],
    device const float* b1       [[buffer(2)]],
    device const float* w2       [[buffer(3)]],
    device const float* b2       [[buffer(4)]],
    device float* out            [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& C             [[buffer(7)]],
    constant uint& C4            [[buffer(8)]],
    constant uint& eps_bits      [[buffer(9)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 thread_pos             [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count         [[threads_per_simdgroup]]
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kFrag = 8u;
    constexpr uint kBM = 16u;
    constexpr uint kHiddenSuperTile = 256u;
    constexpr uint kW1BK = 48u;
    constexpr uint kW1BN = 128u;
    constexpr uint kW1GroupCount = 2u;
    constexpr uint kW1KFragmentCount = kW1BK / kFrag;
    constexpr uint kW1AStride = kW1BK + 1u;
    constexpr uint kW1AFloat4PerRow = kW1BK / 4u;
    constexpr uint kW1ColumnTiles = kW1BN / kFrag;
    constexpr uint kW1WeightFloat4PerRow = kW1BN / 4u;
    constexpr uint kW1WeightFloat4Count = kW1BK * kW1WeightFloat4PerRow;
    constexpr uint kW2BK = 64u;
    constexpr uint kW2BN = 64u;
    constexpr uint kW2OutputGroups = 2u;
    constexpr uint kW2OutputSuperTile = kW2BN * kW2OutputGroups;
    constexpr uint kW2KFragmentCount = kW2BK / kFrag;
    constexpr uint kW2ColumnTiles = kW2BN / kFrag;
    constexpr uint kW2WeightFloat4PerRow = kW2BN / 4u;
    constexpr uint kW2WeightFloat4Count = kW2BK * kW2WeightFloat4PerRow;
    constexpr uint kHColumnTiles = kHiddenSuperTile / kFrag;
    constexpr uint kTileElements = kFrag * kFrag;
    constexpr uint kArenaElements = 8192u;
    constexpr uint kW1WeightOffset = 1024u;
    constexpr uint kW2WeightOffset = 4096u;

    threadgroup float arena[kArenaElements];
    threadgroup float* A_tg = arena;
    threadgroup float* W1_tg = arena + kW1WeightOffset;
    threadgroup float* H_tg = arena;
    threadgroup float* W2_tg = arena + kW2WeightOffset;

    const uint tid = thread_pos.x;
    if (tid >= kThreads || simd_lane_count != kSimdWidth ||
        tg_pos.x != 0u || tg_pos.z != 0u || N == 0u || C == 0u) {
        return;
    }

    const uint row_base = tg_pos.y * kBM;
    if (row_base >= N) {
        return;
    }

    const float eps = as_type<float>(eps_bits);
    const uint stats_local_row0 = simd_id * 2u;
    const uint stats_local_row1 = stats_local_row0 + 1u;
    const uint stats_global_row0 = row_base + stats_local_row0;
    const uint stats_global_row1 = stats_global_row0 + 1u;
    const bool stats_row0_valid = stats_global_row0 < N;
    const bool stats_row1_valid = stats_global_row1 < N;

    float local_square_sum0 = 0.0f;
    float local_square_sum1 = 0.0f;

    if (stats_row0_valid) {
        const uint x_offset = stats_global_row0 * C;

        for (uint c = lane; c < C; c += kSimdWidth) {
            const float value = x[x_offset + c];
            local_square_sum0 = fma(
                value,
                value,
                local_square_sum0
            );
        }
    }

    if (stats_row1_valid) {
        const uint x_offset = stats_global_row1 * C;

        for (uint c = lane; c < C; c += kSimdWidth) {
            const float value = x[x_offset + c];
            local_square_sum1 = fma(
                value,
                value,
                local_square_sum1
            );
        }
    }

    const float inv_rms0 = stats_row0_valid
        ? rsqrt(
            simd_sum(local_square_sum0) / float(C) + eps
        )
        : 0.0f;

    const float inv_rms1 = stats_row1_valid
        ? rsqrt(
            simd_sum(local_square_sum1) / float(C) + eps
        )
        : 0.0f;

    const uint matrix_row_group = simd_id >> 2u;
    const uint matrix_col_group = simd_id & 3u;
    const uint simd_row_base = matrix_row_group * kFrag;
    const uint w1_simd_col_base = matrix_col_group * 32u;
    const uint w2_simd_col_base = matrix_col_group * 16u;
    const ushort2 matrix_offset = morton_order(ushort(lane));
    for (uint hidden_base = 0u; hidden_base < C4; hidden_base += kHiddenSuperTile) {
        const uint tile_hidden = min(kHiddenSuperTile, C4 - hidden_base);
        float2 w1_acc00 = float2(0.0f);
        float2 w1_acc01 = float2(0.0f);
        float2 w1_acc02 = float2(0.0f);
        float2 w1_acc03 = float2(0.0f);
        float2 w1_acc10 = float2(0.0f);
        float2 w1_acc11 = float2(0.0f);
        float2 w1_acc12 = float2(0.0f);
        float2 w1_acc13 = float2(0.0f);

        for (uint input_base = 0u; input_base < C; input_base += kW1BK) {
            #pragma unroll
            for (uint local_c4 = lane; local_c4 < kW1AFloat4PerRow; local_c4 += kSimdWidth) {
                const uint local_c = local_c4 * 4u;
                const uint global_c = input_base + local_c;

                float4 normalized0 = float4(0.0f);
                float4 normalized1 = float4(0.0f);

                if (stats_row0_valid && global_c + 3u < C) {
                    const float4 x_value = float4(
                        *reinterpret_cast<device const packed_float4*>(
                            x + stats_global_row0 * C + global_c
                        )
                    );

                    normalized0 = x_value * float4(inv_rms0);
                } else if (stats_row0_valid && global_c < C) {
                    #pragma unroll
                    for (uint component = 0u; component < 4u; ++component) {
                        const uint c = global_c + component;

                        if (c < C) {
                            normalized0[component] =
                                x[stats_global_row0 * C + c] * inv_rms0;
                        }
                    }
                }

                if (stats_row1_valid && global_c + 3u < C) {
                    const float4 x_value = float4(
                        *reinterpret_cast<device const packed_float4*>(
                            x + stats_global_row1 * C + global_c
                        )
                    );

                    normalized1 = x_value * float4(inv_rms1);
                } else if (stats_row1_valid && global_c < C) {
                    #pragma unroll
                    for (uint component = 0u; component < 4u; ++component) {
                        const uint c = global_c + component;

                        if (c < C) {
                            normalized1[component] =
                                x[stats_global_row1 * C + c] * inv_rms1;
                        }
                    }
                }

                *reinterpret_cast<threadgroup packed_float4*>(
                    A_tg +
                    stats_local_row0 * kW1AStride +
                    local_c
                ) = packed_float4(normalized0);

                *reinterpret_cast<threadgroup packed_float4*>(
                    A_tg +
                    stats_local_row1 * kW1AStride +
                    local_c
                ) = packed_float4(normalized1);
            }

            float2 a_fragment_cache[kW1KFragmentCount];
            #pragma unroll
            for (uint hidden_group = 0u; hidden_group < kW1GroupCount; ++hidden_group) {
                const uint hidden_group_base = hidden_base + hidden_group * kW1BN;
                for (uint index = tid; index < kW1WeightFloat4Count; index += kThreads) {
                    const uint local_k = index / kW1WeightFloat4PerRow;
                    const uint local_hidden4 =
                        index -
                        local_k *
                        kW1WeightFloat4PerRow;

                    const uint local_hidden =
                        local_hidden4 * 4u;

                    const uint global_c =
                        input_base + local_k;

                    const uint global_hidden =
                        hidden_group_base +
                        local_hidden;

                    float4 weight =
                        float4(0.0f);

                    if (global_c < C &&
                        global_hidden + 3u < C4) {

                        weight =
                            float4(
                                *reinterpret_cast<
                                    device const packed_float4*
                                >(
                                    w1 +
                                    global_c * C4 +
                                    global_hidden
                                )
                            );
                    } else if (
                        global_c < C &&
                        global_hidden < C4
                    ) {
                        #pragma unroll
                        for (uint component = 0u;
                             component < 4u;
                             ++component) {

                            const uint h =
                                global_hidden +
                                component;

                            if (h < C4) {
                                weight[component] =
                                    w1[
                                        global_c *
                                        C4 + h
                                    ];
                            }
                        }
                    }

                    const uint tile_row =
                        local_k / kFrag;

                    const uint tile_col =
                        local_hidden / kFrag;

                    const uint in_tile_row =
                        local_k &
                        (kFrag - 1u);

                    const uint in_tile_col =
                        local_hidden &
                        (kFrag - 1u);

                    const uint tile_index =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col
                        ) * kTileElements;

                    const uint physical_index =
                        tile_index +
                        in_tile_row * kFrag +
                        in_tile_col;

                    *reinterpret_cast<
                        threadgroup packed_float4*
                    >(
                        W1_tg +
                        physical_index
                    ) = packed_float4(weight);
                }

                threadgroup_barrier(
                    mem_flags::mem_threadgroup
                );

                if (hidden_group == 0u) {
                    #pragma unroll
                    for (uint k_index = 0u;
                         k_index < kW1KFragmentCount;
                         ++k_index) {

                        const uint k_sub =
                            k_index * kFrag;

                        a_fragment_cache[k_index] =
                            lowreg_matrix::load_tg_8x8(
                                A_tg,
                                kW1AStride,
                                ushort2(
                                    ushort(simd_row_base),
                                    ushort(k_sub)
                                ),
                                ushort(lane)
                            );
                    }
                }

                float2 accumulator0 =
                    hidden_group == 0u
                        ? w1_acc00
                        : w1_acc10;

                float2 accumulator1 =
                    hidden_group == 0u
                        ? w1_acc01
                        : w1_acc11;

                float2 accumulator2 =
                    hidden_group == 0u
                        ? w1_acc02
                        : w1_acc12;

                float2 accumulator3 =
                    hidden_group == 0u
                        ? w1_acc03
                        : w1_acc13;

                #pragma unroll
                for (uint k_index = 0u;
                     k_index < kW1KFragmentCount;
                     ++k_index) {

                    const uint tile_row =
                        k_index;

                    const uint tile_col0 =
                        w1_simd_col_base /
                        kFrag;

                    const uint tile_col1 =
                        tile_col0 + 1u;

                    const uint tile_col2 =
                        tile_col0 + 2u;

                    const uint tile_col3 =
                        tile_col0 + 3u;

                    const uint tile_index0 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col0
                        ) * kTileElements;

                    const uint tile_index1 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col1
                        ) * kTileElements;

                    const uint tile_index2 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col2
                        ) * kTileElements;

                    const uint tile_index3 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col3
                        ) * kTileElements;

                    const float2 w_fragment0 =
                        lowreg_matrix::load_tg_8x8(
                            W1_tg + tile_index0,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment1 =
                        lowreg_matrix::load_tg_8x8(
                            W1_tg + tile_index1,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment2 =
                        lowreg_matrix::load_tg_8x8(
                            W1_tg + tile_index2,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment3 =
                        lowreg_matrix::load_tg_8x8(
                            W1_tg + tile_index3,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 a_fragment =
                        a_fragment_cache[k_index];

                    accumulator0 =
                        lowreg_matrix::mma_8x8(
                            a_fragment,
                            w_fragment0,
                            accumulator0
                        );

                    accumulator1 =
                        lowreg_matrix::mma_8x8(
                            a_fragment,
                            w_fragment1,
                            accumulator1
                        );

                    accumulator2 =
                        lowreg_matrix::mma_8x8(
                            a_fragment,
                            w_fragment2,
                            accumulator2
                        );

                    accumulator3 =
                        lowreg_matrix::mma_8x8(
                            a_fragment,
                            w_fragment3,
                            accumulator3
                        );
                }

                if (hidden_group == 0u) {
                    w1_acc00 = accumulator0;
                    w1_acc01 = accumulator1;
                    w1_acc02 = accumulator2;
                    w1_acc03 = accumulator3;
                } else {
                    w1_acc10 = accumulator0;
                    w1_acc11 = accumulator1;
                    w1_acc12 = accumulator2;
                    w1_acc13 = accumulator3;
                }

                threadgroup_barrier(
                    mem_flags::mem_threadgroup
                );
            }
        }

        #pragma unroll
        for (uint hidden_group = 0u;
             hidden_group < kW1GroupCount;
             ++hidden_group) {

            #pragma unroll
            for (uint fragment_index = 0u;
                 fragment_index < 4u;
                 ++fragment_index) {

                float2 accumulator =
                    float2(0.0f);

                if (hidden_group == 0u) {
                    if (fragment_index == 0u) {
                        accumulator = w1_acc00;
                    } else if (fragment_index == 1u) {
                        accumulator = w1_acc01;
                    } else if (fragment_index == 2u) {
                        accumulator = w1_acc02;
                    } else {
                        accumulator = w1_acc03;
                    }
                } else {
                    if (fragment_index == 0u) {
                        accumulator = w1_acc10;
                    } else if (fragment_index == 1u) {
                        accumulator = w1_acc11;
                    } else if (fragment_index == 2u) {
                        accumulator = w1_acc12;
                    } else {
                        accumulator = w1_acc13;
                    }
                }

                const uint logical_hidden =
                    hidden_group * kW1BN +
                    w1_simd_col_base +
                    fragment_index * kFrag;

                const uint global_hidden =
                    hidden_base +
                    logical_hidden +
                    uint(matrix_offset.x);

                float2 biased_accumulator = accumulator;

                if (global_hidden < C4) {
                    biased_accumulator[0] += b1[global_hidden];
                }

                if (global_hidden + 1u < C4) {
                    biased_accumulator[1] += b1[global_hidden + 1u];
                }

                const float2 hidden_value =
                    float2(
                        fused_mlp_gelu_scalar(
                            biased_accumulator[0]
                        ),
                        fused_mlp_gelu_scalar(
                            biased_accumulator[1]
                        )
                    );

                const uint h_tile_row =
                    simd_row_base / kFrag;

                const uint h_tile_col =
                    logical_hidden / kFrag;

                const uint h_tile_index =
                    (
                        h_tile_row *
                        kHColumnTiles +
                        h_tile_col
                    ) * kTileElements;

                lowreg_matrix::store_tg_8x8(
                    H_tg + h_tile_index,
                    kFrag,
                    ushort2(0u, 0u),
                    ushort(lane),
                    hidden_value
                );
            }
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (uint output_super_base = 0u;
             output_super_base < C;
             output_super_base +=
                kW2OutputSuperTile) {

            float2 w2_acc00 =
                float2(0.0f);

            float2 w2_acc01 =
                float2(0.0f);

            float2 w2_acc10 =
                float2(0.0f);

            float2 w2_acc11 =
                float2(0.0f);

            for (uint hidden_k_base = 0u;
                 hidden_k_base < tile_hidden;
                 hidden_k_base += kW2BK) {

                float2 h_fragment_cache[
                    kW2KFragmentCount
                ];

                #pragma unroll
                for (uint k_index = 0u;
                     k_index < kW2KFragmentCount;
                     ++k_index) {

                    const uint logical_hidden =
                        hidden_k_base +
                        k_index * kFrag;

                    const uint h_tile_row =
                        simd_row_base / kFrag;

                    const uint h_tile_col =
                        logical_hidden / kFrag;

                    const uint h_tile_index =
                        (
                            h_tile_row *
                            kHColumnTiles +
                            h_tile_col
                        ) * kTileElements;

                    h_fragment_cache[k_index] =
                        lowreg_matrix::load_tg_8x8(
                            H_tg + h_tile_index,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );
                }

                #pragma unroll
                for (uint output_group = 0u;
                     output_group < kW2OutputGroups;
                     ++output_group) {

                    const uint output_base =
                        output_super_base +
                        output_group * kW2BN;

                    for (uint index = tid;
                         index < kW2WeightFloat4Count;
                         index += kThreads) {

                        const uint local_k =
                            index /
                            kW2WeightFloat4PerRow;

                        const uint local_output4 =
                            index -
                            local_k *
                            kW2WeightFloat4PerRow;

                        const uint local_output =
                            local_output4 * 4u;

                        const uint global_hidden =
                            hidden_base +
                            hidden_k_base +
                            local_k;

                        const uint global_output =
                            output_base +
                            local_output;

                        float4 weight =
                            float4(0.0f);

                        if (global_hidden < C4 &&
                            global_output + 3u < C) {

                            weight =
                                float4(
                                    *reinterpret_cast<
                                        device const packed_float4*
                                    >(
                                        w2 +
                                        global_hidden * C +
                                        global_output
                                    )
                                );
                        } else if (
                            global_hidden < C4 &&
                            global_output < C
                        ) {
                            #pragma unroll
                            for (uint component = 0u;
                                 component < 4u;
                                 ++component) {

                                const uint output_c =
                                    global_output +
                                    component;

                                if (output_c < C) {
                                    weight[component] =
                                        w2[
                                            global_hidden *
                                            C +
                                            output_c
                                        ];
                                }
                            }
                        }

                        const uint tile_row =
                            local_k / kFrag;

                        const uint tile_col =
                            local_output / kFrag;

                        const uint in_tile_row =
                            local_k &
                            (kFrag - 1u);

                        const uint in_tile_col =
                            local_output &
                            (kFrag - 1u);

                        const uint tile_index =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col
                            ) * kTileElements;

                        const uint physical_index =
                            tile_index +
                            in_tile_row * kFrag +
                            in_tile_col;

                        *reinterpret_cast<
                            threadgroup packed_float4*
                        >(
                            W2_tg +
                            physical_index
                        ) = packed_float4(weight);
                    }

                    threadgroup_barrier(
                        mem_flags::mem_threadgroup
                    );

                    float2 accumulator0 =
                        output_group == 0u
                            ? w2_acc00
                            : w2_acc10;

                    float2 accumulator1 =
                        output_group == 0u
                            ? w2_acc01
                            : w2_acc11;

                    #pragma unroll
                    for (uint k_index = 0u;
                         k_index < kW2KFragmentCount;
                         ++k_index) {

                        const uint tile_row =
                            k_index;

                        const uint tile_col0 =
                            w2_simd_col_base /
                            kFrag;

                        const uint tile_col1 =
                            tile_col0 + 1u;

                        const uint tile_index0 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col0
                            ) * kTileElements;

                        const uint tile_index1 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col1
                            ) * kTileElements;

                        const float2 w_fragment0 =
                            lowreg_matrix::load_tg_8x8(
                                W2_tg + tile_index0,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 w_fragment1 =
                            lowreg_matrix::load_tg_8x8(
                                W2_tg + tile_index1,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 h_fragment =
                            h_fragment_cache[k_index];

                        accumulator0 =
                            lowreg_matrix::mma_8x8(
                                h_fragment,
                                w_fragment0,
                                accumulator0
                            );

                        accumulator1 =
                            lowreg_matrix::mma_8x8(
                                h_fragment,
                                w_fragment1,
                                accumulator1
                            );
                    }

                    if (output_group == 0u) {
                        w2_acc00 = accumulator0;
                        w2_acc01 = accumulator1;
                    } else {
                        w2_acc10 = accumulator0;
                        w2_acc11 = accumulator1;
                    }

                    threadgroup_barrier(
                        mem_flags::mem_threadgroup
                    );
                }
            }

            const uint local_row =
                simd_row_base +
                uint(matrix_offset.y);

            const uint global_row =
                row_base + local_row;

            if (global_row < N) {
                #pragma unroll
                for (uint output_group = 0u;
                     output_group < kW2OutputGroups;
                     ++output_group) {

                    const uint output_base =
                        output_super_base +
                        output_group * kW2BN;

                    #pragma unroll
                    for (uint fragment_index = 0u;
                         fragment_index < 2u;
                         ++fragment_index) {

                        float2 accumulator =
                            float2(0.0f);

                        if (output_group == 0u) {
                            accumulator =
                                fragment_index == 0u
                                    ? w2_acc00
                                    : w2_acc01;
                        } else {
                            accumulator =
                                fragment_index == 0u
                                    ? w2_acc10
                                    : w2_acc11;
                        }

                        const uint fragment_col =
                            w2_simd_col_base +
                            fragment_index * kFrag +
                            uint(matrix_offset.x);

                        #pragma unroll
                        for (uint component = 0u;
                             component < 2u;
                             ++component) {

                            const uint global_output =
                                output_base +
                                fragment_col +
                                component;

                            if (global_output >= C) {
                                continue;
                            }

                            const uint output_index =
                                global_row * C +
                                global_output;

                            float result = accumulator[component];
                            if (hidden_base != 0u) {
                                result += out[output_index];
                            }
                            if (hidden_base + tile_hidden >= C4) {
                                result += b2[global_output];
                            }
                            out[output_index] = result;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fused_rmsnorm_mlp_forward_kernel_bm32(
    device const float* x        [[buffer(0)]],
    device const float* w1       [[buffer(1)]],
    device const float* b1       [[buffer(2)]],
    device const float* w2       [[buffer(3)]],
    device const float* b2       [[buffer(4)]],
    device float* out            [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& C             [[buffer(7)]],
    constant uint& C4            [[buffer(8)]],
    constant uint& eps_bits      [[buffer(9)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 thread_pos             [[thread_position_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count         [[threads_per_simdgroup]]
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kFrag = 8u;
    constexpr uint kBM = 32u;
    constexpr uint kHiddenSuperTile = 128u;

    constexpr uint kW1BK = 48u;
    constexpr uint kW1BN = 128u;
    constexpr uint kW1GroupCount = 1u;
    constexpr uint kW1KFragmentCount =
        kW1BK / kFrag;

    constexpr uint kW1AStride =
        kW1BK + 1u;

    constexpr uint kW1AFloat4PerRow =
        kW1BK / 4u;

    constexpr uint kW1ColumnTiles =
        kW1BN / kFrag;

    constexpr uint kW1WeightFloat4PerRow =
        kW1BN / 4u;

    constexpr uint kW1WeightFloat4Count =
        kW1BK *
        kW1WeightFloat4PerRow;

    constexpr uint kW2BK = 64u;
    constexpr uint kW2BN = 64u;
    constexpr uint kW2OutputGroups = 1u;

    constexpr uint kW2OutputSuperTile =
        kW2BN *
        kW2OutputGroups;

    constexpr uint kW2KFragmentCount =
        kW2BK / kFrag;

    constexpr uint kW2ColumnTiles =
        kW2BN / kFrag;

    constexpr uint kW2WeightFloat4PerRow =
        kW2BN / 4u;

    constexpr uint kW2WeightFloat4Count =
        kW2BK *
        kW2WeightFloat4PerRow;

    constexpr uint kHColumnTiles =
        kHiddenSuperTile /
        kFrag;

    constexpr uint kTileElements =
        kFrag * kFrag;

    constexpr uint kArenaElements =
        8192u;

    constexpr uint kW1WeightOffset =
        2048u;

    constexpr uint kW2WeightOffset =
        4096u;

    threadgroup float arena[
        kArenaElements
    ];

    threadgroup float* A_tg =
        arena;

    threadgroup float* W1_tg =
        arena +
        kW1WeightOffset;

    threadgroup float* H_tg =
        arena;

    threadgroup float* W2_tg =
        arena +
        kW2WeightOffset;

    const uint tid =
        thread_pos.x;

    if (tid >= kThreads ||
        simd_lane_count != kSimdWidth ||
        tg_pos.x != 0u ||
        tg_pos.z != 0u ||
        N == 0u ||
        C == 0u) {
        return;
    }

    const uint row_base =
        tg_pos.y * kBM;

    if (row_base >= N) {
        return;
    }

    const float eps =
        as_type<float>(
            eps_bits
        );

    uint stats_local_rows[4] = {
        simd_id * 2u,
        simd_id * 2u + 1u,
        simd_id * 2u + 16u,
        simd_id * 2u + 17u
    };

    uint stats_global_rows[4];
    bool stats_rows_valid[4];
    float inv_rms[4];

    #pragma unroll
    for (uint stats_index = 0u; stats_index < 4u; ++stats_index) {
        const uint global_row =
            row_base + stats_local_rows[stats_index];

        stats_global_rows[stats_index] = global_row;

        const bool valid = global_row < N;
        stats_rows_valid[stats_index] = valid;

        float local_square_sum = 0.0f;

        if (valid) {
            const uint x_offset = global_row * C;

            for (uint c = lane; c < C; c += kSimdWidth) {
                const float value = x[x_offset + c];
                local_square_sum = fma(
                    value,
                    value,
                    local_square_sum
                );
            }
        }

        inv_rms[stats_index] = valid
            ? rsqrt(
                simd_sum(local_square_sum) / float(C) + eps
            )
            : 0.0f;
    }

    const uint matrix_row_group = simd_id >> 1u;
    const uint matrix_col_group = simd_id & 1u;
    const uint simd_row_base = matrix_row_group * kFrag;
    const uint w1_simd_col_base = matrix_col_group * 64u;
    const uint w2_simd_col_base = matrix_col_group * 32u;
    const ushort2 matrix_offset = morton_order(ushort(lane));
    for (uint hidden_base = 0u; hidden_base < C4; hidden_base += kHiddenSuperTile) {
        const uint tile_hidden = min(kHiddenSuperTile, C4 - hidden_base);

        float2 w1_acc00 = float2(0.0f);
        float2 w1_acc01 = float2(0.0f);
        float2 w1_acc02 = float2(0.0f);
        float2 w1_acc03 = float2(0.0f);
        float2 w1_acc10 = float2(0.0f);
        float2 w1_acc11 = float2(0.0f);
        float2 w1_acc12 = float2(0.0f);
        float2 w1_acc13 = float2(0.0f);

        for (uint input_base = 0u; input_base < C; input_base += kW1BK) {
            #pragma unroll
            for (uint local_c4 = lane; local_c4 < kW1AFloat4PerRow; local_c4 += kSimdWidth) {
                const uint local_c = local_c4 * 4u;
                const uint global_c = input_base + local_c;
                #pragma unroll
                for (uint stats_index = 0u; stats_index < 4u; ++stats_index) {
                    float4 normalized = float4(0.0f);

                    if (
                        stats_rows_valid[stats_index] &&
                        global_c + 3u < C
                    ) {
                        const float4 x_value = float4(
                            *reinterpret_cast<device const packed_float4*>(
                                x +
                                stats_global_rows[stats_index] * C +
                                global_c
                            )
                        );

                        normalized =
                            x_value * float4(inv_rms[stats_index]);
                    } else if (
                        stats_rows_valid[stats_index] &&
                        global_c < C
                    ) {
                        #pragma unroll
                        for (uint component = 0u; component < 4u; ++component) {
                            const uint c = global_c + component;

                            if (c < C) {
                                normalized[component] =
                                    x[
                                        stats_global_rows[stats_index] * C + c
                                    ] * inv_rms[stats_index];
                            }
                        }
                    }

                    *reinterpret_cast<threadgroup packed_float4*>(
                        A_tg +
                        stats_local_rows[stats_index] *
                            kW1AStride +
                        local_c
                    ) = packed_float4(normalized);
                }
            }

            float2 a_fragment_cache[
                kW1KFragmentCount
            ];

            #pragma unroll
            for (uint hidden_group = 0u;
                 hidden_group <
                    kW1GroupCount;
                 ++hidden_group) {

                const uint hidden_group_base =
                    hidden_base +
                    hidden_group *
                    kW1BN;

                for (uint index = tid;
                     index <
                        kW1WeightFloat4Count;
                     index +=
                        kThreads) {

                    const uint local_k =
                        index /
                        kW1WeightFloat4PerRow;

                    const uint local_hidden4 =
                        index -
                        local_k *
                        kW1WeightFloat4PerRow;

                    const uint local_hidden =
                        local_hidden4 *
                        4u;

                    const uint global_c =
                        input_base +
                        local_k;

                    const uint global_hidden =
                        hidden_group_base +
                        local_hidden;

                    float4 weight =
                        float4(0.0f);

                    if (
                        global_c < C &&
                        global_hidden + 3u <
                            C4
                    ) {
                        weight =
                            float4(
                                *reinterpret_cast<
                                    device const
                                    packed_float4*
                                >(
                                    w1 +
                                    global_c *
                                    C4 +
                                    global_hidden
                                )
                            );
                    } else if (
                        global_c < C &&
                        global_hidden < C4
                    ) {
                        #pragma unroll
                        for (
                            uint component = 0u;
                            component < 4u;
                            ++component
                        ) {
                            const uint h =
                                global_hidden +
                                component;

                            if (h < C4) {
                                weight[
                                    component
                                ] =
                                    w1[
                                        global_c *
                                        C4 +
                                        h
                                    ];
                            }
                        }
                    }

                    const uint tile_row =
                        local_k /
                        kFrag;

                    const uint tile_col =
                        local_hidden /
                        kFrag;

                    const uint in_tile_row =
                        local_k &
                        (kFrag - 1u);

                    const uint in_tile_col =
                        local_hidden &
                        (kFrag - 1u);

                    const uint tile_index =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col
                        ) *
                        kTileElements;

                    const uint physical_index =
                        tile_index +
                        in_tile_row *
                        kFrag +
                        in_tile_col;

                    *reinterpret_cast<
                        threadgroup
                        packed_float4*
                    >(
                        W1_tg +
                        physical_index
                    ) =
                        packed_float4(
                            weight
                        );
                }

                threadgroup_barrier(
                    mem_flags::
                    mem_threadgroup
                );

                if (hidden_group == 0u) {
                    #pragma unroll
                    for (
                        uint k_index = 0u;
                        k_index <
                            kW1KFragmentCount;
                        ++k_index
                    ) {
                        const uint k_sub =
                            k_index *
                            kFrag;

                        a_fragment_cache[
                            k_index
                        ] =
                            lowreg_matrix::
                            load_tg_8x8(
                                A_tg,
                                kW1AStride,
                                ushort2(
                                    ushort(
                                        simd_row_base
                                    ),
                                    ushort(k_sub)
                                ),
                                ushort(lane)
                            );
                    }
                }

                float2 accumulator0 =
                    w1_acc00;

                float2 accumulator1 =
                    w1_acc01;

                float2 accumulator2 =
                    w1_acc02;

                float2 accumulator3 =
                    w1_acc03;

                float2 accumulator4 =
                    w1_acc10;

                float2 accumulator5 =
                    w1_acc11;

                float2 accumulator6 =
                    w1_acc12;

                float2 accumulator7 =
                    w1_acc13;

                #pragma unroll
                for (uint k_index = 0u;
                     k_index <
                        kW1KFragmentCount;
                     ++k_index) {

                    const uint tile_row =
                        k_index;

                    const uint tile_col0 =
                        w1_simd_col_base /
                        kFrag;

                    const uint tile_col1 =
                        tile_col0 + 1u;

                    const uint tile_col2 =
                        tile_col0 + 2u;

                    const uint tile_col3 =
                        tile_col0 + 3u;

                    const uint tile_col4 =
                        tile_col0 + 4u;

                    const uint tile_col5 =
                        tile_col0 + 5u;

                    const uint tile_col6 =
                        tile_col0 + 6u;

                    const uint tile_col7 =
                        tile_col0 + 7u;

                    const uint tile_index0 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col0
                        ) *
                        kTileElements;

                    const uint tile_index1 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col1
                        ) *
                        kTileElements;

                    const uint tile_index2 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col2
                        ) *
                        kTileElements;

                    const uint tile_index3 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col3
                        ) *
                        kTileElements;

                    const uint tile_index4 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col4
                        ) *
                        kTileElements;

                    const uint tile_index5 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col5
                        ) *
                        kTileElements;

                    const uint tile_index6 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col6
                        ) *
                        kTileElements;

                    const uint tile_index7 =
                        (
                            tile_row *
                            kW1ColumnTiles +
                            tile_col7
                        ) *
                        kTileElements;

                    const float2 w_fragment0 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index0,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment1 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index1,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment2 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index2,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment3 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index3,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment4 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index4,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment5 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index5,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment6 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index6,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 w_fragment7 =
                        lowreg_matrix::
                        load_tg_8x8(
                            W1_tg +
                            tile_index7,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );

                    const float2 a_fragment =
                        a_fragment_cache[
                            k_index
                        ];

                    accumulator0 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment0,
                            accumulator0
                        );

                    accumulator1 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment1,
                            accumulator1
                        );

                    accumulator2 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment2,
                            accumulator2
                        );

                    accumulator3 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment3,
                            accumulator3
                        );

                    accumulator4 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment4,
                            accumulator4
                        );

                    accumulator5 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment5,
                            accumulator5
                        );

                    accumulator6 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment6,
                            accumulator6
                        );

                    accumulator7 =
                        lowreg_matrix::
                        mma_8x8(
                            a_fragment,
                            w_fragment7,
                            accumulator7
                        );
                }

                w1_acc00 =
                    accumulator0;

                w1_acc01 =
                    accumulator1;

                w1_acc02 =
                    accumulator2;

                w1_acc03 =
                    accumulator3;

                w1_acc10 =
                    accumulator4;

                w1_acc11 =
                    accumulator5;

                w1_acc12 =
                    accumulator6;

                w1_acc13 =
                    accumulator7;

                threadgroup_barrier(
                    mem_flags::
                    mem_threadgroup
                );
            }
        }

        #pragma unroll
        for (uint hidden_group = 0u;
             hidden_group <
                kW1GroupCount;
             ++hidden_group) {

            #pragma unroll
            for (uint fragment_index = 0u;
                 fragment_index < 8u;
                 ++fragment_index) {

                float2 accumulator =
                    float2(0.0f);

                if (fragment_index == 0u) {
                    accumulator =
                        w1_acc00;
                } else if (
                    fragment_index == 1u
                ) {
                    accumulator =
                        w1_acc01;
                } else if (
                    fragment_index == 2u
                ) {
                    accumulator =
                        w1_acc02;
                } else if (
                    fragment_index == 3u
                ) {
                    accumulator =
                        w1_acc03;
                } else if (
                    fragment_index == 4u
                ) {
                    accumulator =
                        w1_acc10;
                } else if (
                    fragment_index == 5u
                ) {
                    accumulator =
                        w1_acc11;
                } else if (
                    fragment_index == 6u
                ) {
                    accumulator =
                        w1_acc12;
                } else {
                    accumulator =
                        w1_acc13;
                }


                const uint logical_hidden =
                    hidden_group *
                    kW1BN +
                    w1_simd_col_base +
                    fragment_index *
                    kFrag;

                const uint global_hidden =
                    hidden_base +
                    logical_hidden +
                    uint(matrix_offset.x);

                float2 biased_accumulator = accumulator;

                if (global_hidden < C4) {
                    biased_accumulator[0] += b1[global_hidden];
                }

                if (global_hidden + 1u < C4) {
                    biased_accumulator[1] += b1[global_hidden + 1u];
                }

                const float2 hidden_value = float2(
                    fused_mlp_gelu_scalar(biased_accumulator[0]),
                    fused_mlp_gelu_scalar(biased_accumulator[1])
                );

                const uint h_tile_row =
                    simd_row_base /
                    kFrag;

                const uint h_tile_col =
                    logical_hidden /
                    kFrag;

                const uint h_tile_index =
                    (
                        h_tile_row *
                        kHColumnTiles +
                        h_tile_col
                    ) *
                    kTileElements;

                lowreg_matrix::
                store_tg_8x8(
                    H_tg +
                    h_tile_index,
                    kFrag,
                    ushort2(0u, 0u),
                    ushort(lane),
                    hidden_value
                );
            }
        }

        threadgroup_barrier(
            mem_flags::
            mem_threadgroup
        );

        for (
            uint output_super_base = 0u;
            output_super_base < C;
            output_super_base +=
                kW2OutputSuperTile
        ) {
            float2 w2_acc00 =
                float2(0.0f);

            float2 w2_acc01 =
                float2(0.0f);

            float2 w2_acc10 =
                float2(0.0f);

            float2 w2_acc11 =
                float2(0.0f);

            for (
                uint hidden_k_base = 0u;
                hidden_k_base <
                    tile_hidden;
                hidden_k_base +=
                    kW2BK
            ) {
                float2 h_fragment_cache[
                    kW2KFragmentCount
                ];

                #pragma unroll
                for (uint k_index = 0u;
                     k_index <
                        kW2KFragmentCount;
                     ++k_index) {

                    const uint logical_hidden =
                        hidden_k_base +
                        k_index *
                        kFrag;

                    const uint h_tile_row =
                        simd_row_base /
                        kFrag;

                    const uint h_tile_col =
                        logical_hidden /
                        kFrag;

                    const uint h_tile_index =
                        (
                            h_tile_row *
                            kHColumnTiles +
                            h_tile_col
                        ) *
                        kTileElements;

                    h_fragment_cache[
                        k_index
                    ] =
                        lowreg_matrix::
                        load_tg_8x8(
                            H_tg +
                            h_tile_index,
                            kFrag,
                            ushort2(0u, 0u),
                            ushort(lane)
                        );
                }

                #pragma unroll
                for (uint output_group = 0u;
                     output_group <
                        kW2OutputGroups;
                     ++output_group) {

                    const uint output_base =
                        output_super_base +
                        output_group *
                        kW2BN;

                    for (uint index = tid;
                         index <
                            kW2WeightFloat4Count;
                         index +=
                            kThreads) {

                        const uint local_k =
                            index /
                            kW2WeightFloat4PerRow;

                        const uint local_output4 =
                            index -
                            local_k *
                            kW2WeightFloat4PerRow;

                        const uint local_output =
                            local_output4 *
                            4u;

                        const uint global_hidden =
                            hidden_base +
                            hidden_k_base +
                            local_k;

                        const uint global_output =
                            output_base +
                            local_output;

                        float4 weight =
                            float4(0.0f);

                        if (
                            global_hidden < C4 &&
                            global_output + 3u <
                                C
                        ) {
                            weight =
                                float4(
                                    *reinterpret_cast<
                                        device const
                                        packed_float4*
                                    >(
                                        w2 +
                                        global_hidden *
                                        C +
                                        global_output
                                    )
                                );
                        } else if (
                            global_hidden < C4 &&
                            global_output < C
                        ) {
                            #pragma unroll
                            for (
                                uint component = 0u;
                                component < 4u;
                                ++component
                            ) {
                                const uint output_c =
                                    global_output +
                                    component;

                                if (output_c < C) {
                                    weight[
                                        component
                                    ] =
                                        w2[
                                            global_hidden *
                                            C +
                                            output_c
                                        ];
                                }
                            }
                        }

                        const uint tile_row =
                            local_k /
                            kFrag;

                        const uint tile_col =
                            local_output /
                            kFrag;

                        const uint in_tile_row =
                            local_k &
                            (kFrag - 1u);

                        const uint in_tile_col =
                            local_output &
                            (kFrag - 1u);

                        const uint tile_index =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col
                            ) *
                            kTileElements;

                        const uint physical_index =
                            tile_index +
                            in_tile_row *
                            kFrag +
                            in_tile_col;

                        *reinterpret_cast<
                            threadgroup
                            packed_float4*
                        >(
                            W2_tg +
                            physical_index
                        ) =
                            packed_float4(
                                weight
                            );
                    }

                    threadgroup_barrier(
                        mem_flags::
                        mem_threadgroup
                    );

                    float2 accumulator0 =
                        w2_acc00;

                    float2 accumulator1 =
                        w2_acc01;

                    float2 accumulator2 =
                        w2_acc10;

                    float2 accumulator3 =
                        w2_acc11;

                    #pragma unroll
                    for (uint k_index = 0u;
                         k_index <
                            kW2KFragmentCount;
                         ++k_index) {

                        const uint tile_row =
                            k_index;

                        const uint tile_col0 =
                            w2_simd_col_base /
                            kFrag;

                        const uint tile_col1 =
                            tile_col0 + 1u;

                        const uint tile_col2 =
                            tile_col0 + 2u;

                        const uint tile_col3 =
                            tile_col0 + 3u;

                        const uint tile_index0 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col0
                            ) *
                            kTileElements;

                        const uint tile_index1 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col1
                            ) *
                            kTileElements;

                        const uint tile_index2 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col2
                            ) *
                            kTileElements;

                        const uint tile_index3 =
                            (
                                tile_row *
                                kW2ColumnTiles +
                                tile_col3
                            ) *
                            kTileElements;

                        const float2 w_fragment0 =
                            lowreg_matrix::
                            load_tg_8x8(
                                W2_tg +
                                tile_index0,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 w_fragment1 =
                            lowreg_matrix::
                            load_tg_8x8(
                                W2_tg +
                                tile_index1,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 w_fragment2 =
                            lowreg_matrix::
                            load_tg_8x8(
                                W2_tg +
                                tile_index2,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 w_fragment3 =
                            lowreg_matrix::
                            load_tg_8x8(
                                W2_tg +
                                tile_index3,
                                kFrag,
                                ushort2(0u, 0u),
                                ushort(lane)
                            );

                        const float2 h_fragment =
                            h_fragment_cache[
                                k_index
                            ];

                        accumulator0 =
                            lowreg_matrix::
                            mma_8x8(
                                h_fragment,
                                w_fragment0,
                                accumulator0
                            );

                        accumulator1 =
                            lowreg_matrix::
                            mma_8x8(
                                h_fragment,
                                w_fragment1,
                                accumulator1
                            );

                        accumulator2 =
                            lowreg_matrix::
                            mma_8x8(
                                h_fragment,
                                w_fragment2,
                                accumulator2
                            );

                        accumulator3 =
                            lowreg_matrix::
                            mma_8x8(
                                h_fragment,
                                w_fragment3,
                                accumulator3
                            );
                    }

                    w2_acc00 =
                        accumulator0;

                    w2_acc01 =
                        accumulator1;

                    w2_acc10 =
                        accumulator2;

                    w2_acc11 =
                        accumulator3;

                    threadgroup_barrier(
                        mem_flags::
                        mem_threadgroup
                    );
                }
            }

            const uint local_row =
                simd_row_base +
                uint(
                    matrix_offset.y
                );

            const uint global_row =
                row_base +
                local_row;

            if (global_row < N) {
                #pragma unroll
                for (
                    uint output_group = 0u;
                    output_group <
                        kW2OutputGroups;
                    ++output_group
                ) {
                    const uint output_base =
                        output_super_base +
                        output_group *
                        kW2BN;

                    #pragma unroll
                    for (
                        uint fragment_index = 0u;
                        fragment_index < 4u;
                        ++fragment_index
                    ) {
                        float2 accumulator =
                            float2(0.0f);

                        if (
                            fragment_index == 0u
                        ) {
                            accumulator =
                                w2_acc00;
                        } else if (
                            fragment_index == 1u
                        ) {
                            accumulator =
                                w2_acc01;
                        } else if (
                            fragment_index == 2u
                        ) {
                            accumulator =
                                w2_acc10;
                        } else {
                            accumulator =
                                w2_acc11;
                        }

                        const uint fragment_col =
                            w2_simd_col_base +
                            fragment_index *
                            kFrag +
                            uint(
                                matrix_offset.x
                            );

                        const uint pair_output =
                            output_base +
                            fragment_col;

                        if (
                            pair_output + 1u <
                            C
                        ) {
                            const uint output_index =
                                global_row *
                                C +
                                pair_output;

                            float2 result = accumulator;

                            if (hidden_base != 0u) {
                                result += float2(
                                    *reinterpret_cast<device const packed_float2*>(
                                        out + output_index
                                    )
                                );
                            }

                            if (hidden_base + tile_hidden >= C4) {
                                result += float2(
                                    *reinterpret_cast<device const packed_float2*>(
                                        b2 + pair_output
                                    )
                                );
                            }

                            *reinterpret_cast<
                                device
                                packed_float2*
                            >(
                                out +
                                output_index
                            ) =
                                packed_float2(
                                    result
                                );

                            continue;
                        }

                        #pragma unroll
                        for (
                            uint component = 0u;
                            component < 2u;
                            ++component
                        ) {
                            const uint global_output =
                                output_base +
                                fragment_col +
                                component;

                            if (
                                global_output >= C
                            ) {
                                continue;
                            }

                            const uint output_index =
                                global_row *
                                C +
                                global_output;

                            float result = accumulator[component];

                            if (hidden_base != 0u) {
                                result += out[output_index];
                            }

                            if (hidden_base + tile_hidden >= C4) {
                                result += b2[global_output];
                            }

                            out[output_index] = result;
                        }
                    }
                }
            }
        }

        threadgroup_barrier(
            mem_flags::
            mem_threadgroup
        );
    }
}

kernel void rmsnorm_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& C [[buffer(2)]],
    constant uint& eps_bits [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kMaxSimdGroups = kThreads / kSimdWidth;

    if (tid >= kThreads || C == 0u) {
        return;
    }

    const uint row = tg_id;
    const uint base = row * C;
    const float eps = as_type<float>(eps_bits);

    threadgroup float reduce_buf[kMaxSimdGroups];

    const uint vector_count = C / 4u;
    const uint vector_end = vector_count * 4u;

    device const packed_float4* x4 =
        reinterpret_cast<device const packed_float4*>(x + base);

    float local_square_sum = 0.0f;

    for (
        uint vector_index = tid;
        vector_index < vector_count;
        vector_index += tpg
    ) {
        const float4 value = float4(x4[vector_index]);
        local_square_sum = fma(
            value.x,
            value.x,
            local_square_sum
        );
        local_square_sum = fma(
            value.y,
            value.y,
            local_square_sum
        );
        local_square_sum = fma(
            value.z,
            value.z,
            local_square_sum
        );
        local_square_sum = fma(
            value.w,
            value.w,
            local_square_sum
        );
    }

    for (
        uint c = vector_end + tid;
        c < C;
        c += tpg
    ) {
        const float value = x[base + c];
        local_square_sum = fma(
            value,
            value,
            local_square_sum
        );
    }

    const float simd_square_sum =
        simd_sum(local_square_sum);

    if (lane == 0u) {
        reduce_buf[simd_id] = simd_square_sum;
    }

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );

    if (tid == 0u) {
        float total_square_sum = 0.0f;
        const uint simd_group_count =
            (tpg + kSimdWidth - 1u) / kSimdWidth;

        for (
            uint group = 0u;
            group < simd_group_count;
            ++group
        ) {
            total_square_sum += reduce_buf[group];
        }

        reduce_buf[0] = rsqrt(
            total_square_sum / float(C) + eps
        );
    }

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );

    const float inv_rms = reduce_buf[0];

    device packed_float4* out4 =
        reinterpret_cast<device packed_float4*>(out + base);

    for (
        uint vector_index = tid;
        vector_index < vector_count;
        vector_index += tpg
    ) {
        const float4 value = float4(x4[vector_index]);
        out4[vector_index] = packed_float4(
            value * inv_rms
        );
    }

    for (
        uint c = vector_end + tid;
        c < C;
        c += tpg
    ) {
        out[base + c] =
            x[base + c] * inv_rms;
    }
}

kernel void rmsnorm_weighted_forward(
    device const float* x       [[buffer(0)]],
    device const float* gamma   [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint& C            [[buffer(3)]],
    constant uint& eps_bits     [[buffer(4)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint3 thread_pos            [[thread_position_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]]
) {
    const uint tid = thread_pos.x;
    constexpr uint kSimdWidth = 32u;

    // 只保留第一组执行，消除重复写入。
    if (tid >= kSimdWidth || C == 0u) {
        return;
    }

    const uint row = tg_pos.x;
    const uint base = row * C;

    float local_sum = 0.0f;

    for (uint c = lane; c < C; c += kSimdWidth) {
        const float value = x[base + c];
        local_sum = fma(value, value, local_sum);
    }

    const float sum = simd_sum(local_sum);
    const float mean = sum / float(C);
    const float eps = as_type<float>(eps_bits);
    const float inv_rms = rsqrt(mean + eps);

    for (uint c = lane; c < C; c += kSimdWidth) {
        out[base + c] =
            x[base + c] * inv_rms * gamma[c];
    }
}

kernel void layernorm_forward(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& eps_bits [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    const uint row = tg_id;
    const uint offset = row * C;
    const float eps = as_type<float>(eps_bits);      // ← 新增：从 bits 还原 eps
    threadgroup float2 reduce_buf[256];  // 复用，存储 mean 和 inv_std

    // ------------------------------------------------------------
    // 第一遍：计算均值 mean = sum(x) / C
    // ------------------------------------------------------------
    float sum_x = 0.0f;
    uint i = tid * 4;
    for (; i + 4 <= C; i += tpg * 4) {
        float4 v = float4(x[offset + i], x[offset + i + 1], x[offset + i + 2], x[offset + i + 3]);
        sum_x += v.x + v.y + v.z + v.w;
    }
    // 处理剩余元素（非 4 倍数部分）
    uint rem = C - (C / 4) * 4;
    uint base = C - rem;
    for (uint j = tid; j < rem; j += tpg) {
        sum_x += x[offset + base + j];
    }

    // 线程组内归约 sum_x（使用 SIMD shuffle 和 threadgroup 共享）
    sum_x = simd_sum(sum_x);
    if (tid % 32 == 0) {
        reduce_buf[tid / 32].x = sum_x;   // 每个 SIMD 组的总和存到 .x
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 汇总所有 SIMD 组的和，并计算均值
    if (tid == 0) {
        float total = 0.0f;
        uint simd_groups = (tpg + 31) / 32;
        for (uint s = 0; s < simd_groups; ++s) {
            total += reduce_buf[s].x;
        }
        float mean = total / (float)C;
        reduce_buf[0].x = mean;   // 存储均值
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = reduce_buf[0].x;

    // ------------------------------------------------------------
    // 第二遍：计算方差 var = sum((x - mean)^2) / C  （两遍算法，数值稳定）
    // ------------------------------------------------------------
    float sum_sq_diff = 0.0f;
    i = tid * 4;
    for (; i + 4 <= C; i += tpg * 4) {
        float4 v = float4(x[offset + i], x[offset + i + 1], x[offset + i + 2], x[offset + i + 3]);
        float4 diff = v - float4(mean);
        sum_sq_diff += dot(diff, diff);
    }
    rem = C - (C / 4) * 4;
    base = C - rem;
    for (uint j = tid; j < rem; j += tpg) {
        float v = x[offset + base + j];
        float diff = v - mean;
        sum_sq_diff += diff * diff;
    }

    sum_sq_diff = simd_sum(sum_sq_diff);
    if (tid % 32 == 0) {
        reduce_buf[tid / 32].x = sum_sq_diff;   // 复用 .x
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total_sq = 0.0f;
        uint simd_groups = (tpg + 31) / 32;
        for (uint s = 0; s < simd_groups; ++s) {
            total_sq += reduce_buf[s].x;
        }
        float var = total_sq / (float)C;
        float inv_std = rsqrt(var + eps);
        reduce_buf[0].y = inv_std;   // 存储 inv_std
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = reduce_buf[0].y;

    // ------------------------------------------------------------
    // 第三遍：归一化输出
    // ------------------------------------------------------------
    i = tid * 4;
    for (; i + 4 <= C; i += tpg * 4) {
        float4 v = float4(x[offset + i], x[offset + i + 1], x[offset + i + 2], x[offset + i + 3]);
        float4 w = float4(weight[i], weight[i + 1], weight[i + 2], weight[i + 3]);
        float4 b = float4(bias[i], bias[i + 1], bias[i + 2], bias[i + 3]);
        float4 result = (v - float4(mean)) * float4(inv_std) * w + b;
        out[offset + i]     = result.x;
        out[offset + i + 1] = result.y;
        out[offset + i + 2] = result.z;
        out[offset + i + 3] = result.w;
    }
    rem = C - (C / 4) * 4;
    base = C - rem;
    for (uint j = tid; j < rem; j += tpg) {
        uint idx = base + j;
        float v = x[offset + idx];
        out[offset + idx] = (v - mean) * inv_std * weight[idx] + bias[idx];
    }
}

kernel void layernorm_backward_kernel(
    device const float* x      [[buffer(0)]],
    device const float* dy     [[buffer(1)]],
    device const float* gamma  [[buffer(2)]],
    device float* dx           [[buffer(3)]],
    device atomic_uint* dgamma [[buffer(4)]],
    device atomic_uint* dbeta  [[buffer(5)]],
    constant uint& B_T         [[buffer(6)]],
    constant uint& C           [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], 
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (row >= B_T) return;
    uint base = row * C;
    threadgroup float buf1[1024];
    threadgroup float buf2[1024];

    float local_sum = 0.0f;
    for (uint i = tid; i < C; i += tpg) {
        local_sum += x[base + i];
    }
    buf1[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) buf1[tid] += buf1[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = buf1[0] / (float)C;

    float local_var = 0.0f;
    for (uint i = tid; i < C; i += tpg) {
        float diff = x[base + i] - mean;
        local_var += diff * diff;
    }
    buf1[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) buf1[tid] += buf1[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rsqrt_var = rsqrt((buf1[0] / (float)C) + 1e-5f);

    float local_dy = 0.0f;
    float local_dy_xhat = 0.0f;
    for (uint i = tid; i < C; i += tpg) {
        float xhat = (x[base + i] - mean) * rsqrt_var;
        float dy_i = dy[base + i];
        float dyg = dy_i * gamma[i];
        
        local_dy += dyg;
        local_dy_xhat += dyg * xhat;
    }
    buf1[tid] = local_dy;
    buf2[tid] = local_dy_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buf1[tid] += buf1[tid + s];
            buf2[tid] += buf2[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_dy = buf1[0] / (float)C;
    float mean_dy_xhat = buf2[0] / (float)C;

    for (uint i = tid; i < C; i += tpg) {
        float xhat = (x[base + i] - mean) * rsqrt_var;
        float dy_i = dy[base + i];
        if (dx) {
            float dyg = dy_i * gamma[i];
            dx[base + i] = rsqrt_var * (dyg - mean_dy - xhat * mean_dy_xhat);
        }
        
        if (dgamma) {
            atomic_add_float(&dgamma[i], dy_i * xhat);
        }
        if (dbeta) {
            atomic_add_float(&dbeta[i], dy_i);
        }
    }
}

kernel void gelu_add_bias_kernel(
    device const float* h1 [[buffer(0)]],
    device const float* b1 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    constant uint& C4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint c = gid % C4;
    float val = h1[gid] + b1[c];
    float x_cube = val * val * val;
    float inner = clamp(sqrt(2.0f/M_PI_F)*(val + 0.044715f*x_cube), -20.0f, 20.0f);
    out[gid] = 0.5f * val * (1.0f + fast::tanh(inner));
}

kernel void gelu_backward_bias_kernel(
    device const float* grad_out [[buffer(0)]],
    device const float* h1 [[buffer(1)]],
    device const float* b1 [[buffer(2)]],
    device float* grad_in [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    constant uint& C4 [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint c = gid % C4;
    float val = h1[gid] + b1[c];
    float x_sq = val * val;
    float x_cube = x_sq * val;
    float inner = sqrt(2.0f/M_PI_F)*(val + 0.044715f*x_cube);
    float tanh_inner = fast::tanh(clamp(inner, -20.0f, 20.0f));
    
    float left = 0.5f * (1.0f + tanh_inner);
    float right = 0.5f * val * (1.0f - tanh_inner * tanh_inner) * sqrt(2.0f/M_PI_F) * (1.0f + 3.0f * 0.044715f * x_sq);
    grad_in[gid] = grad_out[gid] * (left + right);
}

kernel void gelu_forward_kernel(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    if (id >= threads_per_grid) return;
    const float val = x[id];
    const float x_cube = val * val * val;
    const float inner = clamp(sqrt(2.0f / M_PI_F) * (val + 0.044715f * x_cube), -20.0f, 20.0f);
    out[id] = 0.5f * val * (1.0f + fast::tanh(inner));
}

kernel void gelu_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    if (id >= threads_per_grid) return;
    const float val = x[id];
    const float x_sq = val * val;
    const float x_cube = x_sq * val;
    const float inner = sqrt(2.0f / M_PI_F) * (val + 0.044715f * x_cube);
    const float tanh_inner = fast::tanh(clamp(inner, -20.0f, 20.0f));
    const float left = 0.5f * (1.0f + tanh_inner);
    const float right = 0.5f * val * (1.0f - tanh_inner * tanh_inner)
        * sqrt(2.0f / M_PI_F) * (1.0f + 3.0f * 0.044715f * x_sq);
    // 与 CPU cpu_gelu_backward 一致：累加
    grad_x[id] = grad_out[id] * (left + right);
}

kernel void swiglu_forward_kernel(
    device const float* gu [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint VEC = 4u;
    const uint vectors_per_row = (H + VEC - 1u) / VEC;
    const uint row = gid / vectors_per_row;
    const uint vec_id = gid % vectors_per_row;
    const uint col = vec_id * VEC;

    if (row >= N || col >= H) {
        return;
    }

    const uint gu_base = row * (2u * H);
    const uint hidden_base = row * H;
    const uint remain = H - col;
    const uint lanes = min(VEC, remain);

    float4 gate = float4(0.0f);
    float4 up = float4(0.0f);

    for (uint i = 0u; i < lanes; ++i) {
        gate[i] = gu[gu_base + col + i];
        up[i] = gu[gu_base + H + col + i];
    }

    const float4 sigmoid_gate =
        1.0f / (1.0f + exp(-gate));

    const float4 result =
        gate * sigmoid_gate * up;

    for (uint i = 0u; i < lanes; ++i) {
        hidden[hidden_base + col + i] = result[i];
    }
}

kernel void swiglu_backward_inplace_kernel(
    device float* gu [[buffer(0)]],
    device const float* dhidden [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint VEC = 4u;
    const uint vectors_per_row = (H + VEC - 1u) / VEC;
    const uint row = gid / vectors_per_row;
    const uint vec_id = gid % vectors_per_row;
    const uint col = vec_id * VEC;

    if (row >= N || col >= H) {
        return;
    }

    const uint gu_base = row * (2u * H);
    const uint hidden_base = row * H;
    const uint remain = H - col;
    const uint lanes = min(VEC, remain);

    float4 gate = float4(0.0f);
    float4 up = float4(0.0f);
    float4 d_hidden = float4(0.0f);

    for (uint i = 0u; i < lanes; ++i) {
        gate[i] = gu[gu_base + col + i];
        up[i] = gu[gu_base + H + col + i];
        d_hidden[i] = dhidden[hidden_base + col + i];
    }

    const float4 sigmoid_gate =
        1.0f / (1.0f + exp(-gate));

    const float4 silu_gate =
        gate * sigmoid_gate;

    const float4 dsilu_dgate =
        sigmoid_gate *
        (1.0f + gate * (1.0f - sigmoid_gate));

    const float4 d_gate =
        d_hidden * up * dsilu_dgate;

    const float4 d_up =
        d_hidden * silu_gate;

    // 每个线程负责同一组 gate/up 元素，
    // 先读取原始 gate/up，再写回 dgate/dup，
    // 不会发生线程间写冲突。
    for (uint i = 0u; i < lanes; ++i) {
        gu[gu_base + col + i] = d_gate[i];
        gu[gu_base + H + col + i] = d_up[i];
    }
}

METAL_FUNC static float2 fa_qkv_load_weight_pair_checked_bm16(
    device const float* src,
    uint row,
    uint col,
    uint C
) {
    float2 result = float2(0.0f);

    if (row >= C || col >= C) {
        return result;
    }

    if (col + 1u < C) {
        return float2(
            *reinterpret_cast<device const packed_float2*>(
                src + row * C + col
            )
        );
    }

    result.x = src[row * C + col];
    return result;
}

METAL_FUNC static float2 fa_qkv_load_bias_pair_bm16(
    device const float* bias,
    uint channel,
    uint C
) {
    if (channel + 1u < C) {
        return float2(
            *reinterpret_cast<device const packed_float2*>(
                bias + channel
            )
        );
    }

    float2 result = float2(0.0f);

    if (channel < C) {
        result.x = bias[channel];
    }

    return result;
}

METAL_FUNC static void fa_qkv_fill_z_tile_16x64(
    threadgroup float* dst,
    device const float* x,
    uint C,
    uint input_base,
    uint simd_id,
    uint lane,
    float inv_rms0,
    float inv_rms1,
    uint global_row0,
    uint global_row1,
    bool row0_valid,
    bool row1_valid
) {
    constexpr uint kBK = 64u;
    constexpr uint kZStride = kBK + 1u;
    const uint local_c = lane * 2u;
    const uint global_c = input_base + local_c;
    const uint local_row0 = simd_id * 2u;
    const uint local_row1 = local_row0 + 1u;

    float2 z0 = float2(0.0f);
    float2 z1 = float2(0.0f);
    if (row0_valid && global_c < C) {
        float2 x0 = float2(0.0f);
        if (global_c + 1u < C) {
            x0 = float2(*reinterpret_cast<device const packed_float2*>(x + global_row0 * C + global_c));
        } else {
            x0.x = x[global_row0 * C + global_c];
        }
        z0 = fma(x0, float2(inv_rms0), 0.0);
    }

    if (row1_valid && global_c < C) {
        float2 x1 = float2(0.0f);
        if (global_c + 1u < C) {
            x1 = float2(*reinterpret_cast<device const packed_float2*>(x + global_row1 * C + global_c));
        } else {
            x1.x = x[global_row1 * C + global_c];
        }
        z1 = fma(x1, float2(inv_rms1), 0.0);
    }
    *reinterpret_cast<threadgroup packed_float2*>(dst + local_row0 * kZStride + local_c) = packed_float2(z0);
    *reinterpret_cast<threadgroup packed_float2*>(dst + local_row1 * kZStride + local_c) = packed_float2(z1);
}

METAL_FUNC static void fa_qkv_store_pair_bm16(
    device float* q_cache,
    device float* k_cache,
    device float* v_cache,
    float2 q_value,
    float2 k_value,
    float2 v_value,
    float2 q_bias,
    float2 k_bias,
    float2 v_bias,
    uint b,
    uint t,
    uint T,
    uint H,
    uint D,
    uint h0,
    uint d0,
    bool second_valid,
    bool same_head
) {
    q_value += q_bias;
    k_value += k_bias;
    v_value += v_bias;

    const uint cache_index0 =
        ((b * H + h0) * T + t) * D + d0;

    if (second_valid && same_head) {
        *reinterpret_cast<device packed_float2*>(
            q_cache + cache_index0
        ) = packed_float2(q_value);

        *reinterpret_cast<device packed_float2*>(
            k_cache + cache_index0
        ) = packed_float2(k_value);

        *reinterpret_cast<device packed_float2*>(
            v_cache + cache_index0
        ) = packed_float2(v_value);

        return;
    }

    q_cache[cache_index0] = q_value.x;
    k_cache[cache_index0] = k_value.x;
    v_cache[cache_index0] = v_value.x;

    if (second_valid) {
        const uint h1 = h0 + 1u;

        const uint cache_index1 =
            ((b * H + h1) * T + t) * D;

        q_cache[cache_index1] = q_value.y;
        k_cache[cache_index1] = k_value.y;
        v_cache[cache_index1] = v_value.y;
    }
}

/*
kernel void fused_rms_fa_qkv(
    device const float* x             [[buffer(0)]],
    device const float* wq            [[buffer(1)]],
    device const float* bq            [[buffer(2)]],
    device const float* wk            [[buffer(3)]],
    device const float* bk            [[buffer(4)]],
    device const float* wv            [[buffer(5)]],
    device const float* bv            [[buffer(6)]],
    device float* q_cache             [[buffer(7)]],
    device float* k_cache             [[buffer(8)]],
    device float* v_cache             [[buffer(9)]],
    constant uint& B                  [[buffer(10)]],
    constant uint& T                  [[buffer(11)]],
    constant uint& C                  [[buffer(12)]],
    constant uint& H                  [[buffer(13)]],
    constant uint& eps_bits           [[buffer(14)]],
    uint3 tg_pos                      [[threadgroup_position_in_grid]],
    uint3 thread_pos                 [[thread_position_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kBM = 16u;
    constexpr uint kBK = 64u;
    constexpr uint kOutputSuperTile = 128u;
    constexpr uint kFrag = 8u;
    constexpr uint kZStride = kBK + 1u;

    threadgroup float Z_tg[kBM][kZStride];
    const uint tid = thread_pos.x;

    if (tid >= kThreads || tg_pos.x != 0u || tg_pos.z != 0u ||
        B == 0u || T == 0u || C == 0u || H == 0u || (C & 3u) != 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    if (D == 0u || D > 128u) {
        return;
    }

    const uint token_count = B * T;
    const uint row_base = tg_pos.y * kBM;

    if (row_base >= token_count) {
        return;
    }

    const float eps = as_type<float>(eps_bits);
    const uint stats_local_row0 = simd_id * 2u;
    const uint stats_local_row1 = stats_local_row0 + 1u;

    const uint stats_global_row0 = row_base + stats_local_row0;
    const uint stats_global_row1 = row_base + stats_local_row1;
    const bool stats_row0_valid = stats_global_row0 < token_count;
    const bool stats_row1_valid = stats_global_row1 < token_count;

    float local_square_sum0 = 0.0f;
    float local_square_sum1 = 0.0f;
    if (stats_row0_valid) {
        const uint x_base0 = stats_global_row0 * C;
        for (uint channel = lane; channel < C; channel += kSimdWidth) {
            local_square_sum0 = fma(x[x_base0 + channel], x[x_base0 + channel], local_square_sum0);
        }
    }
    if (stats_row1_valid) {
        const uint x_base1 = stats_global_row1 * C;
        for (uint channel = lane; channel < C; channel += kSimdWidth) {
            local_square_sum1 = fma(x[x_base1 + channel], x[x_base1 + channel], local_square_sum1);
        }
    }

    const float inv_rms0 = stats_row0_valid ? rsqrt(simd_sum(local_square_sum0) / float(C) + eps) : 0.0f;
    const float inv_rms1 = stats_row1_valid ? rsqrt(simd_sum(local_square_sum1) / float(C) + eps) : 0.0f;
    const ushort2 matrix_offset = morton_order(ushort(lane));
    const uint matrix_x = uint(matrix_offset.x);
    const uint matrix_y = uint(matrix_offset.y);
    const uint simd_col_base = simd_id * 16u;
    const uint output_local_row0 = matrix_y;
    const uint output_local_row1 = 8u + matrix_y;
    const uint output_global_row0 = row_base + output_local_row0;
    const uint output_global_row1 = row_base + output_local_row1;
    const bool output_row0_valid = output_global_row0 < token_count;
    const bool output_row1_valid = output_global_row1 < token_count;

    uint output_b0 = 0u;
    uint output_t0 = 0u;
    uint output_b1 = 0u;
    uint output_t1 = 0u;
    if (output_row0_valid) {
        output_b0 = output_global_row0 / T;
        output_t0 = output_global_row0 - output_b0 * T;
    }
    if (output_row1_valid) {
        output_b1 = output_global_row1 / T;
        output_t1 = output_global_row1 - output_b1 * T;
    }

    fa_qkv_fill_z_tile_16x64(&Z_tg[0][0], x, C, 0u, simd_id, lane, inv_rms0, inv_rms1,
        stats_global_row0, stats_global_row1, stats_row0_valid, stats_row1_valid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint output_base = 0u; output_base < C; output_base += kOutputSuperTile) {
        float2 q_r0_c0 = float2(0.0f);
        float2 q_r0_c1 = float2(0.0f);
        float2 q_r1_c0 = float2(0.0f);
        float2 q_r1_c1 = float2(0.0f);

        float2 k_r0_c0 = float2(0.0f);
        float2 k_r0_c1 = float2(0.0f);
        float2 k_r1_c0 = float2(0.0f);
        float2 k_r1_c1 = float2(0.0f);

        float2 v_r0_c0 = float2(0.0f);
        float2 v_r0_c1 = float2(0.0f);
        float2 v_r1_c0 = float2(0.0f);
        float2 v_r1_c1 = float2(0.0f);

        const uint weight_col0 = output_base + simd_col_base + matrix_x;
        const uint weight_col1 = weight_col0 + 8u;
        const bool full_output_tile = output_base + simd_col_base + 16u <= C;
        for (uint input_base = 0u; input_base < C; input_base += kBK) {
            threadgroup const float* z_tile = &Z_tg[0][0];
            const bool full_input_tile = input_base + kBK <= C;
            const bool full_weight_tile = full_input_tile && full_output_tile;
            uint z_offset0 = matrix_y * kZStride + matrix_x;
            uint z_offset1 = (8u + matrix_y) * kZStride + matrix_x;
            uint weight_row = input_base + matrix_y;
            uint weight_offset = weight_row * C + weight_col0;
            const uint weight_k_step = kFrag * C;
            for (uint k_sub = 0u; k_sub < kBK; k_sub += kFrag) {
                const float2 z_fragment0 = float2(*reinterpret_cast<threadgroup const packed_float2*>(z_tile + z_offset0));
                const float2 z_fragment1 = float2(*reinterpret_cast<threadgroup const packed_float2*>(z_tile + z_offset1));
                float2 wq_fragment0;
                float2 wq_fragment1;

                if (full_weight_tile) {
                    wq_fragment0 = float2(*reinterpret_cast<device const packed_float2*>(wq + weight_offset));
                    wq_fragment1 = float2(*reinterpret_cast<device const packed_float2*>(wq + weight_offset + 8u));
                } else {
                    wq_fragment0 = fa_qkv_load_weight_pair_checked_bm16(wq, weight_row, weight_col0, C);
                    wq_fragment1 = fa_qkv_load_weight_pair_checked_bm16(wq, weight_row, weight_col1, C);
                }
                q_r0_c0 = lowreg_matrix::mma_8x8(z_fragment0, wq_fragment0, q_r0_c0);
                q_r1_c0 = lowreg_matrix::mma_8x8(z_fragment1, wq_fragment0, q_r1_c0);
                q_r0_c1 = lowreg_matrix::mma_8x8(z_fragment0, wq_fragment1, q_r0_c1);
                q_r1_c1 = lowreg_matrix::mma_8x8(z_fragment1, wq_fragment1, q_r1_c1);

                float2 wk_fragment0;
                float2 wk_fragment1;
                if (full_weight_tile) {
                    wk_fragment0 = float2(*reinterpret_cast<device const packed_float2*>(wk + weight_offset));
                    wk_fragment1 = float2(*reinterpret_cast<device const packed_float2*>(wk + weight_offset + 8u));
                } else {
                    wk_fragment0 = fa_qkv_load_weight_pair_checked_bm16(wk, weight_row, weight_col0, C);
                    wk_fragment1 = fa_qkv_load_weight_pair_checked_bm16(wk, weight_row, weight_col1, C);
                }
                k_r0_c0 = lowreg_matrix::mma_8x8(z_fragment0, wk_fragment0, k_r0_c0);
                k_r1_c0 = lowreg_matrix::mma_8x8(z_fragment1, wk_fragment0, k_r1_c0);
                k_r0_c1 = lowreg_matrix::mma_8x8(z_fragment0, wk_fragment1, k_r0_c1);
                k_r1_c1 = lowreg_matrix::mma_8x8(z_fragment1, wk_fragment1, k_r1_c1);

                float2 wv_fragment0;
                float2 wv_fragment1;
                if (full_weight_tile) {
                    wv_fragment0 = float2(*reinterpret_cast<device const packed_float2*>(wv + weight_offset));
                    wv_fragment1 = float2(*reinterpret_cast<device const packed_float2*>(wv + weight_offset + 8u));
                } else {
                    wv_fragment0 = fa_qkv_load_weight_pair_checked_bm16(wv, weight_row, weight_col0, C);
                    wv_fragment1 = fa_qkv_load_weight_pair_checked_bm16(wv, weight_row, weight_col1, C);
                }
                v_r0_c0 = lowreg_matrix::mma_8x8(z_fragment0, wv_fragment0, v_r0_c0);
                v_r1_c0 = lowreg_matrix::mma_8x8(z_fragment1, wv_fragment0, v_r1_c0);
                v_r0_c1 = lowreg_matrix::mma_8x8(z_fragment0, wv_fragment1, v_r0_c1);
                v_r1_c1 = lowreg_matrix::mma_8x8(z_fragment1, wv_fragment1, v_r1_c1);
                z_offset0 += kFrag;
                z_offset1 += kFrag;
                weight_row += kFrag;
                weight_offset += weight_k_step;
            }

            const bool has_next_input = input_base + kBK < C;
            const bool has_next_output = output_base + kOutputSuperTile < C;
            const bool has_next_task = has_next_input || has_next_output;
            if (has_next_task) {
                const uint next_input_base = has_next_input ? input_base + kBK : 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                fa_qkv_fill_z_tile_16x64(&Z_tg[0][0], x, C, next_input_base, simd_id, lane, inv_rms0, inv_rms1,
                    stats_global_row0, stats_global_row1, stats_row0_valid, stats_row1_valid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        const uint output_channel0 = output_base + simd_col_base + matrix_x;
        const uint output_channel1 = output_channel0 + 8u;
        const bool pair0_first_valid = output_channel0 < C;
        const bool pair0_second_valid =
            output_channel0 + 1u < C;

        const bool pair1_first_valid =
            output_channel1 < C;

        const bool pair1_second_valid =
            output_channel1 + 1u < C;

        uint pair0_h = 0u;
        uint pair0_d = 0u;
        bool pair0_same_head = false;

        uint pair1_h = 0u;
        uint pair1_d = 0u;
        bool pair1_same_head = false;

        if (pair0_first_valid) {
            pair0_h = output_channel0 / D;
            pair0_d =
                output_channel0 - pair0_h * D;

            pair0_same_head =
                pair0_second_valid &&
                pair0_d + 1u < D;
        }

        if (pair1_first_valid) {
            pair1_h = output_channel1 / D;
            pair1_d =
                output_channel1 - pair1_h * D;

            pair1_same_head =
                pair1_second_valid &&
                pair1_d + 1u < D;
        }

        const float2 bq0 =
            fa_qkv_load_bias_pair_bm16(
                bq,
                output_channel0,
                C
            );

        const float2 bk0 =
            fa_qkv_load_bias_pair_bm16(
                bk,
                output_channel0,
                C
            );

        const float2 bv0 =
            fa_qkv_load_bias_pair_bm16(
                bv,
                output_channel0,
                C
            );

        const float2 bq1 =
            fa_qkv_load_bias_pair_bm16(
                bq,
                output_channel1,
                C
            );

        const float2 bk1 =
            fa_qkv_load_bias_pair_bm16(
                bk,
                output_channel1,
                C
            );

        const float2 bv1 =
            fa_qkv_load_bias_pair_bm16(
                bv,
                output_channel1,
                C
            );

        if (output_row0_valid) {
            if (pair0_first_valid) {
                fa_qkv_store_pair_bm16(
                    q_cache,
                    k_cache,
                    v_cache,

                    q_r0_c0,
                    k_r0_c0,
                    v_r0_c0,

                    bq0,
                    bk0,
                    bv0,

                    output_b0,
                    output_t0,
                    T,
                    H,
                    D,

                    pair0_h,
                    pair0_d,

                    pair0_second_valid,
                    pair0_same_head
                );
            }

            if (pair1_first_valid) {
                fa_qkv_store_pair_bm16(
                    q_cache,
                    k_cache,
                    v_cache,

                    q_r0_c1,
                    k_r0_c1,
                    v_r0_c1,

                    bq1,
                    bk1,
                    bv1,

                    output_b0,
                    output_t0,
                    T,
                    H,
                    D,

                    pair1_h,
                    pair1_d,

                    pair1_second_valid,
                    pair1_same_head
                );
            }
        }

        if (output_row1_valid) {
            if (pair0_first_valid) {
                fa_qkv_store_pair_bm16(
                    q_cache,
                    k_cache,
                    v_cache,

                    q_r1_c0,
                    k_r1_c0,
                    v_r1_c0,

                    bq0,
                    bk0,
                    bv0,

                    output_b1,
                    output_t1,
                    T,
                    H,
                    D,

                    pair0_h,
                    pair0_d,

                    pair0_second_valid,
                    pair0_same_head
                );
            }

            if (pair1_first_valid) {
                fa_qkv_store_pair_bm16(
                    q_cache,
                    k_cache,
                    v_cache,

                    q_r1_c1,
                    k_r1_c1,
                    v_r1_c1,

                    bq1,
                    bk1,
                    bv1,

                    output_b1,
                    output_t1,
                    T,
                    H,
                    D,

                    pair1_h,
                    pair1_d,

                    pair1_second_valid,
                    pair1_same_head
                );
            }
        }
    }
}
*/


inline float4 fq2_load_x4(device const float* row_ptr, uint col, uint C) {
    if (col + 4u <= C) {
        return float4(*reinterpret_cast<device const packed_float4*>(row_ptr + col));
    }
    float4 v = float4(0.0f);
    if (col + 0u < C) v.x = row_ptr[col + 0u];
    if (col + 1u < C) v.y = row_ptr[col + 1u];
    if (col + 2u < C) v.z = row_ptr[col + 2u];
    if (col + 3u < C) v.w = row_ptr[col + 3u];
    return v;
}

inline void fq2_acc_sq(float4 v, thread float& s) {
    s = fma(v.x, v.x, s);
    s = fma(v.y, v.y, s);
    s = fma(v.z, v.z, s);
    s = fma(v.w, v.w, s);
}

inline float fq2_reduce16(float v) {
    v += simd_shuffle_xor(v, ushort(8u));
    v += simd_shuffle_xor(v, ushort(4u));
    v += simd_shuffle_xor(v, ushort(2u));
    v += simd_shuffle_xor(v, ushort(1u));
    return v;
}

// ==== MMA 扫描: 分支在模板参数里, 内层循环零分支 ====
template <bool kChecked> inline void fq2_mma_sweep(
    threadgroup const float* Z,
    device const float* wq, device const float* wk, device const float* wv,
    uint C, uint input_base,
    uint matrix_x, uint matrix_y,
    uint weight_col0, uint weight_col1,
    thread float2& q00, thread float2& q01, thread float2& q10, thread float2& q11,
    thread float2& k00, thread float2& k01, thread float2& k10, thread float2& k11,
    thread float2& v00, thread float2& v01, thread float2& v10, thread float2& v11)
{
    constexpr uint kFq2BM      = 16u;
    constexpr uint kFq2BK      = 128u;              // 可调: 128 / 192 / 256
    constexpr uint kFq2BN      = 128u;              // 输出 supertile, 不变
    constexpr uint kFq2Frag    = 8u;
    constexpr uint kFq2ZStride = kFq2BK + 4u;       // 132: 行长 528B, 16B 对齐
    constexpr uint kFq2Fill    = kFq2BK / 64u;      // 每线程每次 fill 的 float4 数 = 2

    uint z_off0 = matrix_y * kFq2ZStride + matrix_x;
    uint z_off1 = z_off0 + 8u * kFq2ZStride;
    uint w_row  = input_base + matrix_y;
    uint w_off  = w_row * C + weight_col0;
    const uint w_step = kFq2Frag * C;

    for (uint ks = 0u; ks < kFq2BK; ks += kFq2Frag) {
        const float2 z0 = float2(*reinterpret_cast<threadgroup const packed_float2*>(Z + z_off0));
        const float2 z1 = float2(*reinterpret_cast<threadgroup const packed_float2*>(Z + z_off1));

        float2 wqf0, wqf1, wkf0, wkf1, wvf0, wvf1;
        if (kChecked) {
            wqf0 = fa_qkv_load_weight_pair_checked_bm16(wq, w_row, weight_col0, C);
            wqf1 = fa_qkv_load_weight_pair_checked_bm16(wq, w_row, weight_col1, C);
            wkf0 = fa_qkv_load_weight_pair_checked_bm16(wk, w_row, weight_col0, C);
            wkf1 = fa_qkv_load_weight_pair_checked_bm16(wk, w_row, weight_col1, C);
            wvf0 = fa_qkv_load_weight_pair_checked_bm16(wv, w_row, weight_col0, C);
            wvf1 = fa_qkv_load_weight_pair_checked_bm16(wv, w_row, weight_col1, C);
        } else {
            // 6 个 load 集中发射, 提高访存级并行
            wqf0 = float2(*reinterpret_cast<device const packed_float2*>(wq + w_off));
            wqf1 = float2(*reinterpret_cast<device const packed_float2*>(wq + w_off + 8u));
            wkf0 = float2(*reinterpret_cast<device const packed_float2*>(wk + w_off));
            wkf1 = float2(*reinterpret_cast<device const packed_float2*>(wk + w_off + 8u));
            wvf0 = float2(*reinterpret_cast<device const packed_float2*>(wv + w_off));
            wvf1 = float2(*reinterpret_cast<device const packed_float2*>(wv + w_off + 8u));
        }

        q00 = lowreg_matrix::mma_8x8(z0, wqf0, q00);
        q10 = lowreg_matrix::mma_8x8(z1, wqf0, q10);
        q01 = lowreg_matrix::mma_8x8(z0, wqf1, q01);
        q11 = lowreg_matrix::mma_8x8(z1, wqf1, q11);
        k00 = lowreg_matrix::mma_8x8(z0, wkf0, k00);
        k10 = lowreg_matrix::mma_8x8(z1, wkf0, k10);
        k01 = lowreg_matrix::mma_8x8(z0, wkf1, k01);
        k11 = lowreg_matrix::mma_8x8(z1, wkf1, k11);
        v00 = lowreg_matrix::mma_8x8(z0, wvf0, v00);
        v10 = lowreg_matrix::mma_8x8(z1, wvf0, v10);
        v01 = lowreg_matrix::mma_8x8(z0, wvf1, v01);
        v11 = lowreg_matrix::mma_8x8(z1, wvf1, v11);

        z_off0 += kFq2Frag;
        z_off1 += kFq2Frag;
        w_row  += kFq2Frag;
        w_off  += w_step;
    }
}

// ==== 内核: 签名/调度与原版完全一致 (grid.x = 256, 即 x=1) ====
kernel void fused_rms_fa_qkv(
    device const float* x        [[buffer(0)]],
    device const float* wq       [[buffer(1)]],
    device const float* bq       [[buffer(2)]],
    device const float* wk       [[buffer(3)]],
    device const float* bk       [[buffer(4)]],
    device const float* wv       [[buffer(5)]],
    device const float* bv       [[buffer(6)]],
    device float* q_cache        [[buffer(7)]],
    device float* k_cache        [[buffer(8)]],
    device float* v_cache        [[buffer(9)]],
    constant uint& B             [[buffer(10)]],
    constant uint& T             [[buffer(11)]],
    constant uint& C             [[buffer(12)]],
    constant uint& H             [[buffer(13)]],
    constant uint& eps_bits      [[buffer(14)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 thread_pos             [[thread_position_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  simd_id                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kFq2BM      = 16u;
    constexpr uint kFq2BK      = 128u;              // 可调: 128 / 192 / 256
    constexpr uint kFq2BN      = 128u;              // 输出 supertile, 不变
    constexpr uint kFq2Frag    = 8u;
    constexpr uint kFq2ZStride = kFq2BK + 4u;       // 132: 行长 528B, 16B 对齐
    constexpr uint kFq2Fill    = kFq2BK / 64u;      // 每线程每次 fill 的 float4 数 = 2

    threadgroup float4 Z4[kFq2BM][kFq2ZStride / 4u];   // 8.25KB (kBK=128)
    threadgroup float  inv_rms_tg[kFq2BM];
    threadgroup float* Z = reinterpret_cast<threadgroup float*>(&Z4[0][0]);

    const uint tid = thread_pos.x;
    if (tg_pos.x != 0u || tg_pos.z != 0u ||
        B == 0u || T == 0u || C == 0u || H == 0u || (C & 3u) != 0u || C % H != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > 128u) return;

    const uint token_count = B * T;
    const uint row_base = tg_pos.y * kFq2BM;
    if (row_base >= token_count) return;

    const float eps = as_type<float>(eps_bits);

    // ---- fill 坐标: 16 线程一行, f_row == simd_id*2 + (lane>>4) ----
    const uint f_row  = tid >> 4;
    const uint f_c16  = tid & 15u;
    const uint f_grow = row_base + f_row;
    const bool f_valid = f_grow < token_count;
    device const float* x_row = x + f_grow * C;

    // ---- 输出坐标(与原版一致) ----
    const ushort2 mo = morton_order(ushort(lane));
    const uint matrix_x = uint(mo.x);
    const uint matrix_y = uint(mo.y);
    const uint simd_col_base = simd_id * 16u;
    const uint out_grow0 = row_base + matrix_y;
    const uint out_grow1 = out_grow0 + 8u;
    const bool out_row0_valid = out_grow0 < token_count;
    const bool out_row1_valid = out_grow1 < token_count;
    uint ob0 = 0u, ot0 = 0u, ob1 = 0u, ot1 = 0u;
    if (out_row0_valid) { ob0 = out_grow0 / T; ot0 = out_grow0 - ob0 * T; }
    if (out_row1_valid) { ob1 = out_grow1 / T; ot1 = out_grow1 - ob1 * T; }

    // ---- 首次填充: 原始 x(无乘法), 顺手累加 x^2 ----
    float sumsq = 0.0f;
    for (uint i = 0u; i < kFq2Fill; ++i) {
        const uint col = (f_c16 + i * 16u) * 4u;
        const float4 v = f_valid ? fq2_load_x4(x_row, col, C) : float4(0.0f);
        Z4[f_row][f_c16 + i * 16u] = v;
        fq2_acc_sq(v, sumsq);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool sumsq_pending = true;
    const bool z_resident = (C <= kFq2BK);   // 小 C 时 Z 常驻, 后续 output tile 零 refill

    for (uint output_base = 0u; output_base < C; output_base += kFq2BN) {
        float2 q00 = float2(0.0f), q01 = float2(0.0f), q10 = float2(0.0f), q11 = float2(0.0f);
        float2 k00 = float2(0.0f), k01 = float2(0.0f), k10 = float2(0.0f), k11 = float2(0.0f);
        float2 v00 = float2(0.0f), v01 = float2(0.0f), v10 = float2(0.0f), v11 = float2(0.0f);

        const uint weight_col0 = output_base + simd_col_base + matrix_x;
        const uint weight_col1 = weight_col0 + 8u;
        const bool full_output_tile = output_base + simd_col_base + 16u <= C;

        for (uint input_base = 0u; input_base < C; input_base += kFq2BK) {
            // 1) 先发射下一块的 device load 到寄存器, 与 MMA 重叠
            const bool has_next_input = input_base + kFq2BK < C;
            const bool prep_next_out  = (!z_resident) && (!has_next_input) &&
                                        (output_base + kFq2BN < C);
            const bool do_refill = has_next_input || prep_next_out;
            const uint next_base = has_next_input ? (input_base + kFq2BK) : 0u;

            float4 pre[kFq2Fill];
            if (do_refill) {
                for (uint i = 0u; i < kFq2Fill; ++i) {
                    const uint col = next_base + (f_c16 + i * 16u) * 4u;
                    pre[i] = f_valid ? fq2_load_x4(x_row, col, C) : float4(0.0f);
                }
            }

            // 2) MMA: 分支已提出内层循环
            if (full_output_tile && (input_base + kFq2BK <= C)) {
                fq2_mma_sweep<false>(Z, wq, wk, wv, C, input_base,
                    matrix_x, matrix_y, weight_col0, weight_col1,
                    q00, q01, q10, q11, k00, k01, k10, k11, v00, v01, v10, v11);
            } else {
                fq2_mma_sweep<true>(Z, wq, wk, wv, C, input_base,
                    matrix_x, matrix_y, weight_col0, weight_col1,
                    q00, q01, q10, q11, k00, k01, k10, k11, v00, v01, v10, v11);
            }

            // 3) 写回 Z; 第一遍扫描期间继续累加 x^2 (prep_next_out 重装 block0 不重复累加)
            if (do_refill) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0u; i < kFq2Fill; ++i) {
                    Z4[f_row][f_c16 + i * 16u] = pre[i];
                }
                if (sumsq_pending && has_next_input) {
                    for (uint i = 0u; i < kFq2Fill; ++i) fq2_acc_sq(pre[i], sumsq);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // ---- 第一遍 K 扫描结束: sumsq 完整, 归约一次, 全程复用 ----
        if (sumsq_pending) {
            const float s = fq2_reduce16(sumsq);
            if ((lane & 15u) == 0u) {
                inv_rms_tg[f_row] = f_valid ? rsqrt(s / float(C) + eps) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sumsq_pending = false;
        }

        // ---- 出口: s·acc, 再由 store helper 加 bias(与原版一致) ----
        const float s0 = inv_rms_tg[matrix_y];
        const float s1 = inv_rms_tg[8u + matrix_y];

        const uint output_channel0 = output_base + simd_col_base + matrix_x;
        const uint output_channel1 = output_channel0 + 8u;
        const bool pair0_first_valid  = output_channel0 < C;
        const bool pair0_second_valid = output_channel0 + 1u < C;
        const bool pair1_first_valid  = output_channel1 < C;
        const bool pair1_second_valid = output_channel1 + 1u < C;

        uint pair0_h = 0u, pair0_d = 0u; bool pair0_same_head = false;
        uint pair1_h = 0u, pair1_d = 0u; bool pair1_same_head = false;
        if (pair0_first_valid) {
            pair0_h = output_channel0 / D;
            pair0_d = output_channel0 - pair0_h * D;
            pair0_same_head = pair0_second_valid && (pair0_d + 1u < D);
        }
        if (pair1_first_valid) {
            pair1_h = output_channel1 / D;
            pair1_d = output_channel1 - pair1_h * D;
            pair1_same_head = pair1_second_valid && (pair1_d + 1u < D);
        }

        const float2 bq0 = fa_qkv_load_bias_pair_bm16(bq, output_channel0, C);
        const float2 bk0 = fa_qkv_load_bias_pair_bm16(bk, output_channel0, C);
        const float2 bv0 = fa_qkv_load_bias_pair_bm16(bv, output_channel0, C);
        const float2 bq1 = fa_qkv_load_bias_pair_bm16(bq, output_channel1, C);
        const float2 bk1 = fa_qkv_load_bias_pair_bm16(bk, output_channel1, C);
        const float2 bv1 = fa_qkv_load_bias_pair_bm16(bv, output_channel1, C);

        if (out_row0_valid) {
            if (pair0_first_valid) {
                fa_qkv_store_pair_bm16(q_cache, k_cache, v_cache,
                    q00 * s0, k00 * s0, v00 * s0, bq0, bk0, bv0,
                    ob0, ot0, T, H, D, pair0_h, pair0_d,
                    pair0_second_valid, pair0_same_head);
            }
            if (pair1_first_valid) {
                fa_qkv_store_pair_bm16(q_cache, k_cache, v_cache,
                    q01 * s0, k01 * s0, v01 * s0, bq1, bk1, bv1,
                    ob0, ot0, T, H, D, pair1_h, pair1_d,
                    pair1_second_valid, pair1_same_head);
            }
        }
        if (out_row1_valid) {
            if (pair0_first_valid) {
                fa_qkv_store_pair_bm16(q_cache, k_cache, v_cache,
                    q10 * s1, k10 * s1, v10 * s1, bq0, bk0, bv0,
                    ob1, ot1, T, H, D, pair0_h, pair0_d,
                    pair0_second_valid, pair0_same_head);
            }
            if (pair1_first_valid) {
                fa_qkv_store_pair_bm16(q_cache, k_cache, v_cache,
                    q11 * s1, k11 * s1, v11 * s1, bq1, bk1, bv1,
                    ob1, ot1, T, H, D, pair1_h, pair1_d,
                    pair1_second_valid, pair1_same_head);
            }
        }
    }
}

kernel void fused_fa_from_qkv(
    device const float* q_cache     [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device float* lse_out           [[buffer(4)]],
    constant uint& B                [[buffer(5)]],
    constant uint& H                [[buffer(6)]],
    constant uint& T                [[buffer(7)]],
    constant uint& C                [[buffer(8)]],
    constant uint& is_causal_int    [[buffer(9)]],
    constant uint& write_lse        [[buffer(10)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]]
) {
    constexpr uint kBQ = 8u;
    constexpr uint kBK = 32u;
    constexpr uint kBD = 64u;
    constexpr uint kFrag = 8u;
    constexpr uint kThreads = 128u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kMaxHeadDim = 128u;
    // Keep Q in up to two 64-wide sub-tiles. This preserves the compact
    // stride for D<=64 while still covering every D<=128 configuration.
    constexpr uint kQStride = kBD + 1u;
    constexpr uint kKTStride = kFrag + 1u;
    constexpr uint kPStride = kBK + 1u;
    constexpr uint kQChunkCount = kBQ * kQStride;
    constexpr uint kQChunks = kMaxHeadDim / kBD;
    constexpr uint kQCount = kQChunks * kQChunkCount;
    constexpr uint kKTGroupCount = kBD * kKTStride;
    constexpr uint kKTCount = kSimdGroups * kKTGroupCount;
    constexpr uint kTileScratchCount = kQCount + kKTCount;
    constexpr uint kFloat4 = 4u;
    threadgroup float tile_scratch[kTileScratchCount];
    threadgroup float P_tg[kBQ][kPStride];
    threadgroup float running_max_tg[kBQ];
    threadgroup float running_sum_tg[kBQ];
    threadgroup float alpha_tg[kBQ];
    const uint tid = thread_pos.x;

    // 所有线程必须根据统一条件退出，避免barrier死锁。
    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u || 
        simd_lane_count != 32u || tg_pos.x != 0u || B == 0u ||  H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > kMaxHeadDim) {
        return;
    }
    const uint bh = tg_pos.z;
    if (bh >= B * H) {
        return;
    }
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) {
        return;
    }
    const bool is_causal = is_causal_int != 0u;
    const float score_scale = rsqrt(float(D));
    float2 output_accumulator0 = float2(0.0f);
    float2 output_accumulator1 = float2(0.0f);
    float2 output_accumulator2 = float2(0.0f);
    float2 output_accumulator3 = float2(0.0f);
    if (tid < kBQ) {
        running_max_tg[tid] = -INFINITY;
        running_sum_tg[tid] = 0.0f;
        alpha_tg[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* Q_all = tile_scratch;
    threadgroup float* KT_all = tile_scratch + kQCount;

    // Q is invariant across all key blocks.  Keep it in the Q half of
    // tile_scratch and load it once; the K half is reused by V below.
    // Write each active 64-wide sub-tile, including zero padding. The MMA
    // loop always consumes a complete 64-wide slice.
    const uint q_chunks = (D + kBD - 1u) / kBD;
    constexpr uint kQFloat4PerChunk = kBD / kFloat4;
    for (uint index = tid; index < q_chunks * kBQ * kQFloat4PerChunk; index += kThreads) {
        const uint chunk = index / (kBQ * kQFloat4PerChunk);
        const uint chunk_index = index - chunk * kBQ * kQFloat4PerChunk;
        const uint local_q = chunk_index / kQFloat4PerChunk;
        const uint local_d = (chunk_index - local_q * kQFloat4PerChunk) * kFloat4;
        const uint query = query_base + local_q;
        const uint global_d = chunk * kBD + local_d;
        float4 value = float4(0.0f);
        if (query < T && global_d + 3u < D) {
            const uint q_index = (bh * T + query) * D + global_d;
            value = float4(*reinterpret_cast<device const packed_float4*>(q_cache + q_index));
        } else if (query < T && global_d < D) {
            #pragma unroll
            for (uint component = 0u; component < 4u; ++component) {
                if (global_d + component < D) {
                    value[component] = q_cache[(bh * T + query) * D + global_d + component];
                }
            }
        }
        threadgroup float* Q_tg = Q_all + chunk * kQChunkCount;
        Q_tg[local_q * kQStride + local_d + 0u] = value[0];
        Q_tg[local_q * kQStride + local_d + 1u] = value[1];
        Q_tg[local_q * kQStride + local_d + 2u] = value[2];
        Q_tg[local_q * kQStride + local_d + 3u] = value[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;
    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {
        // QK^T
        float2 score_accumulator = float2(0.0f);
        for (uint d_base = 0u; d_base < D; d_base += kBD) {
            threadgroup float* Q_tg = Q_all + (d_base / kBD) * kQChunkCount;

            // Only a later slice needs protection before reusing KT_tg.
            if (d_base != 0u) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            // 加载K并转置：device K：[key,D] KT_tg：[D,32]
            threadgroup float* KT_tg = KT_all + simd_id * kKTGroupCount;
            for (uint index = lane; index < (kFrag * kBD) / kFloat4; index += 32u) {
                const uint local_key = index / (kBD / kFloat4);
                const uint local_d = (index - local_key * (kBD / kFloat4)) * kFloat4;
                const uint key = key_base + simd_id * kFrag + local_key;
                const uint global_d = d_base + local_d;
                float4 value = float4(0.0f);
                if (key < T && global_d + 3u < D) {
                    const uint k_index = (bh * T + key) * D + global_d;
                    value = float4(*reinterpret_cast<device const packed_float4*>(k_cache + k_index));
                } else if (key < T && global_d < D) {
                    #pragma unroll
                    for (uint component = 0u; component < 4u; ++component) {
                        if (global_d + component < D) {
                            value[component] = k_cache[(bh * T + key) * D + global_d + component];
                        }
                    }
                }
                KT_tg[(local_d + 0u) * kKTStride + local_key] = value[0];
                KT_tg[(local_d + 1u) * kKTStride + local_key] = value[1];
                KT_tg[(local_d + 2u) * kKTStride + local_key] = value[2];
                KT_tg[(local_d + 3u) * kKTStride + local_key] = value[3];
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint d_sub = 0u; d_sub < kBD; d_sub += kFrag) {
                const float2 q_fragment = lowreg_matrix::load_tg_8x8(Q_tg,
                        kQStride, ushort2(ushort(0u), ushort(d_sub)), ushort(lane));
                const float2 k_fragment = lowreg_matrix::load_tg_8x8(KT_tg,
                        kKTStride, ushort2(ushort(d_sub), ushort(0u)), ushort(lane));
                score_accumulator = lowreg_matrix::mma_8x8(q_fragment, k_fragment, score_accumulator);
            }
        }

        // Scale和causal mask
        const ushort2 score_offset = morton_order(ushort(lane));
        const uint local_q = uint(score_offset.y);
        const uint local_key0 = simd_id * kFrag + uint(score_offset.x);
        const uint local_key1 = local_key0 + 1u;
        const uint query = query_base + local_q;
        const uint key0 = key_base + local_key0;
        const uint key1 = key_base + local_key1;
        const bool valid0 = query < T && key0 < T && (!is_causal || key0 <= query);
        const bool valid1 = query < T && key1 < T && (!is_causal || key1 <= query);
        P_tg[local_q][local_key0] = valid0 ? score_accumulator[0] * score_scale : -INFINITY;
        P_tg[local_q][local_key1] = valid1 ? score_accumulator[1] * score_scale : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 32-key block online softmax
        const uint row_half = lane >> 4u;
        const uint half_lane = lane & 15u;
        const uint softmax_row = simd_id * 2u + row_half;
        const uint probability_col0 = half_lane;
        const uint probability_col1 = half_lane + 16u;
        const float score0 = P_tg[softmax_row][probability_col0];
        const float score1 = P_tg[softmax_row][probability_col1];
        float local_max = max(score0, score1);
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(8u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(4u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(2u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(1u)));

        const ushort leader_lane = ushort(row_half * 16u);
        const float tile_max = simd_shuffle(local_max, leader_lane);
        const float old_max = running_max_tg[softmax_row];
        const float new_max = max(old_max, tile_max);
        const float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        const float probability0 = isfinite(score0) ? exp(score0 - new_max) : 0.0f;
        const float probability1 = isfinite(score1) ? exp(score1 - new_max) : 0.0f;
        float local_probability_sum = probability0 + probability1;
        local_probability_sum += simd_shuffle_down(local_probability_sum, ushort(8u));
        local_probability_sum += simd_shuffle_down(local_probability_sum, ushort(4u));
        local_probability_sum += simd_shuffle_down(local_probability_sum, ushort(2u));
        local_probability_sum += simd_shuffle_down(local_probability_sum, ushort(1u));
        const float tile_probability_sum = simd_shuffle(local_probability_sum, leader_lane);
        P_tg[softmax_row][probability_col0] = probability0;
        P_tg[softmax_row][probability_col1] = probability1;

        if (half_lane == 0u) {
            running_max_tg[softmax_row] = new_max;
            running_sum_tg[softmax_row] = running_sum_tg[softmax_row] * old_scale + tile_probability_sum;
            alpha_tg[softmax_row] = old_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 对旧的输出numerator进行online-softmax缩放。
        const ushort2 output_offset = morton_order(ushort(lane));
        const uint output_local_q = uint(output_offset.y);
        const float output_old_scale = alpha_tg[output_local_q];
        output_accumulator0 *= output_old_scale;
        output_accumulator1 *= output_old_scale;
        output_accumulator2 *= output_old_scale;
        output_accumulator3 *= output_old_scale;

        // P × V.  Load only the two V fragments consumed by this lane
        // directly into registers.  This avoids a shared V tile and the
        // cross-threadgroup overwrite race between successive key tiles.
        for (uint d_base = 0u; d_base < D; d_base += kBD) {
            const ushort2 v_offset = morton_order(ushort(lane));
            const uint v_key = key_base + uint(v_offset.y);
            const uint v_col0 = d_base + simd_id * kFrag + uint(v_offset.x);
            const uint v_col1 = d_base + (simd_id + kSimdGroups) * kFrag + uint(v_offset.x);
            float2 block_accumulator0;
            float2 block_accumulator1;
            if (d_base == 0u) {
                block_accumulator0 = output_accumulator0;
                block_accumulator1 = output_accumulator1;
            } else {
                block_accumulator0 = output_accumulator2;
                block_accumulator1 = output_accumulator3;
            }

            constexpr uint kVSubtiles = kBK / kFrag;
            float2 v_fragments0[kVSubtiles];
            float2 v_fragments1[kVSubtiles];
            #pragma unroll
            for (uint sub = 0u; sub < kVSubtiles; ++sub) {
                float2 v_fragment0 = float2(0.0f);
                float2 v_fragment1 = float2(0.0f);
                const uint key = v_key + sub * kFrag;
                if (key < T && v_col0 + 1u < D) {
                    const uint v_index = (bh * T + key) * D + v_col0;
                    v_fragment0 = float2(*reinterpret_cast<device const packed_float2*>(v_cache + v_index));
                } else if (key < T && v_col0 < D) {
                    v_fragment0[0] = v_cache[(bh * T + key) * D + v_col0];
                    if (v_col0 + 1u < D) v_fragment0[1] = v_cache[(bh * T + key) * D + v_col0 + 1u];
                }
                if (key < T && v_col1 + 1u < D) {
                    const uint v_index = (bh * T + key) * D + v_col1;
                    v_fragment1 = float2(*reinterpret_cast<device const packed_float2*>(v_cache + v_index));
                } else if (key < T && v_col1 < D) {
                    v_fragment1[0] = v_cache[(bh * T + key) * D + v_col1];
                    if (v_col1 + 1u < D) v_fragment1[1] = v_cache[(bh * T + key) * D + v_col1 + 1u];
                }
                v_fragments0[sub] = v_fragment0;
                v_fragments1[sub] = v_fragment1;
            }
            #pragma unroll
            for (uint sub = 0u; sub < kVSubtiles; ++sub) {
                const uint key_sub = sub * kFrag;
                const float2 probability_fragment = lowreg_matrix::load_tg_8x8(reinterpret_cast<threadgroup const float*>(P_tg),
                        kPStride, ushort2(ushort(0u), ushort(key_sub)), ushort(lane));
                const float2 v_fragment0 = v_fragments0[sub];
                const float2 v_fragment1 = v_fragments1[sub];
                block_accumulator0 = lowreg_matrix::mma_8x8(probability_fragment, v_fragment0, block_accumulator0);
                block_accumulator1 = lowreg_matrix::mma_8x8(probability_fragment, v_fragment1, block_accumulator1);
            }
            if (d_base == 0u) {
                output_accumulator0 = block_accumulator0;
                output_accumulator1 = block_accumulator1;
            } else {
                output_accumulator2 = block_accumulator0;
                output_accumulator3 = block_accumulator1;
            }
        }

        // P_tg is reused by the next key tile.  All SIMD groups must finish
        // their current P×V reads before any group overwrites P_tg.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 最终归一化并写入[B,T,C]
    const ushort2 final_offset = morton_order(ushort(lane));
    const uint final_local_q = uint(final_offset.y);
    const uint final_query = query_base + final_local_q;
    if (final_query < T) {
        const float denominator = running_sum_tg[final_local_q];
        const float inverse_sum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        const uint pair_offset = uint(final_offset.x);
        if (write_lse != 0u && simd_id == 0u && pair_offset == 0u) {
            lse_out[bh * T + final_query] = denominator > 0.0f ? running_max_tg[final_local_q] + log(denominator) : 0.0f;
        }
        const uint output_row_base = (b * T + final_query) * C + h * D;
        const uint d00 = simd_id * kFrag + pair_offset;
        if (d00 < D) {
            out[output_row_base + d00] = output_accumulator0[0] * inverse_sum;
        }
        if (d00 + 1u < D) {
            out[output_row_base + d00 + 1u] = output_accumulator0[1] * inverse_sum;
        }
        const uint d01 = (simd_id + kSimdGroups) * kFrag + pair_offset;
        if (d01 < D) {
            out[output_row_base + d01] = output_accumulator1[0] * inverse_sum;
        }
        if (d01 + 1u < D) {
            out[output_row_base + d01 + 1u] = output_accumulator1[1] * inverse_sum;
        }

        const uint d10 = kBD + simd_id * kFrag + pair_offset;
        if (d10 < D) {
            out[output_row_base + d10] = output_accumulator2[0] * inverse_sum;
        }
        if (d10 + 1u < D) {
            out[output_row_base + d10 + 1u] = output_accumulator2[1] * inverse_sum;
        }
        const uint d11 = kBD + (simd_id + kSimdGroups) * kFrag + pair_offset;
        if (d11 < D) {
            out[output_row_base + d11] = output_accumulator3[0] * inverse_sum;
        }
        if (d11 + 1u < D) {
            out[output_row_base + d11 + 1u] = output_accumulator3[1] * inverse_sum;
        }
    }
}

kernel void fused_fa_from_qkv_d64(
    device const float* q_cache     [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device float* lse_out           [[buffer(4)]],
    constant uint& B                [[buffer(5)]],
    constant uint& H                [[buffer(6)]],
    constant uint& T                [[buffer(7)]],
    constant uint& C                [[buffer(8)]],
    constant uint& is_causal_int    [[buffer(9)]],
    constant uint& write_lse        [[buffer(10)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]])
{
    constexpr uint kBQ         = 8u;
    constexpr uint kBK         = 32u;
    constexpr uint kD          = 64u;          // tile 宽度，编译期固定
    constexpr uint kFrag       = 8u;
    constexpr uint kThreads    = 128u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kQStride    = kD + 1u;      // 65
    constexpr uint kKTStride   = kFrag + 1u;   // 9
    constexpr uint kPStride    = kBK + 1u;     // 33
    constexpr uint kKTGroup    = kD * kKTStride;   // 576
    constexpr uint kVSub       = kBK / kFrag;      // 4

    threadgroup float Q_tg[kBQ][kQStride];                  // 520 f
    threadgroup float KT_all[kSimdGroups * kKTGroup];       // 2304 f
    threadgroup float P_tg[kBQ][kPStride];                  // 264 f
    threadgroup float running_max_tg[kBQ];
    threadgroup float running_sum_tg[kBQ];
    threadgroup float alpha_tg[kBQ];

    const uint tid = thread_pos.x;

    // 统一退出条件，避免 barrier 死锁
    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u ||
        simd_lane_count != 32u || tg_pos.x != 0u ||
        B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;                       // 真实 head_dim
    if (D == 0u || D > kD) return;

    const uint bh = tg_pos.z;
    if (bh >= B * H) return;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) return;

    const bool  is_causal   = (is_causal_int != 0u);
    const bool  d_full      = (D == kD);        // 整个 threadgroup 一致的 uniform 分支
    const float score_scale = rsqrt(float(D));  // 必须用真实 D

    // D=64 只需要两组 accumulator：d = simd_id*8+x 与 (simd_id+4)*8+x
    float2 acc0 = float2(0.0f);
    float2 acc1 = float2(0.0f);

    if (tid < kBQ) {
        running_max_tg[tid] = -INFINITY;
        running_sum_tg[tid] = 0.0f;
        alpha_tg[tid]       = 0.0f;
    }

    // ---------------- Q：整个 kernel 只加载一次 ----------------
    // 8 行 × 16 个 float4 = 128 = kThreads，每线程恰好一个 float4
    {
        const uint local_q = tid >> 4;            // 0..7
        const uint d0      = (tid & 15u) << 2;    // 0,4,...,60
        const uint query   = query_base + local_q;
        float4 value = float4(0.0f);
        if (query < T) {
            device const float* row = q_cache + (bh * T + query) * D;
            if (d_full) {
                // fast path：无任何 D 边界判断
                value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
            } else if (d0 + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
            } else {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    if (d0 + c < D) value[c] = row[d0 + c];
                }
            }
        }
        Q_tg[local_q][d0 + 0u] = value.x;
        Q_tg[local_q][d0 + 1u] = value.y;
        Q_tg[local_q][d0 + 2u] = value.z;
        Q_tg[local_q][d0 + 3u] = value.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* KT_tg = KT_all + simd_id * kKTGroup;   // 每个 simdgroup 私有
    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;

    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {

        // 完整 tile / 尾 tile 判定（threadgroup 内一致）
        const bool tile_full     = (key_base + kBK) <= T;
        const bool tile_nocausal = (!is_causal) || (key_base + kBK <= query_base + 1u);
        const bool tile_fast     = tile_full && tile_nocausal;

        // ---------------- K 加载 + 转置：KT_tg[d][key] ----------------
        // 8 key × 16 float4 = 128 元素，32 lane × 4 次
        if (d_full && tile_full) {
            // fast path：既无 D 判断，也无 key<T 判断
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                const uint index     = lane + i * 32u;         // 0..127
                const uint local_key = index >> 4;             // 0..7
                const uint d0        = (index & 15u) << 2;     // 0..60
                const uint key       = key_base + simd_id * kFrag + local_key;
                const float4 value = float4(*reinterpret_cast<device const packed_float4*>(
                        k_cache + (bh * T + key) * kD + d0));
                KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                const uint index     = lane + i * 32u;
                const uint local_key = index >> 4;
                const uint d0        = (index & 15u) << 2;
                const uint key       = key_base + simd_id * kFrag + local_key;
                float4 value = float4(0.0f);
                if (key < T) {
                    device const float* row = k_cache + (bh * T + key) * D;
                    if (d0 + 3u < D) {
                        value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
                    } else if (d0 < D) {
                        #pragma unroll
                        for (uint c = 0u; c < 4u; ++c) {
                            if (d0 + c < D) value[c] = row[d0 + c];
                        }
                    }
                }
                KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // KT_tg 为 simdgroup 私有

        // ---------------- QK^T ----------------
        float2 score = float2(0.0f);
        #pragma unroll
        for (uint d_sub = 0u; d_sub < kD; d_sub += kFrag) {
            const float2 q_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(&Q_tg[0][0]),
                    kQStride, ushort2(ushort(0u), ushort(d_sub)), ushort(lane));
            const float2 k_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(KT_tg),
                    kKTStride, ushort2(ushort(d_sub), ushort(0u)), ushort(lane));
            score = lowreg_matrix::mma_8x8(q_frag, k_frag, score);
        }

        // ---------------- 提前发射 V 的 device load，掩盖延迟 ----------------
        const ushort2 v_off  = morton_order(ushort(lane));
        const uint    v_row  = uint(v_off.y);
        const uint    v_col0 = simd_id * kFrag + uint(v_off.x);
        const uint    v_col1 = (simd_id + kSimdGroups) * kFrag + uint(v_off.x);
        float2 v_frag0[kVSub];
        float2 v_frag1[kVSub];
        if (d_full && tile_full) {
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                const uint key = key_base + v_row + sub * kFrag;
                device const float* row = v_cache + (bh * T + key) * kD;
                v_frag0[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                v_frag1[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
            }
        } else {
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                float2 f0 = float2(0.0f);
                float2 f1 = float2(0.0f);
                const uint key = key_base + v_row + sub * kFrag;
                if (key < T) {
                    device const float* row = v_cache + (bh * T + key) * D;
                    if (v_col0 + 1u < D) {
                        f0 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                    } else if (v_col0 < D) {
                        f0[0] = row[v_col0];
                    }
                    if (v_col1 + 1u < D) {
                        f1 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                    } else if (v_col1 < D) {
                        f1[0] = row[v_col1];
                    }
                }
                v_frag0[sub] = f0;
                v_frag1[sub] = f1;
            }
        }

        // ---------------- scale + mask 写入 P ----------------
        const ushort2 s_off = morton_order(ushort(lane));
        const uint local_q  = uint(s_off.y);
        const uint local_k0 = simd_id * kFrag + uint(s_off.x);
        if (tile_fast) {
            // fast path：无 causal 判断、无 key<T 判断
            // （query >= T 的行 Q 已填 0，score=0，不影响任何被写出的行）
            P_tg[local_q][local_k0]      = score[0] * score_scale;
            P_tg[local_q][local_k0 + 1u] = score[1] * score_scale;
        } else {
            const uint query = query_base + local_q;
            const uint key0  = key_base + local_k0;
            const uint key1  = key0 + 1u;
            const bool ok0 = (query < T) && (key0 < T) && (!is_causal || key0 <= query);
            const bool ok1 = (query < T) && (key1 < T) && (!is_causal || key1 <= query);
            P_tg[local_q][local_k0]      = ok0 ? score[0] * score_scale : -INFINITY;
            P_tg[local_q][local_k0 + 1u] = ok1 ? score[1] * score_scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 32-key 在线 softmax ----------------
        const uint row_half     = lane >> 4u;
        const uint half_lane    = lane & 15u;
        const uint softmax_row  = simd_id * 2u + row_half;
        const uint p_col0       = half_lane;
        const uint p_col1       = half_lane + 16u;
        const float score0 = P_tg[softmax_row][p_col0];
        const float score1 = P_tg[softmax_row][p_col1];

        float local_max = max(score0, score1);
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(8u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(4u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(2u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(1u)));
        const ushort leader_lane = ushort(row_half * 16u);
        const float tile_max = simd_shuffle(local_max, leader_lane);

        const float old_max   = running_max_tg[softmax_row];
        const float new_max   = max(old_max, tile_max);
        const float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        const float prob0     = isfinite(score0) ? exp(score0 - new_max) : 0.0f;
        const float prob1     = isfinite(score1) ? exp(score1 - new_max) : 0.0f;

        float sum = prob0 + prob1;
        sum += simd_shuffle_down(sum, ushort(8u));
        sum += simd_shuffle_down(sum, ushort(4u));
        sum += simd_shuffle_down(sum, ushort(2u));
        sum += simd_shuffle_down(sum, ushort(1u));
        const float tile_sum = simd_shuffle(sum, leader_lane);

        P_tg[softmax_row][p_col0] = prob0;
        P_tg[softmax_row][p_col1] = prob1;
        if (half_lane == 0u) {
            running_max_tg[softmax_row] = new_max;
            running_sum_tg[softmax_row] = running_sum_tg[softmax_row] * old_scale + tile_sum;
            alpha_tg[softmax_row]       = old_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 旧 numerator 缩放 ----------------
        const float rescale = alpha_tg[uint(morton_order(ushort(lane)).y)];
        acc0 *= rescale;
        acc1 *= rescale;

        // ---------------- P × V ----------------
        #pragma unroll
        for (uint sub = 0u; sub < kVSub; ++sub) {
            const float2 p_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(&P_tg[0][0]),
                    kPStride, ushort2(ushort(0u), ushort(sub * kFrag)), ushort(lane));
            acc0 = lowreg_matrix::mma_8x8(p_frag, v_frag0[sub], acc0);
            acc1 = lowreg_matrix::mma_8x8(p_frag, v_frag1[sub], acc1);
        }

        // 下一个 key tile 会覆写 P_tg，所有 simdgroup 必须先读完
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------------- 归一化并写出 [B,T,C] ----------------
    const ushort2 f_off = morton_order(ushort(lane));
    const uint f_lq  = uint(f_off.y);
    const uint f_q   = query_base + f_lq;
    if (f_q < T) {
        const float denom = running_sum_tg[f_lq];
        const float inv   = denom > 0.0f ? 1.0f / denom : 0.0f;
        const uint  x     = uint(f_off.x);

        if (write_lse != 0u && simd_id == 0u && x == 0u) {
            lse_out[bh * T + f_q] = denom > 0.0f ? running_max_tg[f_lq] + log(denom) : 0.0f;
        }

        device float* orow = out + (b * T + f_q) * C + h * D;
        const uint d0 = simd_id * kFrag + x;
        const uint d1 = (simd_id + kSimdGroups) * kFrag + x;
        if (d0        < D) orow[d0]        = acc0[0] * inv;
        if (d0 + 1u   < D) orow[d0 + 1u]   = acc0[1] * inv;
        if (d1        < D) orow[d1]        = acc1[0] * inv;
        if (d1 + 1u   < D) orow[d1 + 1u]   = acc1[1] * inv;
    }
}

// =====================================================================
//  64 < D <= 128
// =====================================================================
kernel void fused_fa_from_qkv_d128(
    device const float* q_cache     [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device float* lse_out           [[buffer(4)]],
    constant uint& B                [[buffer(5)]],
    constant uint& H                [[buffer(6)]],
    constant uint& T                [[buffer(7)]],
    constant uint& C                [[buffer(8)]],
    constant uint& is_causal_int    [[buffer(9)]],
    constant uint& write_lse        [[buffer(10)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]])
{
    constexpr uint kBQ         = 8u;
    constexpr uint kBK         = 32u;
    constexpr uint kBD         = 64u;          // 单个 d 子块
    constexpr uint kD          = 128u;         // tile 总宽度
    constexpr uint kChunks     = kD / kBD;     // 2
    constexpr uint kFrag       = 8u;
    constexpr uint kThreads    = 128u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kQStride    = kBD + 1u;     // 65
    constexpr uint kKTStride   = kFrag + 1u;   // 9
    constexpr uint kPStride    = kBK + 1u;     // 33
    constexpr uint kKTGroup    = kBD * kKTStride;
    constexpr uint kVSub       = kBK / kFrag;  // 4

    threadgroup float Q_tg[kChunks][kBQ][kQStride];          // 1040 f
    threadgroup float KT_all[kSimdGroups * kKTGroup];        // 2304 f
    threadgroup float P_tg[kBQ][kPStride];
    threadgroup float running_max_tg[kBQ];
    threadgroup float running_sum_tg[kBQ];
    threadgroup float alpha_tg[kBQ];

    const uint tid = thread_pos.x;

    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u ||
        simd_lane_count != 32u || tg_pos.x != 0u ||
        B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D <= kBD || D > kD) return;          // 该 kernel 只处理 64 < D <= 128

    const uint bh = tg_pos.z;
    if (bh >= B * H) return;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) return;

    const bool  is_causal   = (is_causal_int != 0u);
    const bool  d_full      = (D == kD);
    const float score_scale = rsqrt(float(D));

    float2 acc[2u * kChunks];
    #pragma unroll
    for (uint i = 0u; i < 2u * kChunks; ++i) acc[i] = float2(0.0f);

    if (tid < kBQ) {
        running_max_tg[tid] = -INFINITY;
        running_sum_tg[tid] = 0.0f;
        alpha_tg[tid]       = 0.0f;
    }

    // ---------------- Q：加载一次，2 chunk × 8 行 × 16 float4 = 256 ----------------
    #pragma unroll
    for (uint it = 0u; it < 2u; ++it) {
        const uint index   = tid + it * kThreads;   // 0..255
        const uint chunk   = index >> 7;            // 0 / 1
        const uint rem     = index & 127u;
        const uint local_q = rem >> 4;
        const uint d0      = (rem & 15u) << 2;
        const uint gd      = chunk * kBD + d0;
        const uint query   = query_base + local_q;
        float4 value = float4(0.0f);
        if (query < T) {
            device const float* row = q_cache + (bh * T + query) * D;
            if (d_full) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
            } else if (gd + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
            } else if (gd < D) {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    if (gd + c < D) value[c] = row[gd + c];
                }
            }
        }
        Q_tg[chunk][local_q][d0 + 0u] = value.x;
        Q_tg[chunk][local_q][d0 + 1u] = value.y;
        Q_tg[chunk][local_q][d0 + 2u] = value.z;
        Q_tg[chunk][local_q][d0 + 3u] = value.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* KT_tg = KT_all + simd_id * kKTGroup;
    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;

    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {

        const bool tile_full     = (key_base + kBK) <= T;
        const bool tile_nocausal = (!is_causal) || (key_base + kBK <= query_base + 1u);
        const bool tile_fast     = tile_full && tile_nocausal;

        // ---------------- QK^T：两个 d chunk 复用同一块 KT ----------------
        float2 score = float2(0.0f);
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d_base = chunk * kBD;

            if (chunk != 0u) {
                simdgroup_barrier(mem_flags::mem_threadgroup);   // 复用 KT_tg 前
            }

            if (d_full && tile_full) {
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    const uint index     = lane + i * 32u;
                    const uint local_key = index >> 4;
                    const uint d0        = (index & 15u) << 2;
                    const uint key       = key_base + simd_id * kFrag + local_key;
                    const float4 value = float4(*reinterpret_cast<device const packed_float4*>(
                            k_cache + (bh * T + key) * kD + d_base + d0));
                    KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                    KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                    KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                    KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
                }
            } else {
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    const uint index     = lane + i * 32u;
                    const uint local_key = index >> 4;
                    const uint d0        = (index & 15u) << 2;
                    const uint gd        = d_base + d0;
                    const uint key       = key_base + simd_id * kFrag + local_key;
                    float4 value = float4(0.0f);
                    if (key < T) {
                        device const float* row = k_cache + (bh * T + key) * D;
                        if (gd + 3u < D) {
                            value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
                        } else if (gd < D) {
                            #pragma unroll
                            for (uint c = 0u; c < 4u; ++c) {
                                if (gd + c < D) value[c] = row[gd + c];
                            }
                        }
                    }
                    KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                    KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                    KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                    KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint d_sub = 0u; d_sub < kBD; d_sub += kFrag) {
                const float2 q_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(&Q_tg[chunk][0][0]),
                        kQStride, ushort2(ushort(0u), ushort(d_sub)), ushort(lane));
                const float2 k_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(KT_tg),
                        kKTStride, ushort2(ushort(d_sub), ushort(0u)), ushort(lane));
                score = lowreg_matrix::mma_8x8(q_frag, k_frag, score);
            }
        }

        // ---------------- scale + mask ----------------
        const ushort2 s_off = morton_order(ushort(lane));
        const uint local_q  = uint(s_off.y);
        const uint local_k0 = simd_id * kFrag + uint(s_off.x);
        if (tile_fast) {
            P_tg[local_q][local_k0]      = score[0] * score_scale;
            P_tg[local_q][local_k0 + 1u] = score[1] * score_scale;
        } else {
            const uint query = query_base + local_q;
            const uint key0  = key_base + local_k0;
            const uint key1  = key0 + 1u;
            const bool ok0 = (query < T) && (key0 < T) && (!is_causal || key0 <= query);
            const bool ok1 = (query < T) && (key1 < T) && (!is_causal || key1 <= query);
            P_tg[local_q][local_k0]      = ok0 ? score[0] * score_scale : -INFINITY;
            P_tg[local_q][local_k0 + 1u] = ok1 ? score[1] * score_scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 在线 softmax ----------------
        const uint row_half    = lane >> 4u;
        const uint half_lane   = lane & 15u;
        const uint softmax_row = simd_id * 2u + row_half;
        const float score0 = P_tg[softmax_row][half_lane];
        const float score1 = P_tg[softmax_row][half_lane + 16u];

        float local_max = max(score0, score1);
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(8u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(4u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(2u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(1u)));
        const ushort leader_lane = ushort(row_half * 16u);
        const float tile_max = simd_shuffle(local_max, leader_lane);

        const float old_max   = running_max_tg[softmax_row];
        const float new_max   = max(old_max, tile_max);
        const float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        const float prob0     = isfinite(score0) ? exp(score0 - new_max) : 0.0f;
        const float prob1     = isfinite(score1) ? exp(score1 - new_max) : 0.0f;

        float sum = prob0 + prob1;
        sum += simd_shuffle_down(sum, ushort(8u));
        sum += simd_shuffle_down(sum, ushort(4u));
        sum += simd_shuffle_down(sum, ushort(2u));
        sum += simd_shuffle_down(sum, ushort(1u));
        const float tile_sum = simd_shuffle(sum, leader_lane);

        P_tg[softmax_row][half_lane]       = prob0;
        P_tg[softmax_row][half_lane + 16u] = prob1;
        if (half_lane == 0u) {
            running_max_tg[softmax_row] = new_max;
            running_sum_tg[softmax_row] = running_sum_tg[softmax_row] * old_scale + tile_sum;
            alpha_tg[softmax_row]       = old_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rescale = alpha_tg[uint(morton_order(ushort(lane)).y)];
        #pragma unroll
        for (uint i = 0u; i < 2u * kChunks; ++i) acc[i] *= rescale;

        // ---------------- P × V ----------------
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d_base = chunk * kBD;
            const ushort2 v_off = morton_order(ushort(lane));
            const uint v_row  = uint(v_off.y);
            const uint v_col0 = d_base + simd_id * kFrag + uint(v_off.x);
            const uint v_col1 = d_base + (simd_id + kSimdGroups) * kFrag + uint(v_off.x);

            float2 v_frag0[kVSub];
            float2 v_frag1[kVSub];
            if (d_full && tile_full) {
                #pragma unroll
                for (uint sub = 0u; sub < kVSub; ++sub) {
                    const uint key = key_base + v_row + sub * kFrag;
                    device const float* row = v_cache + (bh * T + key) * kD;
                    v_frag0[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                    v_frag1[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                }
            } else {
                #pragma unroll
                for (uint sub = 0u; sub < kVSub; ++sub) {
                    float2 f0 = float2(0.0f);
                    float2 f1 = float2(0.0f);
                    const uint key = key_base + v_row + sub * kFrag;
                    if (key < T) {
                        device const float* row = v_cache + (bh * T + key) * D;
                        if (v_col0 + 1u < D)      f0 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                        else if (v_col0 < D)      f0[0] = row[v_col0];
                        if (v_col1 + 1u < D)      f1 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                        else if (v_col1 < D)      f1[0] = row[v_col1];
                    }
                    v_frag0[sub] = f0;
                    v_frag1[sub] = f1;
                }
            }

            float2 a0 = acc[chunk * 2u + 0u];
            float2 a1 = acc[chunk * 2u + 1u];
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                const float2 p_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(&P_tg[0][0]),
                        kPStride, ushort2(ushort(0u), ushort(sub * kFrag)), ushort(lane));
                a0 = lowreg_matrix::mma_8x8(p_frag, v_frag0[sub], a0);
                a1 = lowreg_matrix::mma_8x8(p_frag, v_frag1[sub], a1);
            }
            acc[chunk * 2u + 0u] = a0;
            acc[chunk * 2u + 1u] = a1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------------- 写出 ----------------
    const ushort2 f_off = morton_order(ushort(lane));
    const uint f_lq = uint(f_off.y);
    const uint f_q  = query_base + f_lq;
    if (f_q < T) {
        const float denom = running_sum_tg[f_lq];
        const float inv   = denom > 0.0f ? 1.0f / denom : 0.0f;
        const uint  x     = uint(f_off.x);

        if (write_lse != 0u && simd_id == 0u && x == 0u) {
            lse_out[bh * T + f_q] = denom > 0.0f ? running_max_tg[f_lq] + log(denom) : 0.0f;
        }

        device float* orow = out + (b * T + f_q) * C + h * D;
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d0 = chunk * kBD + simd_id * kFrag + x;
            const uint d1 = chunk * kBD + (simd_id + kSimdGroups) * kFrag + x;
            const float2 a0 = acc[chunk * 2u + 0u];
            const float2 a1 = acc[chunk * 2u + 1u];
            if (d0      < D) orow[d0]      = a0[0] * inv;
            if (d0 + 1u < D) orow[d0 + 1u] = a0[1] * inv;
            if (d1      < D) orow[d1]      = a1[0] * inv;
            if (d1 + 1u < D) orow[d1 + 1u] = a1[1] * inv;
        }
    }
}

template <uint kMaxD, uint kVectorsPerLane>
METAL_FUNC static void fused_fa_from_qkv_blocked_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device float* out,
    device float* lse_out,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint write_lse,
    uint3 tg_pos,
    uint lane
) {
    if (tg_pos.x != 0u || B == 0u || H == 0u ||
        T == 0u || C == 0u || C % H != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > kMaxD) {
        return;
    }
    const uint query = tg_pos.y;
    const uint bh = tg_pos.z;
    if (query >= T || bh >= B * H) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const bool is_causal = is_causal_int != 0u;
    const float scale = rsqrt(float(D));
    const uint cache_head_base = bh * T * D;
    const uint q_base = cache_head_base + query * D;
    const uint out_base = (b * T + query) * C + h * D;
    const uint vector_count = D / 2u;
    const uint vector_end = vector_count * 2u;
    const bool tail_lane = (D & 1u) != 0u && lane == 0u;
    thread float2 q_register[kVectorsPerLane];
    thread float2 output_accumulator[kVectorsPerLane];
    float q_tail = 0.0f;
    float output_tail = 0.0f;

    #pragma unroll
    for (uint slot = 0u; slot < kVectorsPerLane; ++slot) {
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index * 2u;
            q_register[slot] = float2(*reinterpret_cast<device const packed_float2*>(q_cache + q_base + d));
        } else {
            q_register[slot] = float2(0.0f);
        }
        output_accumulator[slot] = float2(0.0f);
    }

    if (tail_lane) {
        q_tail = q_cache[q_base + vector_end];
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint key_limit = is_causal ? min(T, query + 1u) : T;

    for (uint key = 0u; key < key_limit; ++key) {
        const uint key_base = cache_head_base + key * D;
        float local_score = 0.0f;

        #pragma unroll
        for (uint slot = 0u; slot < kVectorsPerLane; ++slot) {
            const uint vector_index = lane + slot * 32u;
            if (vector_index < vector_count) {
                const uint d = vector_index * 2u;
                const float2 k_value = float2(*reinterpret_cast<device const packed_float2*>(k_cache + key_base + d));
                local_score += dot(q_register[slot], k_value);
            }
        }

        if (tail_lane) {
            local_score += q_tail * k_cache[key_base + vector_end];
        }

        const float score = simd_sum(local_score) * scale;
        const float new_max = max(running_max, score);
        const float previous_scale = isfinite(running_max) ? exp(running_max - new_max) : 0.0f;
        const float probability = exp(score - new_max);
        running_sum = running_sum * previous_scale + probability;
        #pragma unroll
        for (uint slot = 0u; slot < kVectorsPerLane; ++slot) {
            const uint vector_index = lane + slot * 32u;
            if (vector_index < vector_count) {
                const uint d = vector_index * 2u;
                const float2 v_value = float2(*reinterpret_cast<device const packed_float2*>(v_cache + key_base + d));
                output_accumulator[slot] = fma(v_value, float2(probability), output_accumulator[slot] * float2(previous_scale));
            }
        }

        if (tail_lane) {
            output_tail = fma(probability, v_cache[key_base + vector_end], output_tail * previous_scale);
        }
        running_max = new_max;
    }

    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;

    #pragma unroll
    for (uint slot = 0u; slot < kVectorsPerLane; ++slot) {
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index * 2u;
            const float2 result = output_accumulator[slot] * float2(inverse_sum);
            *reinterpret_cast<device packed_float2*>(out + out_base + d) = packed_float2(result);
        }
    }

    if (tail_lane) {
        out[out_base + vector_end] = output_tail * inverse_sum;
    }
    if (write_lse != 0u && lane == 0u) {
        lse_out[bh * T + query] = running_sum > 0.0f ? running_max + log(running_sum) : 0.0f;
    }
}

kernel void fused_fa_bwd_prepare(
    device const float* o_btc        [[buffer(0)]],
    device const float* do_btc       [[buffer(1)]],
    device float* d_val              [[buffer(2)]],
    constant uint& B                 [[buffer(3)]],
    constant uint& H                 [[buffer(4)]],
    constant uint& T                 [[buffer(5)]],
    constant uint& C                 [[buffer(6)]],
    uint3 tg_pos                     [[threadgroup_position_in_grid]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kSimdWidth = 32u;
    constexpr uint kQueriesPerTG = 4u;
    constexpr uint kVectorWidth = 2u;
    constexpr uint kMaxHeadDim = 512u;
    constexpr uint kMaxVectorCount = kMaxHeadDim / kVectorWidth;
    constexpr uint kMaxVectorsPerLane = (kMaxVectorCount + kSimdWidth - 1u) / kSimdWidth;

    if (tg_pos.x != 0u || B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }

    const uint query = tg_pos.y * kQueriesPerTG + simd_id;
    const uint bh = tg_pos.z;
    if (query >= T || bh >= B * H) {
        return;
    }

    const uint D = C / H;
    if (D == 0u || D > kMaxHeadDim) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint output_base = (b * T + query) * C + h * D;
    const uint stats_index = bh * T + query;
    const uint vector_count = D / kVectorWidth;
    const uint vector_end = vector_count * kVectorWidth;
    const bool tail_lane = (vector_end != D) && lane == 0u;

    // D_i = dot(O_i, dO_i)
    float local_di = 0.0f;
    #pragma unroll
    for (uint slot = 0u; slot < kMaxVectorsPerLane; ++slot) {
        const uint vector_index = lane + slot * kSimdWidth;
        if (vector_index < vector_count) {
            const uint d = vector_index * kVectorWidth;
            const float2 o_value = float2(*reinterpret_cast<device const packed_float2*>(o_btc + output_base + d));
            const float2 do_value = float2(*reinterpret_cast<device const packed_float2*>(do_btc + output_base + d));
            local_di += dot(o_value, do_value);
        }
    }

    if (tail_lane) {
        local_di += o_btc[output_base + vector_end] * do_btc[output_base + vector_end];
    }

    const float di = simd_sum(local_di);
    if (lane == 0u) {
        d_val[stats_index] = di;
    }
}

template <uint kSimdGroups, uint kBQ, uint kBK, uint kDTile>
METAL_FUNC static void fused_fa_bwd_dq_tiled64_unified_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dQ,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint3 thread_pos,
    uint3 threads_per_tg,
    uint lane,
    uint sg,
    uint simd_lane_count,
    threadgroup float* Ks,
    threadgroup float* Vs
) {
    constexpr uint kFrag = 8u;
    constexpr uint kKVStride = kDTile + 4u;
    constexpr uint kKSub = kBK / kFrag;
    constexpr uint kDSub = kDTile / kFrag;
    constexpr float kLog2E = 1.4426950408889634f;

    const uint expected_threads = kSimdGroups * 32u;
    if (threads_per_tg.x != expected_threads || threads_per_tg.y != 1u ||
        threads_per_tg.z != 1u || simd_lane_count != 32u ||
        B == 0u || H == 0u || T == 0u || C == 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    const uint bh = tg_pos.z;
    if (D == 0u || D > 128u || bh >= B * H) {
        return;
    }

    const uint d_chunks = (D + kDTile - 1u) / kDTile;
    const uint output_chunk = tg_pos.x;
    if (output_chunk >= d_chunks) {
        return;
    }

    const uint query_block = tg_pos.y * kBQ;
    if (query_block >= T) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint stat_base = bh * T;
    const bool is_causal = is_causal_int != 0u;
    const uint key_limit = is_causal ? min(T, query_block + kBQ) : T;
    const float scale = rsqrt(float(D));
    const float scale_log2e = scale * kLog2E;
    const uint query_frag_base = query_block + sg * kFrag;
    const ushort2 frag_offset = morton_order(ushort(lane));

    thread float2 dq_accum[kDSub];
    #pragma unroll
    for (uint i = 0u; i < kDSub; ++i) {
        dq_accum[i] = float2(0.0f);
    }

    for (uint key_start = 0u; key_start < key_limit; key_start += kBK) {
        thread float2 score_frag[kKSub];
        thread float2 dp_frag[kKSub];
        #pragma unroll
        for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
            score_frag[key_sub] = float2(0.0f);
            dp_frag[key_sub] = float2(0.0f);
        }

        // Visit every D tile exactly once, but schedule this threadgroup's
        // output tile last.  That leaves its K tile resident for dS*K below.
        for (uint pass = 0u; pass < d_chunks; ++pass) {
            const uint d_chunk = pass + (pass >= output_chunk ? 1u : 0u);
            const uint scheduled_chunk = pass + 1u == d_chunks
                ? output_chunk : d_chunk;
            const uint d_base = scheduled_chunk * kDTile;
            const uint pair_count = kBK * (kDTile / 2u);

            for (uint pair = thread_pos.x; pair < pair_count;
                 pair += expected_threads) {
                const uint local_key = pair / (kDTile / 2u);
                const uint local_d = (pair - local_key * (kDTile / 2u)) << 1u;
                const uint key = key_start + local_key;
                const uint d = d_base + local_d;
                float2 kv = float2(0.0f);
                float2 vv = float2(0.0f);
                if (key < key_limit && d + 1u < D) {
                    const uint src = head_base + key * D + d;
                    kv = float2(*reinterpret_cast<device const packed_float2*>(
                        k_cache + src));
                    vv = float2(*reinterpret_cast<device const packed_float2*>(
                        v_cache + src));
                } else if (key < key_limit && d < D) {
                    const uint src = head_base + key * D + d;
                    kv[0] = k_cache[src];
                    vv[0] = v_cache[src];
                }
                *reinterpret_cast<threadgroup packed_float2*>(
                    Ks + local_key * kKVStride + local_d) = packed_float2(kv);
                *reinterpret_cast<threadgroup packed_float2*>(
                    Vs + local_key * kKVStride + local_d) = packed_float2(vv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint d_index = 0u; d_index < kDSub; ++d_index) {
                const uint global_d = d_base + d_index * kFrag;
                const uint q_row = query_frag_base + uint(frag_offset.y);
                const uint q_col = global_d + uint(frag_offset.x);
                float2 q_frag = float2(0.0f);
                float2 do_frag = float2(0.0f);
                if (q_row < T && q_col + 1u < D) {
                    q_frag = float2(*reinterpret_cast<device const packed_float2*>(
                        q_cache + head_base + q_row * D + q_col));
                    do_frag = float2(*reinterpret_cast<device const packed_float2*>(
                        dO + (b * T + q_row) * C + h * D + q_col));
                } else if (q_row < T && q_col < D) {
                    q_frag[0] = q_cache[head_base + q_row * D + q_col];
                    do_frag[0] = dO[(b * T + q_row) * C + h * D + q_col];
                }

                #pragma unroll
                for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
                    const ushort2 origin = ushort2(
                        ushort(d_index * kFrag), ushort(key_sub * kFrag));
                    const float2 k_transposed = lowreg_matrix::load_tg_8x8(
                        Ks, kKVStride, origin, ushort(lane), true);
                    const float2 v_transposed = lowreg_matrix::load_tg_8x8(
                        Vs, kKVStride, origin, ushort(lane), true);
                    score_frag[key_sub] = lowreg_matrix::mma_8x8(
                        q_frag, k_transposed, score_frag[key_sub]);
                    dp_frag[key_sub] = lowreg_matrix::mma_8x8(
                        do_frag, v_transposed, dp_frag[key_sub]);
                }
            }

            // All SIMD groups must finish reading this tile before another
            // pass overwrites it.  No barrier is needed after the last pass:
            // that tile is intentionally retained for the output MMA.
            if (pass + 1u < d_chunks) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        const uint query = query_frag_base + uint(frag_offset.y);
        const bool query_valid = query < T;
        const float row_lse_log2 = query_valid
            ? lse[stat_base + query] * kLog2E : 0.0f;
        const float row_d = query_valid ? d_val[stat_base + query] : 0.0f;

        #pragma unroll
        for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
            const uint key0 = key_start + key_sub * kFrag + uint(frag_offset.x);
            const uint key1 = key0 + 1u;
            const bool valid0 = query_valid && key0 < key_limit &&
                (!is_causal || key0 <= query);
            const bool valid1 = query_valid && key1 < key_limit &&
                (!is_causal || key1 <= query);
            const float p0 = valid0
                ? exp2(score_frag[key_sub][0] * scale_log2e - row_lse_log2)
                : 0.0f;
            const float p1 = valid1
                ? exp2(score_frag[key_sub][1] * scale_log2e - row_lse_log2)
                : 0.0f;
            score_frag[key_sub] = float2(
                scale * p0 * (dp_frag[key_sub][0] - row_d),
                scale * p1 * (dp_frag[key_sub][1] - row_d));
        }

        #pragma unroll
        for (uint d_index = 0u; d_index < kDSub; ++d_index) {
            #pragma unroll
            for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
                const ushort2 origin = ushort2(
                    ushort(key_sub * kFrag), ushort(d_index * kFrag));
                const float2 k_frag = lowreg_matrix::load_tg_8x8(
                    Ks, kKVStride, origin, ushort(lane));
                dq_accum[d_index] = lowreg_matrix::mma_8x8(
                    score_frag[key_sub], k_frag, dq_accum[d_index]);
            }
        }

        if (key_start + kBK < key_limit) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint query = query_frag_base + uint(frag_offset.y);
    if (query < T) {
        #pragma unroll
        for (uint d_index = 0u; d_index < kDSub; ++d_index) {
            const uint d_col = output_chunk * kDTile + d_index * kFrag
                + uint(frag_offset.x);
            if (d_col >= D) {
                continue;
            }
            const uint out_base = (b * T + query) * C + h * D + d_col;
            if (d_col + 1u < D) {
                *reinterpret_cast<device packed_float2*>(dQ + out_base) =
                    packed_float2(dq_accum[d_index]);
            } else {
                dQ[out_base] = dq_accum[d_index][0];
            }
        }
    }
}

#define DECLARE_FA_DQ_UNIFIED_COMPAT_KERNEL(NAME)                              \
kernel void NAME(                                                              \
    device const float* q [[buffer(0)]],                                       \
    device const float* k [[buffer(1)]],                                       \
    device const float* v [[buffer(2)]],                                       \
    device const float* dO [[buffer(3)]],                                      \
    device float* dQ [[buffer(4)]],                                            \
    device const float* lse [[buffer(5)]],                                     \
    device const float* d_val [[buffer(6)]],                                   \
    constant uint& B [[buffer(7)]],                                            \
    constant uint& H [[buffer(8)]],                                            \
    constant uint& T [[buffer(9)]],                                            \
    constant uint& C [[buffer(10)]],                                           \
    constant uint& causal [[buffer(11)]],                                      \
    uint3 tg_pos [[threadgroup_position_in_grid]],                             \
    uint3 thread_pos [[thread_position_in_threadgroup]],                       \
    uint3 threads_per_tg [[threads_per_threadgroup]],                          \
    uint lane [[thread_index_in_simdgroup]],                                   \
    uint sg [[simdgroup_index_in_threadgroup]],                                \
    uint simd_lane_count [[threads_per_simdgroup]]) {                          \
    constexpr uint kKVStride = 68u;                                            \
    threadgroup float Ks[32u * kKVStride];                                     \
    threadgroup float Vs[32u * kKVStride];                                     \
    fused_fa_bwd_dq_tiled64_unified_impl<4u, 32u, 32u, 64u>(                  \
        q, k, v, dO, dQ, lse, d_val, B, H, T, C, causal,                      \
        tg_pos, thread_pos, threads_per_tg, lane, sg, simd_lane_count,         \
        Ks, Vs);                                                               \
}

// Compatibility names are intentionally preserved for the supported range.
// Both aliases call the same implementation; D>128 is rejected above.
DECLARE_FA_DQ_UNIFIED_COMPAT_KERNEL(fused_fa_bwd_dq_block64)
DECLARE_FA_DQ_UNIFIED_COMPAT_KERNEL(fused_fa_bwd_dq_block128)

#undef DECLARE_FA_DQ_UNIFIED_COMPAT_KERNEL

template <uint kMaxD, uint kSlots>
METAL_FUNC static void fused_fa_bwd_dkv_large_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dK,
    device float* dV,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint lane
) {
    if (tg_pos.x != 0u || B == 0u || H == 0u ||
        T == 0u || C == 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    if (D == 0u || D > kMaxD) {
        return;
    }

    const uint key = tg_pos.y;
    const uint bh = tg_pos.z;
    if (key >= T || bh >= B * H) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint cache_head_base = bh * T * D;
    const uint kv_base = cache_head_base + key * D;
    const uint output_base = (b * T + key) * C + h * D;
    const bool is_causal = is_causal_int != 0u;
    const uint query_begin = is_causal ? key : 0u;
    const float scale = rsqrt(float(D));

    thread float dk_accum[kSlots];
    thread float dv_accum[kSlots];

    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        dk_accum[slot] = 0.0f;
        dv_accum[slot] = 0.0f;
    }

    for (uint query = query_begin; query < T; ++query) {
        const uint q_base = cache_head_base + query * D;
        const uint do_base = (b * T + query) * C + h * D;
        const uint stat_index = bh * T + query;

        float local_score = 0.0f;
        float local_dp = 0.0f;

        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            const uint d = lane + slot * 32u;
            if (d < D) {
                local_score = fma(
                    q_cache[q_base + d],
                    k_cache[kv_base + d],
                    local_score
                );
                local_dp = fma(
                    dO[do_base + d],
                    v_cache[kv_base + d],
                    local_dp
                );
            }
        }

        const float score = simd_sum(local_score) * scale;
        const float dp = simd_sum(local_dp);
        const float probability = exp(score - lse[stat_index]);
        const float ds =
            scale * probability * (dp - d_val[stat_index]);

        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            const uint d = lane + slot * 32u;
            if (d < D) {
                dk_accum[slot] = fma(
                    ds,
                    q_cache[q_base + d],
                    dk_accum[slot]
                );
                dv_accum[slot] = fma(
                    probability,
                    dO[do_base + d],
                    dv_accum[slot]
                );
            }
        }
    }

    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        const uint d = lane + slot * 32u;
        if (d < D) {
            dK[output_base + d] = dk_accum[slot];
            dV[output_base + d] = dv_accum[slot];
        }
    }
}

template <uint kSlots, uint kMaxD>
METAL_FUNC static void fused_fa_bwd_dkv_packed_even_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dK,
    device float* dV,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint lane
) {
    if (tg_pos.x != 0u || B == 0u || H == 0u || T == 0u ||
        C == 0u || C % H != 0u) {
        return;
    }
    const uint D = C / H;
    const uint key = tg_pos.y;
    const uint bh = tg_pos.z;
    if (D == 0u || D > kMaxD || (D & 1u) != 0u || key >= T || bh >= B * H) {
        return;
    }
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint kv_base = head_base + key * D;
    const uint out_base = (b * T + key) * C + h * D;
    const uint vector_count = D >> 1u;
    const uint query_begin = is_causal_int != 0u ? key : 0u;
    const float scale = rsqrt(float(D));

    thread float2 kv[kSlots];
    thread float2 vv[kSlots];
    thread float2 dkv[kSlots];
    thread float2 dvv[kSlots];
    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        kv[slot] = float2(0.0f);
        vv[slot] = float2(0.0f);
        dkv[slot] = float2(0.0f);
        dvv[slot] = float2(0.0f);
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index << 1u;
            kv[slot] = float2(*reinterpret_cast<device const packed_float2*>(k_cache + kv_base + d));
            vv[slot] = float2(*reinterpret_cast<device const packed_float2*>(v_cache + kv_base + d));
        }
    }
    for (uint query = query_begin; query < T; ++query) {
        const uint q_base = head_base + query * D;
        const uint do_base = (b * T + query) * C + h * D;
        float local_score = 0.0f;
        float local_dp = 0.0f;
        thread float2 qv[kSlots];
        thread float2 dov[kSlots];
        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            qv[slot] = float2(0.0f);
            dov[slot] = float2(0.0f);
            const uint vector_index = lane + slot * 32u;
            if (vector_index < vector_count) {
                const uint d = vector_index << 1u;
                qv[slot] = float2(*reinterpret_cast<device const packed_float2*>(q_cache + q_base + d));
                dov[slot] = float2(*reinterpret_cast<device const packed_float2*>(dO + do_base + d));
                local_score += dot(qv[slot], kv[slot]);
                local_dp += dot(dov[slot], vv[slot]);
            }
        }
        const float score = simd_sum(local_score) * scale;
        const float dp = simd_sum(local_dp);
        const uint stat_index = bh * T + query;
        const float probability = exp(score - lse[stat_index]);
        const float ds = scale * probability * (dp - d_val[stat_index]);
        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            dkv[slot] = fma(float2(ds), qv[slot], dkv[slot]);
            dvv[slot] = fma(float2(probability), dov[slot], dvv[slot]);
        }
    }
    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index << 1u;
            *reinterpret_cast<device packed_float2*>(dK + out_base + d) = packed_float2(dkv[slot]);
            *reinterpret_cast<device packed_float2*>(dV + out_base + d) = packed_float2(dvv[slot]);
        }
    }
}

template <uint kSlots, uint kMaxD>
METAL_FUNC static void fused_fa_bwd_dkv_packed_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dK,
    device float* dV,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint lane
) {
    if (tg_pos.x != 0u || B == 0u || H == 0u || T == 0u ||
        C == 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    const uint key = tg_pos.y;
    const uint bh = tg_pos.z;
    if (D == 0u || D > kMaxD || key >= T || bh >= B * H) {
        return;
    }
    if ((D & 1u) == 0u) {
        fused_fa_bwd_dkv_packed_even_impl<kSlots, kMaxD>(
            q_cache, k_cache, v_cache, dO, dK, dV, lse, d_val,
            B, H, T, C, is_causal_int, tg_pos, lane);
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint kv_base = head_base + key * D;
    const uint out_base = (b * T + key) * C + h * D;
    const uint vector_count = D >> 1u;
    const uint vector_end = vector_count << 1u;
    const bool tail_lane = vector_end != D && lane == 0u;
    const uint query_begin = is_causal_int != 0u ? key : 0u;
    const float scale = rsqrt(float(D));

    thread float2 kv[kSlots];
    thread float2 vv[kSlots];
    thread float2 dkv[kSlots];
    thread float2 dvv[kSlots];
    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        kv[slot] = float2(0.0f);
        vv[slot] = float2(0.0f);
        dkv[slot] = float2(0.0f);
        dvv[slot] = float2(0.0f);
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index << 1u;
            kv[slot] = float2(*reinterpret_cast<device const packed_float2*>(k_cache + kv_base + d));
            vv[slot] = float2(*reinterpret_cast<device const packed_float2*>(v_cache + kv_base + d));
        }
    }
    const float k_tail = tail_lane ? k_cache[kv_base + vector_end] : 0.0f;
    const float v_tail = tail_lane ? v_cache[kv_base + vector_end] : 0.0f;
    float dk_tail = 0.0f;
    float dv_tail = 0.0f;

    for (uint query = query_begin; query < T; ++query) {
        const uint q_base = head_base + query * D;
        const uint do_base = (b * T + query) * C + h * D;
        float local_score = 0.0f;
        float local_dp = 0.0f;
        thread float2 qv[kSlots];
        thread float2 dov[kSlots];
        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            qv[slot] = float2(0.0f);
            dov[slot] = float2(0.0f);
            const uint vector_index = lane + slot * 32u;
            if (vector_index < vector_count) {
                const uint d = vector_index << 1u;
                qv[slot] = float2(*reinterpret_cast<device const packed_float2*>(q_cache + q_base + d));
                dov[slot] = float2(*reinterpret_cast<device const packed_float2*>(dO + do_base + d));
                local_score += dot(qv[slot], kv[slot]);
                local_dp += dot(dov[slot], vv[slot]);
            }
        }
        float q_tail = 0.0f;
        float do_tail = 0.0f;
        if (tail_lane) {
            q_tail = q_cache[q_base + vector_end];
            do_tail = dO[do_base + vector_end];
            local_score = fma(q_tail, k_tail, local_score);
            local_dp = fma(do_tail, v_tail, local_dp);
        }
        const float score = simd_sum(local_score) * scale;
        const float dp = simd_sum(local_dp);
        const uint stat_index = bh * T + query;
        const float probability = exp(score - lse[stat_index]);
        const float ds = scale * probability * (dp - d_val[stat_index]);
        #pragma unroll
        for (uint slot = 0u; slot < kSlots; ++slot) {
            dkv[slot] = fma(float2(ds), qv[slot], dkv[slot]);
            dvv[slot] = fma(float2(probability), dov[slot], dvv[slot]);
        }
        if (tail_lane) {
            dk_tail = fma(ds, q_tail, dk_tail);
            dv_tail = fma(probability, do_tail, dv_tail);
        }
    }

    #pragma unroll
    for (uint slot = 0u; slot < kSlots; ++slot) {
        const uint vector_index = lane + slot * 32u;
        if (vector_index < vector_count) {
            const uint d = vector_index << 1u;
            *reinterpret_cast<device packed_float2*>(dK + out_base + d) = packed_float2(dkv[slot]);
            *reinterpret_cast<device packed_float2*>(dV + out_base + d) = packed_float2(dvv[slot]);
        }
    }
    if (tail_lane) {
        dK[out_base + vector_end] = dk_tail;
        dV[out_base + vector_end] = dv_tail;
    }
}

kernel void fused_fa_bwd_dkv_block128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* dO [[buffer(3)]],
    device float* dK [[buffer(4)]],
    device float* dV [[buffer(5)]],
    device const float* lse [[buffer(6)]],
    device const float* d_val [[buffer(7)]],
    constant uint& B [[buffer(8)]],
    constant uint& H [[buffer(9)]],
    constant uint& T [[buffer(10)]],
    constant uint& C [[buffer(11)]],
    constant uint& causal [[buffer(12)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    fused_fa_bwd_dkv_packed_impl<2u, 128u>(
        q, k, v, dO, dK, dV, lse, d_val,
        B, H, T, C, causal, tg_pos, lane);
}

// load_device_8x8_pair 假设行列共用一个 extent，[T][D] 的 K/V cache 需要分离的边界。
METAL_FUNC static float2 load_device_frag_8x8(
    device const float* src,
    uint row_stride,
    uint row_base,
    uint col_base,
    uint row_extent,
    uint col_extent,
    ushort lane
) {
    const ushort2 offset = morton_order(lane);
    const uint row = row_base + uint(offset.y);
    const uint col = col_base + uint(offset.x);
    if (row >= row_extent || col >= col_extent) {
        return float2(0.0f);
    }
    if (col + 1u < col_extent) {
        return float2(
            *reinterpret_cast<device const packed_float2*>(
                src + row * row_stride + col
            )
        );
    }
    return float2(src[row * row_stride + col], 0.0f);
}

kernel void fused_fa_bwd_dkv_mma64(
    device const float* q_cache      [[buffer(0)]],
    device const float* k_cache      [[buffer(1)]],
    device const float* v_cache      [[buffer(2)]],
    device const float* dO           [[buffer(3)]],
    device float* dK                 [[buffer(4)]],
    device float* dV                 [[buffer(5)]],
    device const float* lse          [[buffer(6)]],
    device const float* d_val        [[buffer(7)]],
    constant uint& B                 [[buffer(8)]],
    constant uint& H                 [[buffer(9)]],
    constant uint& T                 [[buffer(10)]],
    constant uint& C                 [[buffer(11)]],
    constant uint& is_causal_int     [[buffer(12)]],
    uint3 tg_pos                     [[threadgroup_position_in_grid]],
    uint3 thread_pos                 [[thread_position_in_threadgroup]],
    uint3 threads_per_tg             [[threads_per_threadgroup]],
    uint lane                        [[thread_index_in_simdgroup]],
    uint sg                          [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count             [[threads_per_simdgroup]]
) {
    constexpr uint kFrag = 8u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kThreads = kSimdGroups * 32u;
    constexpr uint kBK = kSimdGroups * kFrag;
    constexpr uint kBQ = 16u;
    constexpr uint kQSub = kBQ / kFrag;
    constexpr uint kMaxD = 64u;
    constexpr uint kDSub = kMaxD / kFrag;
    constexpr uint kQStride = kMaxD + 4u;
    // 输出的 d 方向按 grid.x 切成 kDChunks 份，每份自己重算一遍 S/dP。
    // kDChunks=1 表示不切分，但 d_out_base 仍然要走 tg_pos.x 算出来：这个偏移
    // 是性能承重的，不要把它常量折叠成 0。编译器一旦知道它恒为 0，就会把
    // dK/dV 循环里 8 组 threadgroup 载入聚簇到一起预取，多出的 16 个同时活跃
    // 的值刚好把寄存器推过上面那道悬崖，实测从 18.6ms 掉到 30.7ms。
    // 复测方法见 /tmp/fa_exp/bench_dkv.py（对 float64 numpy 参考验数 + 轮转交错
    // 计时），若哪天升级 Xcode 后这里变慢，先查是不是这个偏移被折叠了。
    constexpr uint kDChunks = 1u;
    constexpr uint kDOutSub = (kDSub + kDChunks - 1u) / kDChunks;

    threadgroup float Qs[kBQ * kQStride];
    threadgroup float dOs[kBQ * kQStride];
    threadgroup float lse_tile[kBQ];
    threadgroup float dval_tile[kBQ];

    // 所有线程必须根据统一条件退出，避免barrier死锁。
    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u ||
        threads_per_tg.z != 1u || simd_lane_count != 32u ||
        tg_pos.x >= kDChunks ||
        B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > kMaxD) {
        return;
    }
    const uint bh = tg_pos.z;
    const uint key_block = tg_pos.y * kBK;
    if (bh >= B * H || key_block >= T) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint stat_base = bh * T;
    const bool is_causal = is_causal_int != 0u;
    const float scale = rsqrt(float(D));
    const float scale_log2e = scale * 1.4426950408889634f;
    constexpr uint d_padded = kMaxD;
    const uint key_frag_base = key_block + sg * kFrag;
    const ushort2 frag_offset = morton_order(ushort(lane));
    const uint key_row = key_frag_base + uint(frag_offset.y);
    const uint d_out_base = tg_pos.x * kDOutSub;

    // K/V 在整个内核生命周期内不变，常驻寄存器。
    thread float2 k_frag[kDSub];
    thread float2 v_frag[kDSub];
    thread float2 dk_accum[kDOutSub];
    thread float2 dv_accum[kDOutSub];

    #pragma unroll
    for (uint i = 0u; i < kDOutSub; ++i) {
        dk_accum[i] = float2(0.0f);
        dv_accum[i] = float2(0.0f);
    }
    #pragma unroll
    for (uint d_index = 0u; d_index < kDSub; ++d_index) {
        k_frag[d_index] = float2(0.0f);
        v_frag[d_index] = float2(0.0f);
        k_frag[d_index] = load_device_frag_8x8(k_cache + head_base, D,
            key_frag_base, d_index * kFrag, T, D, ushort(lane));
        v_frag[d_index] = load_device_frag_8x8(v_cache + head_base, D,
            key_frag_base, d_index * kFrag, T, D, ushort(lane));
    }

    const uint query_begin = is_causal ? key_block : 0u;

    uint q_start = query_begin;
    uint q_tile_base = head_base + q_start * D;
    uint do_tile_base = (b * T + q_start) * C + h * D;
    uint stat_tile_base = stat_base + q_start;
    for (; q_start < T; q_start += kBQ,
         q_tile_base += kBQ * D,
         do_tile_base += kBQ * C,
         stat_tile_base += kBQ) {
        // 上一轮的 tile 读取必须先完成才能覆盖。
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_q = sg; local_q < kBQ; local_q += kSimdGroups) {
            const uint query = q_start + local_q;
            const bool row_valid = query < T;
            const uint q_row = q_tile_base + local_q * D;
            const uint do_row = do_tile_base + local_q * C;
            for (uint d = lane; d < d_padded; d += 32u) {
                float q_value = 0.0f;
                float do_value = 0.0f;
                if (row_valid && d < D) {
                    q_value = q_cache[q_row + d];
                    do_value = dO[do_row + d];
                }
                Qs[local_q * kQStride + d] = q_value;
                dOs[local_q * kQStride + d] = do_value;
            }
        }
        if (thread_pos.x < kBQ) {
            const uint query = q_start + thread_pos.x;
            lse_tile[thread_pos.x] = query < T
                ? lse[stat_tile_base + thread_pos.x] * 1.4426950408889634f
                : 0.0f;
            dval_tile[thread_pos.x] =
                query < T ? d_val[stat_tile_base + thread_pos.x] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S^T 与 dP^T：A 取常驻的 K/V，B 取转置读出的 Q^T/dO^T。
        // d 在外层，一个 d 步里 kQSub 组 score/dp 互不依赖。
        thread float2 score_frag[kQSub];
        thread float2 dp_frag[kQSub];
        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            score_frag[q_sub] = float2(0.0f);
            dp_frag[q_sub] = float2(0.0f);
        }
        #pragma unroll
        for (uint d_index = 0u; d_index < kDSub; ++d_index) {
            #pragma unroll
            for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
                const ushort2 origin = ushort2(
                    ushort(d_index * kFrag), ushort(q_sub * kFrag));
                const float2 q_transposed = lowreg_matrix::load_tg_8x8(
                    Qs, kQStride, origin, ushort(lane), true);
                const float2 do_transposed = lowreg_matrix::load_tg_8x8(
                    dOs, kQStride, origin, ushort(lane), true);
                score_frag[q_sub] = lowreg_matrix::mma_8x8(
                    k_frag[d_index], q_transposed, score_frag[q_sub]);
                dp_frag[q_sub] = lowreg_matrix::mma_8x8(
                    v_frag[d_index], do_transposed, dp_frag[q_sub]);
            }
        }

        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            // P = exp(S/sqrt(D) - LSE)，dS = P * (dP - D_i) / sqrt(D)。
            // 无效列（越界或被 causal 屏蔽）直接置零，零值会自然传播到 dK/dV。
            const uint q_local0 = q_sub * kFrag + uint(frag_offset.x);
            const uint q_local1 = q_local0 + 1u;
            const uint query0 = q_start + q_local0;
            const uint query1 = q_start + q_local1;
            const bool valid0 = query0 < T && (!is_causal || query0 >= key_row);
            const bool valid1 = query1 < T && (!is_causal || query1 >= key_row);
            const float lse0 = lse_tile[q_local0];
            const float lse1 = lse_tile[q_local1];
            const float dval0 = dval_tile[q_local0];
            const float dval1 = dval_tile[q_local1];
            const float p0 = valid0 ? exp2(score_frag[q_sub][0] * scale_log2e - lse0)
                                    : 0.0f;
            const float p1 = valid1 ? exp2(score_frag[q_sub][1] * scale_log2e - lse1)
                                    : 0.0f;
            const float2 prob_frag = float2(p0, p1);
            const float2 dscore_frag = float2(
                scale * p0 * (dp_frag[q_sub][0] - dval0),
                scale * p1 * (dp_frag[q_sub][1] - dval1));

            // dK/dV：A 取刚算出的 dS/P，B 取同一份 tile 的非转置读法。
            // d 在内层，dk_accum[i]/dv_accum[i] 之间互不依赖。
            #pragma unroll
            for (uint i = 0u; i < kDOutSub; ++i) {
                const uint d_index = d_out_base + i;
                const ushort2 origin = ushort2(
                    ushort(q_sub * kFrag), ushort(d_index * kFrag));
                const float2 q_value = lowreg_matrix::load_tg_8x8(
                    Qs, kQStride, origin, ushort(lane));
                const float2 do_value = lowreg_matrix::load_tg_8x8(
                    dOs, kQStride, origin, ushort(lane));
                dk_accum[i] = lowreg_matrix::mma_8x8(
                    dscore_frag, q_value, dk_accum[i]);
                dv_accum[i] = lowreg_matrix::mma_8x8(
                    prob_frag, do_value, dv_accum[i]);
            }
        }
    }

    // 每个 key 只属于一个 threadgroup，所以直接覆盖写。
    if (key_row < T) {
        #pragma unroll
        for (uint i = 0u; i < kDOutSub; ++i) {
            const uint d_index = d_out_base + i;
            const uint d_col = d_index * kFrag + uint(frag_offset.x);
            if (d_col >= D) {
                continue;
            }
            const uint out_base = (b * T + key_row) * C + h * D + d_col;
            if (d_col + 1u < D) {
                *reinterpret_cast<device packed_float2*>(dK + out_base) =
                    packed_float2(dk_accum[i]);
                *reinterpret_cast<device packed_float2*>(dV + out_base) =
                    packed_float2(dv_accum[i]);
            } else {
                dK[out_base] = dk_accum[i][0];
                dV[out_base] = dv_accum[i][0];
            }
        }
    }
}

template <uint kMaxD, uint kSimdGroups, uint kBQ, uint kDOutChunks>
METAL_FUNC static void fused_fa_bwd_dkv_mma_tiled_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dK,
    device float* dV,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint3 thread_pos,
    uint3 threads_per_tg,
    uint lane,
    uint sg,
    uint simd_lane_count,
    threadgroup float* Qs,
    threadgroup float* dOs,
    threadgroup float* lse_tile,
    threadgroup float* dval_tile
) {
    constexpr uint kFrag = 8u;
    constexpr uint kDTile = 64u;
    constexpr uint kDSubPerTile = kDTile / kFrag;
    constexpr uint kQSub = kBQ / kFrag;
    constexpr uint kThreads = kSimdGroups * 32u;
    constexpr uint kBK = kSimdGroups * kFrag;
    constexpr uint kQStride = kDTile + 4u;
    constexpr uint kMaxChunks = (kMaxD + kDTile - 1u) / kDTile;
    constexpr uint kDOutSub = kDOutChunks * kDSubPerTile;
    static_assert((kBQ % kFrag) == 0u, "kBQ must be a multiple of 8");
    static_assert(kDOutChunks > 0u, "at least one output D tile is required");

    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u ||
        threads_per_tg.z != 1u || simd_lane_count != 32u ||
        B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > kMaxD) {
        return;
    }
    const uint d_chunks = (D + kDTile - 1u) / kDTile;
    const uint output_chunk_base = tg_pos.x * kDOutChunks;
    if (output_chunk_base >= d_chunks) {
        return;
    }

    const uint bh = tg_pos.z;
    const uint key_block = tg_pos.y * kBK;
    if (bh >= B * H || key_block >= T) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint stat_base = bh * T;
    const bool is_causal = is_causal_int != 0u;
    const float scale = rsqrt(float(D));
    const float scale_log2e = scale * 1.4426950408889634f;
    const uint key_frag_base = key_block + sg * kFrag;
    const ushort2 frag_offset = morton_order(ushort(lane));
    const uint key_row = key_frag_base + uint(frag_offset.y);

    // Each float2 accumulator is one lane's portion of an 8x8 output fragment.
    // kDOutChunks=1 consumes 32 floats/thread for dK+dV; =2 consumes 64.
    thread float2 dk_accum[kDOutSub];
    thread float2 dv_accum[kDOutSub];
    #pragma unroll
    for (uint i = 0u; i < kDOutSub; ++i) {
        dk_accum[i] = float2(0.0f);
        dv_accum[i] = float2(0.0f);
    }

    const uint query_begin = is_causal ? key_block : 0u;
    uint q_start = query_begin;
    uint q_tile_base = head_base + q_start * D;
    uint do_tile_base = (b * T + q_start) * C + h * D;
    uint stat_tile_base = stat_base + q_start;

    for (; q_start < T; q_start += kBQ,
         q_tile_base += kBQ * D,
         do_tile_base += kBQ * C,
         stat_tile_base += kBQ) {
        // Protect the shared tile from the previous query iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_pos.x < kBQ) {
            const uint query = q_start + thread_pos.x;
            lse_tile[thread_pos.x] = query < T
                ? lse[stat_tile_base + thread_pos.x] * 1.4426950408889634f
                : 0.0f;
            dval_tile[thread_pos.x] = query < T
                ? d_val[stat_tile_base + thread_pos.x]
                : 0.0f;
        }

        thread float2 score_frag[kQSub];
        thread float2 dp_frag[kQSub];
        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            score_frag[q_sub] = float2(0.0f);
            dp_frag[q_sub] = float2(0.0f);
        }

        // Pass 1: stream all 64-column D tiles to form complete score and dP.
        for (uint d_chunk = 0u; d_chunk < kMaxChunks; ++d_chunk) {
            if (d_chunk >= d_chunks) {
                break;
            }
            if (d_chunk != 0u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const uint d_base = d_chunk * kDTile;
            for (uint local_q = sg; local_q < kBQ; local_q += kSimdGroups) {
                const uint query = q_start + local_q;
                const bool row_valid = query < T;
                const uint q_row = q_tile_base + local_q * D;
                const uint do_row = do_tile_base + local_q * C;
                for (uint local_d = lane; local_d < kDTile; local_d += 32u) {
                    const uint d = d_base + local_d;
                    float q_value = 0.0f;
                    float do_value = 0.0f;
                    if (row_valid && d < D) {
                        q_value = q_cache[q_row + d];
                        do_value = dO[do_row + d];
                    }
                    Qs[local_q * kQStride + local_d] = q_value;
                    dOs[local_q * kQStride + local_d] = do_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stream one K/V fragment at a time.  Keeping all eight fragments
            // live would add 32 floats/thread and reduce occupancy for out=2.
            #pragma unroll
            for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                const uint global_d = d_base + d_index * kFrag;
                const float2 k_frag = load_device_frag_8x8(
                    k_cache + head_base, D, key_frag_base, global_d,
                    T, D, ushort(lane));
                const float2 v_frag = load_device_frag_8x8(
                    v_cache + head_base, D, key_frag_base, global_d,
                    T, D, ushort(lane));
                #pragma unroll
                for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
                    const ushort2 origin = ushort2(
                        ushort(d_index * kFrag), ushort(q_sub * kFrag));
                    const float2 q_transposed = lowreg_matrix::load_tg_8x8(
                        Qs, kQStride, origin, ushort(lane), true);
                    const float2 do_transposed = lowreg_matrix::load_tg_8x8(
                        dOs, kQStride, origin, ushort(lane), true);
                    score_frag[q_sub] = lowreg_matrix::mma_8x8(
                        k_frag, q_transposed, score_frag[q_sub]);
                    dp_frag[q_sub] = lowreg_matrix::mma_8x8(
                        v_frag, do_transposed, dp_frag[q_sub]);
                }
            }
        }

        // Reuse score/dP registers for P/dS so no additional fragment arrays
        // become live alongside the output accumulators.
        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            const uint q_local0 = q_sub * kFrag + uint(frag_offset.x);
            const uint q_local1 = q_local0 + 1u;
            const uint query0 = q_start + q_local0;
            const uint query1 = q_start + q_local1;
            const bool valid0 = query0 < T && (!is_causal || query0 >= key_row);
            const bool valid1 = query1 < T && (!is_causal || query1 >= key_row);
            const float p0 = valid0
                ? exp2(score_frag[q_sub][0] * scale_log2e - lse_tile[q_local0])
                : 0.0f;
            const float p1 = valid1
                ? exp2(score_frag[q_sub][1] * scale_log2e - lse_tile[q_local1])
                : 0.0f;
            const float dp0 = dp_frag[q_sub][0];
            const float dp1 = dp_frag[q_sub][1];
            score_frag[q_sub] = float2(p0, p1);
            dp_frag[q_sub] = float2(
                scale * p0 * (dp0 - dval_tile[q_local0]),
                scale * p1 * (dp1 - dval_tile[q_local1]));
        }

        // Pass 2: reload only this threadgroup's output D tiles and accumulate.
        #pragma unroll
        for (uint out_chunk = 0u; out_chunk < kDOutChunks; ++out_chunk) {
            const uint d_chunk = output_chunk_base + out_chunk;
            if (d_chunk >= d_chunks) {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint d_base = d_chunk * kDTile;
            for (uint local_q = sg; local_q < kBQ; local_q += kSimdGroups) {
                const uint query = q_start + local_q;
                const bool row_valid = query < T;
                const uint q_row = q_tile_base + local_q * D;
                const uint do_row = do_tile_base + local_q * C;
                for (uint local_d = lane; local_d < kDTile; local_d += 32u) {
                    const uint d = d_base + local_d;
                    float q_value = 0.0f;
                    float do_value = 0.0f;
                    if (row_valid && d < D) {
                        q_value = q_cache[q_row + d];
                        do_value = dO[do_row + d];
                    }
                    Qs[local_q * kQStride + local_d] = q_value;
                    dOs[local_q * kQStride + local_d] = do_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
                #pragma unroll
                for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                    const ushort2 origin = ushort2(
                        ushort(q_sub * kFrag), ushort(d_index * kFrag));
                    const float2 q_value = lowreg_matrix::load_tg_8x8(
                        Qs, kQStride, origin, ushort(lane));
                    const float2 do_value = lowreg_matrix::load_tg_8x8(
                        dOs, kQStride, origin, ushort(lane));
                    const uint accum_index = out_chunk * kDSubPerTile + d_index;
                    dk_accum[accum_index] = lowreg_matrix::mma_8x8(
                        dp_frag[q_sub], q_value, dk_accum[accum_index]);
                    dv_accum[accum_index] = lowreg_matrix::mma_8x8(
                        score_frag[q_sub], do_value, dv_accum[accum_index]);
                }
            }
        }
    }

    if (key_row < T) {
        #pragma unroll
        for (uint out_chunk = 0u; out_chunk < kDOutChunks; ++out_chunk) {
            const uint d_chunk = output_chunk_base + out_chunk;
            if (d_chunk >= d_chunks) {
                break;
            }
            #pragma unroll
            for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                const uint d_col = d_chunk * kDTile + d_index * kFrag
                    + uint(frag_offset.x);
                if (d_col >= D) {
                    continue;
                }
                const uint accum_index = out_chunk * kDSubPerTile + d_index;
                const uint out_base = (b * T + key_row) * C + h * D + d_col;
                if (d_col + 1u < D) {
                    *reinterpret_cast<device packed_float2*>(dK + out_base) =
                        packed_float2(dk_accum[accum_index]);
                    *reinterpret_cast<device packed_float2*>(dV + out_base) =
                        packed_float2(dv_accum[accum_index]);
                } else {
                    dK[out_base] = dk_accum[accum_index][0];
                    dV[out_base] = dv_accum[accum_index][0];
                }
            }
        }
    }
}

#define DECLARE_FA_DKV_MMA_TILED_KERNEL(NAME, MAX_D, SIMD_GROUPS, BQ, OUT_CHUNKS) \
kernel void NAME(                                                               \
    device const float* q [[buffer(0)]],                                        \
    device const float* k [[buffer(1)]],                                        \
    device const float* v [[buffer(2)]],                                        \
    device const float* dO [[buffer(3)]],                                       \
    device float* dK [[buffer(4)]],                                             \
    device float* dV [[buffer(5)]],                                             \
    device const float* lse [[buffer(6)]],                                      \
    device const float* d_val [[buffer(7)]],                                    \
    constant uint& B [[buffer(8)]],                                             \
    constant uint& H [[buffer(9)]],                                             \
    constant uint& T [[buffer(10)]],                                            \
    constant uint& C [[buffer(11)]],                                            \
    constant uint& causal [[buffer(12)]],                                       \
    uint3 tg_pos [[threadgroup_position_in_grid]],                              \
    uint3 thread_pos [[thread_position_in_threadgroup]],                        \
    uint3 threads_per_tg [[threads_per_threadgroup]],                           \
    uint lane [[thread_index_in_simdgroup]],                                    \
    uint sg [[simdgroup_index_in_threadgroup]],                                 \
    uint simd_lane_count [[threads_per_simdgroup]]) {                           \
    constexpr uint kQStride = 68u;                                              \
    threadgroup float Qs[(BQ) * kQStride];                                      \
    threadgroup float dOs[(BQ) * kQStride];                                     \
    threadgroup float lse_tile[(BQ)];                                           \
    threadgroup float dval_tile[(BQ)];                                          \
    fused_fa_bwd_dkv_mma_tiled_impl<MAX_D, SIMD_GROUPS, BQ, OUT_CHUNKS>(        \
        q, k, v, dO, dK, dV, lse, d_val, B, H, T, C, causal,                   \
        tg_pos, thread_pos, threads_per_tg, lane, sg, simd_lane_count,          \
        Qs, dOs, lse_tile, dval_tile);                                          \
}

// Selected Apple M1 configurations.  D=128 only needs one output group;
// D=256/512 select BQ and SIMD-group count from sequence length in backend.py.
DECLARE_FA_DKV_MMA_TILED_KERNEL(fused_fa_bwd_dkv_mma128_q24_o2_sg4, 128u, 4u, 24u, 2u)

#undef DECLARE_FA_DKV_MMA_TILED_KERNEL

#define DECLARE_FA_DKV_LARGE_KERNEL(NAME, MAX_D, SLOTS)                  \
kernel void NAME(                                                       \
    device const float* q [[buffer(0)]],                                \
    device const float* k [[buffer(1)]],                                \
    device const float* v [[buffer(2)]],                                \
    device const float* dO [[buffer(3)]],                               \
    device float* dK [[buffer(4)]],                                     \
    device float* dV [[buffer(5)]],                                     \
    device const float* lse [[buffer(6)]],                              \
    device const float* d_val [[buffer(7)]],                            \
    constant uint& B [[buffer(8)]],                                     \
    constant uint& H [[buffer(9)]],                                     \
    constant uint& T [[buffer(10)]],                                    \
    constant uint& C [[buffer(11)]],                                    \
    constant uint& causal [[buffer(12)]],                               \
    uint3 tg_pos [[threadgroup_position_in_grid]],                      \
    uint lane [[thread_index_in_simdgroup]]) {                          \
    fused_fa_bwd_dkv_large_impl<MAX_D, SLOTS>(                          \
        q, k, v, dO, dK, dV, lse, d_val,                               \
        B, H, T, C, causal, tg_pos, lane);                              \
}

#undef DECLARE_FA_DKV_LARGE_KERNEL

kernel void fused_rms_recompute(
    device const float* x          [[buffer(0)]],
    device float* z                [[buffer(1)]],
    device float* inv_rms_out      [[buffer(2)]],
    constant uint& B               [[buffer(3)]],
    constant uint& T               [[buffer(4)]],
    constant uint& C               [[buffer(5)]],
    constant uint& eps_bits        [[buffer(6)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kSimdWidth = 32u;
    constexpr uint kTokensPerTG = 4u;
    constexpr uint kVectorWidth = 4u;

    if (tg_pos.x != 0u || C == 0u) {
        return;
    }

    const uint token = tg_pos.y * kTokensPerTG + simd_id;
    const uint token_count = B * T;

    if (token >= token_count) {
        return;
    }

    const float eps = as_type<float>(eps_bits);
    const uint base = token * C;
    const uint vector_count = C / kVectorWidth;
    const uint vector_end = vector_count * kVectorWidth;
    device const packed_float4* x4 = reinterpret_cast<device const packed_float4*>(x + base);

    // Pass 1: sum(x^2)
    float local_square_sum = 0.0f;
    for (uint vector_index = lane; vector_index < vector_count; vector_index += kSimdWidth) {
        const float4 value = float4(x4[vector_index]);
        local_square_sum += dot(value, value);
    }
    for (uint c = vector_end + lane; c < C; c += kSimdWidth) {
        const float value = x[base + c];
        local_square_sum = fma(value, value, local_square_sum);
    }

    const float mean_square = simd_sum(local_square_sum) / float(C);
    const float inv_rms = rsqrt(mean_square + eps);

    if (lane == 0u) {
        inv_rms_out[token] = inv_rms;
    }

    // Pass 2: z = x * inv_rms
    device packed_float4* z4 = reinterpret_cast<device packed_float4*>(z + base);
    for (uint vector_index = lane; vector_index < vector_count; vector_index += kSimdWidth) {
        const float4 value = float4(x4[vector_index]);
        z4[vector_index] = packed_float4(value * float4(inv_rms));
    }

    for (uint c = vector_end + lane; c < C; c += kSimdWidth) {
        z[base + c] = x[base + c] * inv_rms;
    }
}

kernel void mlp_backward_dx(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C_out [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kBM = 128u;
    constexpr uint kBN = 64u;
    constexpr uint kBK = 32u;
    constexpr uint kThreads = 512u;
    constexpr uint kFrag = 8u;
    constexpr uint kATileCols = kBK / kFrag;
    constexpr uint kBTileCols = kBN / kFrag;
    constexpr uint kTileElements = kFrag * kFrag;
    constexpr uint kFloat4PerBK = kBK / 4u;

    // 128 * 32 * 4 = 16 KiB
    threadgroup float A_tg[kBM * kBK];
    // 32 * 64 * 4 = 8 KiB
    threadgroup float BT_tg[kBK * kBN];
    const uint row_block = tg_pos.y * kBM;
    const uint col_block = tg_pos.x * kBN;
    // 每个simdgroup计算一个8x64输出子块。
    const uint simd_row_base = simd_id * kFrag;

    float2 acc0 = float2(0.0f);
    float2 acc1 = float2(0.0f);
    float2 acc2 = float2(0.0f);
    float2 acc3 = float2(0.0f);
    float2 acc4 = float2(0.0f);
    float2 acc5 = float2(0.0f);
    float2 acc6 = float2(0.0f);
    float2 acc7 = float2(0.0f);

    for (uint k_base = 0u; k_base < K; k_base += kBK) {
        // --------------------------------------------------------
        // A tile: [BM, BK]
        // 写入8x8 tile-major布局，使后续simdgroup matrix加载连续。
        // --------------------------------------------------------
        constexpr uint kAFloat4Count = kBM * kFloat4PerBK;
        for (uint idx = tid; idx < kAFloat4Count; idx += kThreads) {
            const uint local_row = idx / kFloat4PerBK;
            const uint local_k = (idx - local_row * kFloat4PerBK) * 4u;
            const uint global_row = row_block + local_row;
            const uint global_k = k_base + local_k;

            float4 value = float4(0.0f);
            if (global_row < M && global_k + 3u < K) {
                value = float4(*reinterpret_cast<device const packed_float4*>(A + global_row * K + global_k));
            } else if (global_row < M && global_k < K) {
                #pragma unroll
                for (uint component = 0u; component < 4u; ++component) {
                    if (global_k + component < K) {
                        value[component] = A[global_row * K + global_k + component];
                    }
                }
            }

            const uint tile_index = ((local_row / kFrag) * kATileCols + local_k / kFrag) * kTileElements;
            const uint physical = tile_index + (local_row & 7u) * kFrag + (local_k & 7u);
            *reinterpret_cast<threadgroup packed_float4*>(A_tg + physical) = packed_float4(value);
        }

        // --------------------------------------------------------
        // B tile: B的全局布局为 [N, K]。内核逻辑上使用B^T，但不生成B的转置buffer。
        // threadgroup内写为 [BK, BN] tile-major布局。
        // --------------------------------------------------------
        constexpr uint kBFloat4Count = kBN * kFloat4PerBK;
        for (uint idx = tid; idx < kBFloat4Count; idx += kThreads) {
            const uint local_col = idx / kFloat4PerBK;
            const uint local_k = (idx - local_col * kFloat4PerBK) * 4u;
            const uint global_col = col_block + local_col;
            const uint global_k = k_base + local_k;

            float4 value = float4(0.0f);
            if (global_col < N && global_k + 3u < K) {
                value = float4(*reinterpret_cast<device const packed_float4*>(B + global_col * K + global_k));
            } else if (global_col < N && global_k < K) {
                #pragma unroll
                for (uint component = 0u; component < 4u; ++component) {
                    if (global_k + component < K) {
                        value[component] = B[global_col * K + global_k + component];
                    }
                }
            }

            #pragma unroll
            for (uint component = 0u; component < 4u; ++component) {
                const uint k = local_k + component;
                const uint tile_index = ((k / kFrag) * kBTileCols + local_col / kFrag) * kTileElements;
                const uint physical = tile_index + (k & 7u) * kFrag + (local_col & 7u);
                BT_tg[physical] = value[component];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --------------------------------------------------------
        // 每个simdgroup计算8x64结果：1个A fragment 8个B fragments
        // --------------------------------------------------------

        #pragma unroll
        for (uint k_tile = 0u; k_tile < kATileCols; ++k_tile) {
            const float2 a = lowreg_matrix::load_tg_8x8(A_tg + (simd_id * kATileCols + k_tile) *
                    kTileElements, kFrag, ushort2(0u), ushort(lane));
            const uint b_base = k_tile * kBTileCols;
            const float2 b0 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 0u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b1 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 1u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b2 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 2u) * kTileElements, 
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b3 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 3u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b4 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 4u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b5 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 5u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b6 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 6u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));
            const float2 b7 = lowreg_matrix::load_tg_8x8(BT_tg + (b_base + 7u) * kTileElements,
                    kFrag, ushort2(0u), ushort(lane));

            acc0 = lowreg_matrix::mma_8x8(a, b0, acc0);
            acc1 = lowreg_matrix::mma_8x8(a, b1, acc1);
            acc2 = lowreg_matrix::mma_8x8(a, b2, acc2);
            acc3 = lowreg_matrix::mma_8x8(a, b3, acc3);
            acc4 = lowreg_matrix::mma_8x8(a, b4, acc4);
            acc5 = lowreg_matrix::mma_8x8(a, b5, acc5);
            acc6 = lowreg_matrix::mma_8x8(a, b6, acc6);
            acc7 = lowreg_matrix::mma_8x8(a, b7, acc7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------
    // 写回 [M, N]
    // ------------------------------------------------------------
    const ushort2 offset = morton_order(ushort(lane));
    const uint global_row = row_block + simd_row_base + uint(offset.y);
    const uint local_col = uint(offset.x);

    if (global_row < M) {
        const float2 results[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        #pragma unroll
        for (uint fragment_idx = 0u; fragment_idx < 8u; ++fragment_idx) {
            const uint global_col = col_block + local_col + fragment_idx * kFrag;
            if (global_col < N) {
                C_out[global_row * N + global_col] = results[fragment_idx][0];
            }
            if (global_col + 1u < N) {
                C_out[global_row * N + global_col + 1u] = results[fragment_idx][1];
            }
        }
    }
}

kernel void rope_qk_inplace(
    device float* q           [[buffer(0)]],
    device float* k           [[buffer(1)]],
    device const float* cos   [[buffer(2)]],
    device const float* sin   [[buffer(3)]],
    constant uint& B          [[buffer(4)]],
    constant uint& H          [[buffer(5)]],
    constant uint& T          [[buffer(6)]],
    constant uint& D          [[buffer(7)]],
    constant uint& mode       [[buffer(8)]],
    uint3 gid                 [[thread_position_in_grid]]
) {
    const uint j = gid.x;
    const uint t = gid.y;
    const uint bh = gid.z;
    const uint half_D = D >> 1u;

    if (
        H == 0u ||
        j >= half_D ||
        t >= T ||
        bh >= B * H
    ) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;

    const uint bhtd_base =
        (bh * T + t) * D;

    const uint bthd_base =
        ((b * T + t) * H + h) * D;

    // mode 0: Q/K 都是 BHTD
    // mode 1: Q/K 都是 BTHD
    // mode 2: Q 是 BTHD，K 是 BHTD
    const uint q_base =
        mode == 0u ? bhtd_base : bthd_base;

    const uint k_base =
        mode == 1u ? bthd_base : bhtd_base;

    const uint q0_index = q_base + j;
    const uint q1_index = q0_index + half_D;
    const uint k0_index = k_base + j;
    const uint k1_index = k0_index + half_D;

    const uint rope_index = t * half_D + j;
    const float c = cos[rope_index];

    // mode 1 是 backward，使用 R^T，即 sin 取反。
    const float s =
        mode == 1u
        ? -sin[rope_index]
        : sin[rope_index];

    const float q0 = q[q0_index];
    const float q1 = q[q1_index];
    const float k0 = k[k0_index];
    const float k1 = k[k1_index];

    q[q0_index] = fma(q0, c, q1 * s);
    q[q1_index] = fma(-q0, s, q1 * c);

    k[k0_index] = fma(k0, c, k1 * s);
    k[k1_index] = fma(-k0, s, k1 * c);
}

kernel void fused_rmsnorm_bwd_dx(
    device const float* x          [[buffer(0)]],
    device const float* dY         [[buffer(1)]],
    device const float* inv_rms_in [[buffer(2)]],
    device float* dX               [[buffer(3)]],
    constant uint& BT              [[buffer(4)]],
    constant uint& C               [[buffer(5)]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]],
    uint3 thread_pos               [[thread_position_in_threadgroup]]
) {
    constexpr uint kThreads = 32u;
    const uint tid = thread_pos.x;
    if (tid >= kThreads || tg_pos.x != 0u) {
        return;
    }

    const uint row = tg_pos.y;
    if (row >= BT || C == 0u) {
        return;
    }

    const uint base = row * C;
    const float inv_rms = inv_rms_in[row];
    float local_dot = 0.0f;
    for (uint c = tid; c < C; c += kThreads) {
        const float x_hat = x[base + c] * inv_rms;
        local_dot = fma(dY[base + c], x_hat, local_dot);
    }

    const float mean_dy_xhat = simd_sum(local_dot) / float(C);
    for (uint c = tid; c < C; c += kThreads) {
        const float x_hat = x[base + c] * inv_rms;
        const float dx = inv_rms * (dY[base + c] - x_hat * mean_dy_xhat);
        dX[base + c] += dx;
    }
}

kernel void fused_qkv_bias_grad(
    device const float* dQ           [[buffer(0)]],
    device const float* dK           [[buffer(1)]],
    device const float* dV           [[buffer(2)]],
    device float* dbq                [[buffer(3)]],
    device float* dbk                [[buffer(4)]],
    device float* dbv                [[buffer(5)]],
    constant uint& rows              [[buffer(6)]],  // BT
    constant uint& C                 [[buffer(7)]],
    constant uint& accumulate        [[buffer(8)]],
    uint3 tg_pos     [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]
) {
    constexpr uint kThreads = 128;
    constexpr uint kChannelTile = 4;
    const uint tid = thread_pos.x;
    if (tid >= kThreads || tg_pos.x != 0) {
        return;
    }
    if (rows == 0 || C == 0) {
        return;
    }
    const uint c_base = tg_pos.y * kChannelTile;
    if (c_base >= C) {
        return;
    }
    const uint tile = min(kChannelTile, C - c_base);
    // 每线程：对本 tile 的每个 channel 做 BT 跨步累加
    // 同一 row 上连续读 dQ/dK/dV，提高缓存局部性
    float sum_q[kChannelTile];
    float sum_k[kChannelTile];
    float sum_v[kChannelTile];
    for (uint i = 0; i < kChannelTile; ++i) {
        sum_q[i] = 0.0f;
        sum_k[i] = 0.0f;
        sum_v[i] = 0.0f;
    }
    for (uint row = tid; row < rows; row += kThreads) {
        const uint base = row * C + c_base;
        for (uint i = 0; i < tile; ++i) {
            sum_q[i] += dQ[base + i];
            sum_k[i] += dK[base + i];
            sum_v[i] += dV[base + i];
        }
    }
    // threadgroup tree reduction（按 channel 串行做 3 次，缓冲可复用）
    threadgroup float reduce_buf[kThreads];
    for (uint i = 0; i < tile; ++i) {
        // ---- dbq ----
        reduce_buf[tid] = sum_q[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0 && dbq) {
            dbq[c_base + i] = accumulate ? dbq[c_base + i] + reduce_buf[0] : reduce_buf[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- dbk ----
        reduce_buf[tid] = sum_k[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0 && dbk) {
            dbk[c_base + i] = accumulate ? dbk[c_base + i] + reduce_buf[0] : reduce_buf[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- dbv ----
        reduce_buf[tid] = sum_v[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0 && dbv) {
            dbv[c_base + i] = accumulate ? dbv[c_base + i] + reduce_buf[0] : reduce_buf[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 仅供 fused_ln_fa_project_kv 使用。
// 前置条件：ptr 已包含 morton 偏移，且 8 字节对齐
//（tile 基址 16 字节对齐 + stride 为偶数 + 列基址为偶数，三者由调用方保证）。
// 语义与 lowreg_matrix::load_tg_8x8 的非转置路径完全一致：返回 (v[col], v[col+1])。
METAL_FUNC static float2 tg_load_frag2(threadgroup const float* ptr) {
    return *reinterpret_cast<threadgroup const float2*>(ptr);
}

kernel void fused_rms_fa_project_kv(
    device const float* x           [[buffer(0)]],
    device const float* wq          [[buffer(1)]],
    device const float* bq          [[buffer(2)]],
    device const float* wk          [[buffer(3)]],
    device const float* bk          [[buffer(4)]],
    device const float* wv          [[buffer(5)]],
    device const float* bv          [[buffer(6)]],
    device float* q_out             [[buffer(7)]],
    device float* k_cache           [[buffer(8)]],
    device float* v_cache           [[buffer(9)]],
    constant uint& B                [[buffer(10)]],
    constant uint& T                [[buffer(11)]],
    constant uint& C                [[buffer(12)]],
    constant uint& H                [[buffer(13)]],
    constant uint& eps_bits         [[buffer(14)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kThreads      = 512u;
    constexpr uint kSimdWidth    = 32u;
    constexpr uint kBM           = 32u;
    constexpr uint kBN           = 64u;
    constexpr uint kBK           = 32u;              // ← 16 → 32
    constexpr uint kFrag         = 8u;
    constexpr uint kXStride      = kBK + 4u;         // ← 36
    constexpr uint kWStride      = 68u;
    constexpr uint kRowsPerSimd  = 2u;
    constexpr uint kC4PerBK      = kBK / 4u;         // 8
    constexpr uint kN4PerTile    = kBN / 4u;         // 16
    constexpr uint kXFloat4Count = kBM * kC4PerBK;   // 256
    constexpr uint kWFloat4Count = kBK * kN4PerTile; // 512

    threadgroup float X_tg[kBM][kXStride];
    threadgroup float Wk_tg[kBK][kWStride];
    threadgroup float Wv_tg[kBK][kWStride];
    threadgroup float Wq_tg[kBK][kWStride];
    threadgroup float inv_rms_tg[kBM];
    threadgroup float bq_tg[kBN];
    threadgroup float bk_tg[kBN];
    threadgroup float bv_tg[kBN];

    const uint tid = thread_pos.x;

    if (tid >= kThreads || tg_pos.x != 0u || tg_pos.z != 0u ||
        B == 0u || T == 0u || C == 0u || H == 0u || (C & 3u) != 0u || (C % H) != 0u) {
        return;
    }

    const uint token_count = B * T;
    const uint D = C / H;
    if (D == 0u) { return; }
    const uint row_base = tg_pos.y * kBM;
    if (row_base >= token_count) { return; }

    const float eps = as_type<float>(eps_bits);

    // ---------------- RMS 统计：与 5.18ms 版逐字相同 ----------------
    const uint slr0 = simd_id * kRowsPerSimd;
    const uint slr1 = slr0 + 1u;
    const uint sgr0 = row_base + slr0;
    const uint sgr1 = row_base + slr1;
    const bool sv0  = sgr0 < token_count;
    const bool sv1  = sgr1 < token_count;
    const uint sxb0 = sgr0 * C;
    const uint sxb1 = sgr1 * C;
    float lss0 = 0.0f, lss1 = 0.0f;
    if (sv0) {
        for (uint c = lane; c < C; c += kSimdWidth) {
            const float v = x[sxb0 + c];
            lss0 = fma(v, v, lss0);
        }
    }
    if (sv1) {
        for (uint c = lane; c < C; c += kSimdWidth) {
            const float v = x[sxb1 + c];
            lss1 = fma(v, v, lss1);
        }
    }
    const float ir0 = sv0 ? rsqrt(simd_sum(lss0) / float(C) + eps) : 0.0f;
    const float ir1 = sv1 ? rsqrt(simd_sum(lss1) / float(C) + eps) : 0.0f;
    if (lane == 0u) { inv_rms_tg[slr0] = ir0; inv_rms_tg[slr1] = ir1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------- SIMD mapping：不变 ----------------
    const uint matrix_row_group = simd_id >> 2u;
    const uint matrix_col_group = simd_id & 3u;
    const uint simd_row_base = matrix_row_group * kFrag;
    const uint simd_col_base = matrix_col_group * 16u;
    const ushort2 frag = morton_order(ushort(lane));

    threadgroup const float* const x_frag_base =
        reinterpret_cast<threadgroup const float*>(X_tg)
        + (simd_row_base + uint(frag.y)) * kXStride + uint(frag.x);
    const uint w_frag_disp = uint(frag.y) * kWStride + simd_col_base + uint(frag.x);
    threadgroup const float* const wk_frag_base = reinterpret_cast<threadgroup const float*>(Wk_tg) + w_frag_disp;
    threadgroup const float* const wv_frag_base = reinterpret_cast<threadgroup const float*>(Wv_tg) + w_frag_disp;
    threadgroup const float* const wq_frag_base = reinterpret_cast<threadgroup const float*>(Wq_tg) + w_frag_disp;

    // ---------------- 循环不变量提前算好（免费）----------------
    // X loader
    const bool  x_loader    = tid < kXFloat4Count;
    const uint  x_local_row = x_loader ? (tid / kC4PerBK) : 0u;
    const uint  x_local_c   = x_loader ? ((tid - x_local_row * kC4PerBK) * 4u) : 0u;
    const uint  x_grow      = row_base + x_local_row;
    const bool  x_row_ok    = x_loader && (x_grow < token_count);
    device const float* const x_row_ptr = x + x_grow * C;
    const float x_scale     = x_loader ? inv_rms_tg[x_local_row] : 0.0f;   // barrier 之后，只读一次

    // W loader（512 线程全部参与）
    const uint  w_local_k   = tid / kN4PerTile;                            // 0..31
    const uint  w_local_n   = (tid - w_local_k * kN4PerTile) * 4u;         // 0,4,...,60
    const uint  w_row_off   = w_local_k * C;

    const bool d_is_pow2 = (D & (D - 1u)) == 0u;
    const uint d_shift   = uint(popcount(D - 1u));
    const uint d_mask    = D - 1u;
    const bool d_even    = (D & 1u) == 0u;
    const uint kv_head_stride = T * D;

    for (uint output_base = 0u; output_base < C; output_base += kBN) {

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < kN4PerTile) {
            const uint lc = tid * 4u;
            const uint gc = output_base + lc;
            float4 b0 = float4(0.0f), b1 = float4(0.0f), b2 = float4(0.0f);
            if (gc < C) {
                b0 = float4(*reinterpret_cast<device const packed_float4*>(bq + gc));
                b1 = float4(*reinterpret_cast<device const packed_float4*>(bk + gc));
                b2 = float4(*reinterpret_cast<device const packed_float4*>(bv + gc));
            }
            *reinterpret_cast<threadgroup packed_float4*>(bq_tg + lc) = packed_float4(b0);
            *reinterpret_cast<threadgroup packed_float4*>(bk_tg + lc) = packed_float4(b1);
            *reinterpret_cast<threadgroup packed_float4*>(bv_tg + lc) = packed_float4(b2);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float2 q_accumulator0 = float2(0.0f), q_accumulator1 = float2(0.0f);
        float2 k_accumulator0 = float2(0.0f), k_accumulator1 = float2(0.0f);
        float2 v_accumulator0 = float2(0.0f), v_accumulator1 = float2(0.0f);

        const uint w_col_off  = output_base + w_local_n;
        const bool w_col_ok   = w_col_off < C;

        for (uint input_base = 0u; input_base < C; input_base += kBK) {

            // X tile：256 线程各 1 个 float4
            if (x_loader) {
                const uint gc = input_base + x_local_c;
                float4 nrm = float4(0.0f);
                if (x_row_ok && gc < C) {
                    const float4 xv = float4(*reinterpret_cast<device const packed_float4*>(x_row_ptr + gc));
                    nrm = xv * float4(x_scale);
                }
                *reinterpret_cast<threadgroup packed_float4*>(&X_tg[x_local_row][x_local_c]) = packed_float4(nrm);
            }

            // W tile：512 线程全员各 3 个 float4（原来只有 256 线程参与）
            {
                const uint gk = input_base + w_local_k;
                float4 wkv = float4(0.0f), wvv = float4(0.0f), wqv = float4(0.0f);
                if (w_col_ok && gk < C) {
                    const uint off = input_base * C + w_row_off + w_col_off;
                    wkv = float4(*reinterpret_cast<device const packed_float4*>(wk + off));
                    wvv = float4(*reinterpret_cast<device const packed_float4*>(wv + off));
                    wqv = float4(*reinterpret_cast<device const packed_float4*>(wq + off));
                }
                *reinterpret_cast<threadgroup packed_float4*>(&Wk_tg[w_local_k][w_local_n]) = packed_float4(wkv);
                *reinterpret_cast<threadgroup packed_float4*>(&Wv_tg[w_local_k][w_local_n]) = packed_float4(wvv);
                *reinterpret_cast<threadgroup packed_float4*>(&Wq_tg[w_local_k][w_local_n]) = packed_float4(wqv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 4 个 k_sub（原来 2 个），MMA 顺序与原版完全一致
            #pragma unroll
            for (uint k_sub = 0u; k_sub < kBK; k_sub += kFrag) {
                const uint w_row_disp = k_sub * kWStride;
                const float2 x_fragment   = tg_load_frag2(x_frag_base + k_sub);

                const float2 wk_fragment0 = tg_load_frag2(wk_frag_base + w_row_disp);
                const float2 wk_fragment1 = tg_load_frag2(wk_frag_base + w_row_disp + kFrag);
                k_accumulator0 = lowreg_matrix::mma_8x8(x_fragment, wk_fragment0, k_accumulator0);
                k_accumulator1 = lowreg_matrix::mma_8x8(x_fragment, wk_fragment1, k_accumulator1);

                const float2 wv_fragment0 = tg_load_frag2(wv_frag_base + w_row_disp);
                const float2 wv_fragment1 = tg_load_frag2(wv_frag_base + w_row_disp + kFrag);
                v_accumulator0 = lowreg_matrix::mma_8x8(x_fragment, wv_fragment0, v_accumulator0);
                v_accumulator1 = lowreg_matrix::mma_8x8(x_fragment, wv_fragment1, v_accumulator1);

                const float2 wq_fragment0 = tg_load_frag2(wq_frag_base + w_row_disp);
                const float2 wq_fragment1 = tg_load_frag2(wq_frag_base + w_row_disp + kFrag);
                q_accumulator0 = lowreg_matrix::mma_8x8(x_fragment, wq_fragment0, q_accumulator0);
                q_accumulator1 = lowreg_matrix::mma_8x8(x_fragment, wq_fragment1, q_accumulator1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---------------- epilogue：与 5.18ms 版逐字相同 ----------------
        const ushort2 mo = morton_order(ushort(lane));
        const uint local_row  = simd_row_base + uint(mo.y);
        const uint global_row = row_base + local_row;

        if (global_row < token_count) {
            const uint b = global_row / T;
            const uint t = global_row - b * T;
            const uint q_row_base = global_row * C;
            const uint kv_bt_base = b * T * C + t * D;

            #pragma unroll
            for (uint fi = 0u; fi < 2u; ++fi) {
                const float2 q_value = fi == 0u ? q_accumulator0 : q_accumulator1;
                const float2 k_value = fi == 0u ? k_accumulator0 : k_accumulator1;
                const float2 v_value = fi == 0u ? v_accumulator0 : v_accumulator1;
                const uint fc  = simd_col_base + uint(mo.x) + fi * 8u;
                const uint oc0 = output_base + fc;
                const bool pair_ok = (fc + 1u < kBN) && (oc0 + 1u < C) && ((oc0 & 1u) == 0u);

                if (pair_ok) {
                    const float2 q_store = q_value + float2(bq_tg[fc], bq_tg[fc + 1u]);
                    const float2 k_store = k_value + float2(bk_tg[fc], bk_tg[fc + 1u]);
                    const float2 v_store = v_value + float2(bv_tg[fc], bv_tg[fc + 1u]);

                    *reinterpret_cast<device packed_float2*>(q_out + q_row_base + oc0) = packed_float2(q_store);

                    uint h0, d0;
                    if (d_is_pow2) { h0 = oc0 >> d_shift; d0 = oc0 & d_mask; }
                    else           { h0 = oc0 / D;        d0 = oc0 - h0 * D; }
                    const uint ci0 = kv_bt_base + h0 * kv_head_stride + d0;

                    if (d_even) {
                        *reinterpret_cast<device packed_float2*>(k_cache + ci0) = packed_float2(k_store);
                        *reinterpret_cast<device packed_float2*>(v_cache + ci0) = packed_float2(v_store);
                    } else {
                        k_cache[ci0] = k_store.x; v_cache[ci0] = v_store.x;
                        uint h1 = h0, d1 = d0 + 1u;
                        if (d1 >= D) { d1 = 0u; h1 = h0 + 1u; }
                        const uint ci1 = kv_bt_base + h1 * kv_head_stride + d1;
                        k_cache[ci1] = k_store.y; v_cache[ci1] = v_store.y;
                    }
                } else {
                    #pragma unroll
                    for (uint cp = 0u; cp < 2u; ++cp) {
                        const uint lc = fc + cp;
                        const uint oc = output_base + lc;
                        if (lc < kBN && oc < C) {
                            q_out[q_row_base + oc] = q_value[cp] + bq_tg[lc];
                            uint h, d;
                            if (d_is_pow2) { h = oc >> d_shift; d = oc & d_mask; }
                            else           { h = oc / D;        d = oc - h * D; }
                            const uint ci = kv_bt_base + h * kv_head_stride + d;
                            k_cache[ci] = k_value[cp] + bk_tg[lc];
                            v_cache[ci] = v_value[cp] + bv_tg[lc];
                        }
                    }
                }
            }
        }
    }
}

template <uint kDChunks>
METAL_FUNC static void fa_br32_load_q(
    device const float* q,
    threadgroup float* Q_tg,
    uint b,
    uint h,
    uint query_base,
    uint T,
    uint C,
    uint D,
    uint tid
) {
    constexpr uint kBQ = 32u;
    constexpr uint kBD = 64u;
    constexpr uint kQStride = 65u;
    constexpr uint kVecPerRow = kBD / 4u;
    constexpr uint kThreads = 256u;
    constexpr uint kVectors = kDChunks * kBQ * kVecPerRow;

    for (uint index = tid; index < kVectors; index += kThreads) {
        const uint chunk = index / (kBQ * kVecPerRow);
        const uint chunk_index = index - chunk * kBQ * kVecPerRow;
        const uint local_q = chunk_index / kVecPerRow;
        const uint vector_index = chunk_index - local_q * kVecPerRow;
        const uint local_d = vector_index * 4u;
        const uint global_d = chunk * kBD + local_d;
        const uint query = query_base + local_q;
        float4 value = float4(0.0f);

        if (query < T && global_d < D) {
            const uint src = (b * T + query) * C + h * D + global_d;
            if (global_d + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(q + src));
            } else {
                if (global_d < D) {
                    value.x = q[src];
                }
                if (global_d + 1u < D) {
                    value.y = q[src + 1u];
                }
                if (global_d + 2u < D) {
                    value.z = q[src + 2u];
                }
            }
        }
        const uint dst = chunk * kBQ * kQStride + local_q * kQStride + local_d;
        *reinterpret_cast<threadgroup packed_float4*>(Q_tg + dst) = packed_float4(value);
    }
}

METAL_FUNC static void fa_br32_load_k_transposed(
    device const float* k_cache,
    threadgroup float* KV_tg,
    uint head_base,
    uint key_base,
    uint valid_keys,
    uint d_base,
    uint D,
    uint tid
) {
    constexpr uint kBK = 32u;
    constexpr uint kBD = 64u;
    constexpr uint kKStride = 33u;
    constexpr uint kVecPerRow = kBD / 4u;
    constexpr uint kVectors = kBK * kVecPerRow;
    constexpr uint kThreads = 256u;

    for (uint index = tid; index < kVectors; index += kThreads) {
        const uint local_key = index / kVecPerRow;
        const uint vector_index = index - local_key * kVecPerRow;
        const uint local_d = vector_index * 4u;
        const uint global_d = d_base + local_d;
        float4 value = float4(0.0f);

        if (local_key < valid_keys && global_d < D) {
            const uint src = head_base + (key_base + local_key) * D + global_d;

            if (global_d + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(k_cache + src));
            } else {
                if (global_d < D) {
                    value.x = k_cache[src];
                }
                if (global_d + 1u < D) {
                    value.y = k_cache[src + 1u];
                }
                if (global_d + 2u < D) {
                    value.z = k_cache[src + 2u];
                }
            }
        }
        // KV_tg is K^T: [64, 33].
        KV_tg[(local_d + 0u) * kKStride + local_key] = value.x;
        KV_tg[(local_d + 1u) * kKStride + local_key] = value.y;
        KV_tg[(local_d + 2u) * kKStride + local_key] = value.z;
        KV_tg[(local_d + 3u) * kKStride + local_key] = value.w;
    }
}

METAL_FUNC static void fa_br32_load_v_row_major(
    device const float* v_cache,
    threadgroup float* KV_tg,
    uint head_base,
    uint key_base,
    uint valid_keys,
    uint d_base,
    uint D,
    uint tid
) {
    constexpr uint kBK = 32u;
    constexpr uint kBD = 64u;
    constexpr uint kVStride = 65u;
    constexpr uint kVecPerRow = kBD / 4u;
    constexpr uint kVectors = kBK * kVecPerRow;
    constexpr uint kThreads = 256u;

    for (uint index = tid; index < kVectors; index += kThreads) {
        const uint local_key = index / kVecPerRow;
        const uint vector_index = index - local_key * kVecPerRow;
        const uint local_d = vector_index * 4u;
        const uint global_d = d_base + local_d;
        float4 value = float4(0.0f);
        if (local_key < valid_keys && global_d < D) {
            const uint src = head_base + (key_base + local_key) * D + global_d;
            if (global_d + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(v_cache + src));
            } else {
                if (global_d < D) {
                    value.x = v_cache[src];
                }
                if (global_d + 1u < D) {
                    value.y = v_cache[src + 1u];
                }
                if (global_d + 2u < D) {
                    value.z = v_cache[src + 2u];
                }
            }
        }
        const uint dst = local_key * kVStride + local_d;
        *reinterpret_cast<threadgroup packed_float4*>(KV_tg + dst) = packed_float4(value);
    }
}

METAL_FUNC static void fa_br32_store_score_fragment(
    threadgroup float* P_tg,
    float2 score,
    uint q_group,
    uint key_group,
    uint query_base,
    uint key_base,
    uint key_limit,
    uint T,
    bool is_causal,
    float scale,
    ushort lane
) {
    constexpr uint kFrag = 8u;
    constexpr uint kPStride = 33u;
    const ushort2 offset = morton_order(lane);
    const uint local_q = q_group * kFrag + uint(offset.y);
    const uint query = query_base + local_q;
    const uint key0 = key_base + key_group * kFrag + uint(offset.x);
    const uint key1 = key0 + 1u;
    const bool valid_query = query < T;
    const bool valid0 = valid_query && key0 < key_limit && (!is_causal || key0 <= query);
    const bool valid1 = valid_query && key1 < key_limit && (!is_causal || key1 <= query);
    float2 masked_score;
    masked_score.x = valid0 ? score.x * scale : -INFINITY;
    masked_score.y = valid1 ? score.y * scale : -INFINITY;
    lowreg_matrix::store_tg_8x8(P_tg, kPStride,
        ushort2(ushort(q_group * kFrag), ushort(key_group * kFrag)), lane, masked_score);
}

METAL_FUNC static void fa_br32_softmax(
    threadgroup float* P_tg,
    threadgroup float* stats_tg,
    uint sg,
    uint lane
) {
    constexpr uint kBQ = 32u;
    constexpr uint kPStride = 33u;
    // stats layout:
    //   [0 * 32 + row] = running_max, in log2 domain
    //   [1 * 32 + row] = running_sum
    //   [2 * 32 + row] = alpha
    #pragma unroll
    for (uint row_pass = 0u; row_pass < 4u; ++row_pass) {
        const uint row = sg + row_pass * 8u;
        const float score = P_tg[row * kPStride + lane];
        const float tile_max = simd_max(score);
        const float old_max = stats_tg[0u * kBQ + row];
        const float old_sum = stats_tg[1u * kBQ + row];
        const bool tile_has_value = tile_max != -INFINITY;
        const float new_max = tile_has_value ? max(old_max, tile_max) : old_max;
        const float alpha = old_sum > 0.0f ? exp(old_max - new_max) : 0.0f;
        const float probability = tile_has_value && score != -INFINITY ? exp(score - new_max) : 0.0f;
        const float tile_sum = simd_sum(probability);
        const float new_sum = old_sum * alpha + tile_sum;
        P_tg[row * kPStride + lane] = probability;
        if (lane == 0u) {
            stats_tg[0u * kBQ + row] = new_max;
            stats_tg[1u * kBQ + row] = new_sum;
            stats_tg[2u * kBQ + row] = alpha;
        }
    }
}

template <uint kDChunks>
METAL_FUNC static void fused_rms_fa_forward_br32_impl(
    device float* q_out,
    device const float* k_cache,
    device const float* v_cache,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint3 tg_pos,
    uint3 threads_per_tg,
    uint tid,
    uint lane_u32,
    uint sg,
    threadgroup float* Q_tg,
    threadgroup float* KV_tg,
    threadgroup float* P_tg,
    threadgroup float* stats_tg
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdGroups = 8u;
    constexpr uint kBQ = 32u;
    constexpr uint kBK = 32u;
    constexpr uint kBD = 64u;
    constexpr uint kFrag = 8u;
    constexpr uint kQGroups = kBQ / kFrag; // 4
    constexpr uint kDSub = kBD / kFrag;    // 8
    constexpr uint kQStride = 65u;
    constexpr uint kKStride = 33u;
    constexpr uint kVStride = 65u;
    constexpr uint kPStride = 33u;
    constexpr float kLog2E = 1.4426950408889634074f;

    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u || tg_pos.x != 0u ||
        B == 0u || H == 0u || T == 0u || C == 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    if (D == 0u || D > kDChunks * kBD) {
        return;
    }
    const uint bh = tg_pos.z;
    if (bh >= B * H || sg >= kSimdGroups) {
        return;
    }
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const bool is_causal = is_causal_int != 0u;
    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;
    const uint head_base = bh * T * D;
    const float scale = rsqrt(float(D));
    const ushort lane = ushort(lane_u32);
    const ushort2 fragment_offset = morton_order(lane);
    // Each SG owns one 8-column D fragment.
    const uint output_d_group = sg;
    // 4 Q8 fragments per D chunk.
    // D64:  4 float2 = 8 floats/thread.
    // D128: 8 float2 = 16 floats/thread.
    thread float2 output_frag[kDChunks * kQGroups];

    #pragma unroll
    for (uint i = 0u; i < kDChunks * kQGroups; ++i) {
        output_frag[i] = float2(0.0f);
    }

    fa_br32_load_q<kDChunks>(q_out, Q_tg, b, h, query_base, T, C, D, tid);

    if (tid < kBQ) {
        stats_tg[0u * kBQ + tid] = -INFINITY;
        stats_tg[1u * kBQ + tid] = 0.0f;
        stats_tg[2u * kBQ + tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {
        const uint valid_keys = min(kBK, key_limit - key_base);
        const uint valid_key_groups = (valid_keys + kFrag - 1u) / kFrag;

        // QK mapping:
        // SG 0..3: key_group 0..3, Q groups 0 and 1
        // SG 4..7: key_group 0..3, Q groups 2 and 3
        const uint key_group = sg & 3u;
        const uint q_pair = sg >> 2u;
        const uint q_group0 = q_pair * 2u;
        const uint q_group1 = q_group0 + 1u;
        thread float2 score0 = float2(0.0f);
        thread float2 score1 = float2(0.0f);

        // ---------------------------------------------------------------------
        // QK: K enters TG once per D chunk, then serves both Q8 fragments.
        // ---------------------------------------------------------------------
        #pragma unroll
        for (uint chunk = 0u; chunk < kDChunks; ++chunk) {
            const uint d_base = chunk * kBD;
            fa_br32_load_k_transposed(k_cache, KV_tg, head_base, key_base, valid_keys, d_base, D, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid_keys == kBK || key_group < valid_key_groups) {
                #pragma unroll
                for (uint d_sub = 0u; d_sub < kDSub; ++d_sub) {
                    const float2 q0 = lowreg_matrix::load_tg_8x8(Q_tg + chunk * kBQ * kQStride, kQStride,
                            ushort2(ushort(q_group0 * kFrag), ushort(d_sub * kFrag)), lane);
                    const float2 q1 = lowreg_matrix::load_tg_8x8(Q_tg + chunk * kBQ * kQStride, kQStride,
                            ushort2(ushort(q_group1 * kFrag), ushort(d_sub * kFrag)), lane);
                    const float2 kt = lowreg_matrix::load_tg_8x8(KV_tg, kKStride, 
                            ushort2(ushort(d_sub * kFrag), ushort(key_group * kFrag)), lane);
                    score0 = lowreg_matrix::mma_8x8(q0, kt, score0);
                    score1 = lowreg_matrix::mma_8x8(q1, kt, score1);
                }
            }
            if (chunk + 1u < kDChunks) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        fa_br32_store_score_fragment(P_tg, score0, q_group0, key_group,
            query_base, key_base, key_limit, T, is_causal, scale, lane);
        fa_br32_store_score_fragment(P_tg, score1, q_group1, key_group,
            query_base, key_base, key_limit, T, is_causal, scale, lane);

        // All 16 score fragments must be visible to softmax.
        // This also guarantees that K is no longer in use, so the
        // first V chunk may overwrite KV_tg.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SG processes four complete softmax rows.
        fa_br32_softmax(P_tg, stats_tg, sg, lane_u32);

        // Load the first V chunk while the softmax result/stats are
        // still waiting for cross-SG visibility.
        fa_br32_load_v_row_major(v_cache, KV_tg, head_base, key_base, valid_keys, 0u, D, tid);

        // Makes P, stats and V visible simultaneously.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scale every resident O fragment exactly once for this K tile.
        #pragma unroll
        for (uint chunk = 0u; chunk < kDChunks; ++chunk) {
            #pragma unroll
            for (uint q_group = 0u; q_group < kQGroups; ++q_group) {
                const uint local_q = q_group * kFrag + uint(fragment_offset.y);
                const float alpha = stats_tg[2u * kBQ + local_q];
                output_frag[chunk * kQGroups + q_group] *= alpha;
            }
        }

        // ---------------------------------------------------------------------
        // PV: each V chunk is loaded once and updates all 32 query rows.
        // ---------------------------------------------------------------------
        #pragma unroll
        for (uint chunk = 0u; chunk < kDChunks; ++chunk) {
            if (chunk != 0u) {
                // All SGs must finish reading the previous V chunk
                // before KV_tg is overwritten.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                fa_br32_load_v_row_major(v_cache, KV_tg, head_base, key_base, valid_keys, chunk * kBD, D, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            #pragma unroll
            for (uint q_group = 0u; q_group < kQGroups; ++q_group) {
                float2 accumulator = output_frag[chunk * kQGroups + q_group];
                if (valid_keys == kBK) {
                    #pragma unroll
                    for (uint key_sub = 0u; key_sub < 4u; ++key_sub) {
                        const float2 probability = lowreg_matrix::load_tg_8x8(P_tg, kPStride,
                                ushort2(ushort(q_group * kFrag), ushort(key_sub * kFrag)), lane);
                        const float2 value = lowreg_matrix::load_tg_8x8(KV_tg, kVStride,
                                ushort2(ushort(key_sub * kFrag), ushort(output_d_group * kFrag)), lane);
                        accumulator = lowreg_matrix::mma_8x8(probability, value, accumulator);
                    }
                } else {
                    // Only the final partial key tile uses this path.
                    for (uint key_sub = 0u; key_sub < valid_key_groups; ++key_sub) {
                        const float2 probability = lowreg_matrix::load_tg_8x8(P_tg, kPStride,
                                ushort2(ushort(q_group * kFrag), ushort(key_sub * kFrag)), lane);
                        const float2 value = lowreg_matrix::load_tg_8x8(KV_tg, kVStride,
                                ushort2(ushort(key_sub * kFrag), ushort(output_d_group * kFrag)), lane);
                        accumulator = lowreg_matrix::mma_8x8(probability, value, accumulator);
                    }
                }
                output_frag[chunk * kQGroups + q_group] = accumulator;
            }
        }

        // Required before the next K tile overwrites KV_tg.
        // The last tile does not need it because no TG memory follows.
        if (key_base + kBK < key_limit) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // -------------------------------------------------------------------------
    // Final normalize and one packed write to q_out.
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint chunk = 0u; chunk < kDChunks; ++chunk) {
        #pragma unroll
        for (uint q_group = 0u; q_group < kQGroups; ++q_group) {
            const uint local_q = q_group * kFrag + uint(fragment_offset.y);
            const uint query = query_base + local_q;
            const uint local_d = output_d_group * kFrag + uint(fragment_offset.x);
            const uint global_d = chunk * kBD + local_d;

            if (query < T && global_d < D) {
                const float running_sum = stats_tg[1u * kBQ + local_q];
                const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
                const float2 result = output_frag[chunk * kQGroups + q_group] * inverse_sum;
                const uint dst = (b * T + query) * C + h * D + global_d;
                if (global_d + 1u < D) {
                    *reinterpret_cast<device packed_float2*>(q_out + dst) = packed_float2(result);
                } else {
                    q_out[dst] = result.x;
                }
            }
        }
    }
}

kernel void fused_rms_fa_forward_block64(
    device float* q_out          [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    constant uint& B             [[buffer(3)]],
    constant uint& H             [[buffer(4)]],
    constant uint& T             [[buffer(5)]],
    constant uint& C             [[buffer(6)]],
    constant uint& is_causal_int [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float Q_tg[32u * 65u];
    threadgroup float KV_tg[64u * 33u];
    threadgroup float P_tg[32u * 33u];
    threadgroup float stats_tg[32u * 3u];
    fused_rms_fa_forward_br32_impl<1u>(q_out, k_cache, v_cache, B, H, T, C, is_causal_int, tg_pos,
        threads_per_tg, thread_pos.x, lane, sg, Q_tg, KV_tg, P_tg, stats_tg);
}

kernel void fused_rms_fa_forward_block128(
    device float* q_out          [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    constant uint& B             [[buffer(3)]],
    constant uint& H             [[buffer(4)]],
    constant uint& T             [[buffer(5)]],
    constant uint& C             [[buffer(6)]],
    constant uint& is_causal_int [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 threads_per_tg [[threads_per_threadgroup]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float Q_tg[2u * 32u * 65u];
    threadgroup float KV_tg[64u * 33u];
    threadgroup float P_tg[32u * 33u];
    threadgroup float stats_tg[32u * 3u];
    fused_rms_fa_forward_br32_impl<2u>(q_out, k_cache, v_cache, B, H, T, C, is_causal_int, tg_pos,
        threads_per_tg, thread_pos.x, lane, sg, Q_tg, KV_tg, P_tg, stats_tg);
}

kernel void rmsnorm_backward(
    device const float* x             [[buffer(0)]],
    device const float* dy            [[buffer(1)]],
    device float* dx_out              [[buffer(2)]],
    device const float* residual_grad [[buffer(3)]],
    constant uint& B_T                [[buffer(4)]],
    constant uint& C                  [[buffer(5)]],
    constant uint& eps_bits           [[buffer(6)]],
    uint3 tg_pos                      [[threadgroup_position_in_grid]],
    uint3 thread_pos                  [[thread_position_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTG = 8u;
    constexpr uint kChannelTile = 64u;
    const uint tid = thread_pos.x;

    if (tid >= kThreads || tg_pos.x != 0u || C == 0u) {
        return;
    }

    const uint local_row = simd_id;
    const uint row = tg_pos.y * kRowsPerTG + local_row;
    const bool row_valid = row < B_T;
    const uint row_base = row * C;
    const float C_f = float(C);
    const float eps = as_type<float>(eps_bits);
    float local_square_sum = 0.0f;
    if (row_valid) {
        for (uint c = lane; c < C; c += kSimdWidth) {
            const float value = x[row_base + c];
            local_square_sum = fma(value, value, local_square_sum);
        }
    }

    const float square_sum = simd_sum(local_square_sum);
    const float inv_rms = row_valid ? rsqrt(square_sum / C_f + eps) : 0.0f;
    float local_dy_xnorm = 0.0f;
    if (row_valid) {
        for (uint c = lane; c < C; c += kSimdWidth) {
            const float x_norm = x[row_base + c] * inv_rms;
            local_dy_xnorm = fma(dy[row_base + c], x_norm, local_dy_xnorm);
        }
    }

    const float dy_xnorm_sum = simd_sum(local_dy_xnorm);
    const float mean_dy_xnorm = row_valid ? dy_xnorm_sum / C_f : 0.0f;
    for (uint channel_base = 0u; channel_base < C; channel_base += kChannelTile) {
        const uint c0 = channel_base + lane;
        const uint c1 = channel_base + kSimdWidth + lane;

        if (row_valid && c0 < C) {
            const uint index0 = row_base + c0;
            const float x_norm0 = x[index0] * inv_rms;
            const float dx0 = inv_rms * (dy[index0] - x_norm0 * mean_dy_xnorm);
            dx_out[index0] = residual_grad[index0] + dx0;
        }
        if (row_valid && c1 < C) {
            const uint index1 = row_base + c1;
            const float x_norm1 = x[index1] * inv_rms;
            const float dx1 = inv_rms * (dy[index1] - x_norm1 * mean_dy_xnorm);
            dx_out[index1] = residual_grad[index1] + dx1;
        }
    }
}

kernel void rmsnorm_weighted_backward(
    device const float* x              [[buffer(0)]],
    device const float* d_xnorm        [[buffer(1)]],
    device const float* gamma          [[buffer(2)]],
    device float* dx                   [[buffer(3)]],
    device const float* residual_grad  [[buffer(4)]],
    constant uint& N                   [[buffer(5)]],
    constant uint& C                   [[buffer(6)]],
    constant uint& eps_bits            [[buffer(7)]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]],
    uint3 thread_pos                   [[thread_position_in_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]]
) {
    const uint tid = thread_pos.x;

    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerThreadgroup = 8u;
    constexpr uint kChannelTile = 64u;

    if (tid >= kThreads || C == 0u) {
        return;
    }

    const uint local_row = tid / kSimdWidth;
    const uint row =
        tg_pos.y * kRowsPerThreadgroup + local_row;

    if (row >= N) {
        return;
    }

    const uint base = row * C;
    const float eps = as_type<float>(eps_bits);

    float square_sum = 0.0f;

    for (uint c = lane; c < C; c += kSimdWidth) {
        const float value = x[base + c];
        square_sum = fma(value, value, square_sum);
    }

    const float total_square_sum = simd_sum(square_sum);
    const float inv_rms =
        rsqrt(total_square_sum / float(C) + eps);

    float local_dot = 0.0f;

    for (uint c = lane; c < C; c += kSimdWidth) {
        const float x_hat = x[base + c] * inv_rms;
        const float dy = d_xnorm[base + c] * gamma[c];
        local_dot = fma(dy, x_hat, local_dot);
    }

    const float mean_dot = simd_sum(local_dot) / float(C);

    for (
        uint channel_base = 0u;
        channel_base < C;
        channel_base += kChannelTile
    ) {
        const uint c0 = channel_base + lane;
        const uint c1 = channel_base + kSimdWidth + lane;

        if (c0 < C) {
            const uint index = base + c0;
            const float x_hat = x[index] * inv_rms;
            const float dy = d_xnorm[index] * gamma[c0];

            dx[index] = residual_grad[index]
                + inv_rms * (dy - x_hat * mean_dot);
        }

        if (c1 < C) {
            const uint index = base + c1;
            const float x_hat = x[index] * inv_rms;
            const float dy = d_xnorm[index] * gamma[c1];

            dx[index] = residual_grad[index]
                + inv_rms * (dy - x_hat * mean_dot);
        }
    }
}

kernel void rmsnorm_weighted_dgamma(
    device const float* x              [[buffer(0)]],
    device const float* d_xnorm        [[buffer(1)]],
    device uchar* dgamma               [[buffer(2)]],
    constant uint& N                   [[buffer(3)]],
    constant uint& C                   [[buffer(4)]],
    constant uint& eps_bits            [[buffer(5)]],
    constant uint& accumulate          [[buffer(6)]],
    constant uint& grad_bf16           [[buffer(7)]],
    uint c                             [[thread_position_in_grid]]
) {
    if (c >= C) {
        return;
    }

    const float eps = as_type<float>(eps_bits);
    float result = 0.0f;

    for (uint row = 0; row < N; ++row) {
        const uint base = row * C;

        float square_sum = 0.0f;

        for (uint j = 0; j < C; ++j) {
            const float value = x[base + j];
            square_sum = fma(value, value, square_sum);
        }

        const float inv_rms =
            rsqrt(square_sum / float(C) + eps);

        const float x_hat = x[base + c] * inv_rms;

        result += d_xnorm[base + c] * x_hat;
    }

    if (grad_bf16 != 0u) {
        device ushort* dst = reinterpret_cast<device ushort*>(dgamma);
        if (accumulate != 0u) {
            result += as_type<float>(uint(dst[c]) << 16);
        }
        const uint bits = as_type<uint>(result);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            dst[c] = ushort((bits >> 16) | ((bits & 0x007fffffu) != 0u ? 0x0040u : 0u));
        } else {
            // Round-to-nearest, ties-to-even。
            dst[c] = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        }
    } else {
        device float* dst = reinterpret_cast<device float*>(dgamma);
        if (accumulate != 0u) {
            dst[c] += result;
        } else {
            dst[c] = result;
        }
    }
}

kernel void kv_cache_rope_store(
    device float* q                 [[buffer(0)]],
    device float* k                 [[buffer(1)]],
    device const float* v           [[buffer(2)]],
    device float* kc                [[buffer(3)]],
    device float* vc                [[buffer(4)]],
    device const int* position      [[buffer(5)]],
    device const float* rope_cos    [[buffer(6)]],
    device const float* rope_sin    [[buffer(7)]],
    constant uint& B                [[buffer(8)]],
    constant uint& H                [[buffer(9)]],
    constant uint& T                [[buffer(10)]],
    constant uint& D                [[buffer(11)]],
    constant uint& S                [[buffer(12)]],
    uint3 group                     [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]]
) {
    const uint half_dim = D / 2u;
    const uint index = group.y * 128u + tid;

    if (index >= B * H * T * half_dim) {
        return;
    }

    const uint d = index % half_dim;
    const uint row = index / half_dim;
    const uint t = row % T;
    const uint bh = row / T;
    const uint batch = bh / H;
    const uint head = bh % H;
    const uint pos = uint(position[0]) + t;

    const uint src = row * D + d;
    const uint dst = ((batch * S + pos) * H + head) * D + d;
    const uint rope_index = pos * half_dim + d;

    const float c = rope_cos[rope_index];
    const float s = rope_sin[rope_index];

    const float q0 = q[src];
    const float q1 = q[src + half_dim];
    const float k0 = k[src];
    const float k1 = k[src + half_dim];

    const float rq0 = fma(q0, c, q1 * s);
    const float rq1 = fma(-q0, s, q1 * c);
    const float rk0 = fma(k0, c, k1 * s);
    const float rk1 = fma(-k0, s, k1 * c);

    q[src] = rq0;
    q[src + half_dim] = rq1;

    k[src] = rk0;
    k[src + half_dim] = rk1;

    kc[dst] = rk0;
    kc[dst + half_dim] = rk1;

    vc[dst] = v[src];
    vc[dst + half_dim] = v[src + half_dim];
}

kernel void kv_cache_rope_store_broadcast(
    device float* q                 [[buffer(0)]],
    device float* k                 [[buffer(1)]],
    device const float* v           [[buffer(2)]],
    device float* kc                [[buffer(3)]],
    device float* vc                [[buffer(4)]],
    device const int* position      [[buffer(5)]],
    device const float* rope_cos    [[buffer(6)]],
    device const float* rope_sin    [[buffer(7)]],
    constant uint& B                [[buffer(8)]],
    constant uint& H                [[buffer(9)]],
    constant uint& T                [[buffer(10)]],
    constant uint& D                [[buffer(11)]],
    constant uint& S                [[buffer(12)]],
    constant uint& BC               [[buffer(13)]],
    uint3 group                     [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]]
) {
    const uint half_dim = D / 2u;
    const uint index = group.y * 128u + tid;

    // 输入 prefill 当前应为 B=1。
    if (index >= B * H * T * half_dim) {
        return;
    }

    const uint d = index % half_dim;
    const uint row = index / half_dim;
    const uint t = row % T;
    const uint bh = row / T;
    const uint head = bh % H;
    const uint pos = uint(position[0]) + t;

    const uint src = row * D + d;
    const uint rope_index = pos * half_dim + d;

    const float c = rope_cos[rope_index];
    const float s = rope_sin[rope_index];

    const float q0 = q[src];
    const float q1 = q[src + half_dim];
    const float k0 = k[src];
    const float k1 = k[src + half_dim];

    const float rq0 = fma(q0, c, q1 * s);
    const float rq1 = fma(-q0, s, q1 * c);
    const float rk0 = fma(k0, c, k1 * s);
    const float rk1 = fma(-k0, s, k1 * c);

    // q/k 仍然只对应输入的单行 prefill。
    q[src] = rq0;
    q[src + half_dim] = rq1;

    k[src] = rk0;
    k[src + half_dim] = rk1;

    // [新增] 将同一 KV 行直接写入所有目标 batch。
    for (uint batch = 0u; batch < BC; ++batch) {
        const uint dst =
            ((batch * S + pos) * H + head) * D + d;

        kc[dst] = rk0;
        kc[dst + half_dim] = rk1;

        vc[dst] = v[src];
        vc[dst + half_dim] = v[src + half_dim];
    }
}

kernel void kv_cache_rope_store_rows(
    device float* q                 [[buffer(0)]],
    device float* k                 [[buffer(1)]],
    device const float* v           [[buffer(2)]],
    device float* kc                [[buffer(3)]],
    device float* vc                [[buffer(4)]],
    device const int* position      [[buffer(5)]],
    device const float* rope_cos    [[buffer(6)]],
    device const float* rope_sin    [[buffer(7)]],
    constant uint& B                [[buffer(8)]],
    constant uint& H                [[buffer(9)]],
    constant uint& T                [[buffer(10)]],
    constant uint& D                [[buffer(11)]],
    constant uint& S                [[buffer(12)]],
    uint3 group                     [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]]
) {
    const uint half_dim = D / 2u;
    const uint index = group.y * 128u + tid;

    if (index >= B * H * T * half_dim) {
        return;
    }

    const uint d = index % half_dim;
    const uint row = index / half_dim;
    const uint t = row % T;
    const uint bh = row / T;
    const uint batch = bh / H;
    const uint head = bh % H;
    const uint pos = uint(position[batch]) + t;

    const uint src = row * D + d;
    const uint dst = ((batch * S + pos) * H + head) * D + d;
    const uint rope_index = pos * half_dim + d;

    const float c = rope_cos[rope_index];
    const float s = rope_sin[rope_index];

    const float q0 = q[src];
    const float q1 = q[src + half_dim];
    const float k0 = k[src];
    const float k1 = k[src + half_dim];

    const float rq0 = fma(q0, c, q1 * s);
    const float rq1 = fma(-q0, s, q1 * c);
    const float rk0 = fma(k0, c, k1 * s);
    const float rk1 = fma(-k0, s, k1 * c);

    q[src] = rq0;
    q[src + half_dim] = rq1;

    k[src] = rk0;
    k[src + half_dim] = rk1;

    kc[dst] = rk0;
    kc[dst + half_dim] = rk1;

    vc[dst] = v[src];
    vc[dst + half_dim] = v[src + half_dim];
}

template <uint Capacity>
METAL_FUNC void kv_cache_fa_impl(
    device const float* q,
    device const float* kc,
    device const float* vc,
    device const int* position,
    device float* out,
    uint H,
    uint T,
    uint C,
    uint S,
    uint3 group,
    uint tid,
    uint lane,
    uint sg,
    threadgroup float* maxima,
    threadgroup float* sums,
    threadgroup float* partial)
{
    constexpr uint Parts = Capacity / 32u;
    const uint D = C / H;
    const uint query = group.y;
    const uint bh = group.z;
    const uint batch = bh / H;
    const uint head = bh % H;
    const uint limit = uint(position[0]) + query + 1u;

    const uint qbase = (bh * T + query) * D;
    const float scale = rsqrt(float(D));

    float qr[Parts];
    float acc[Parts];

    #pragma unroll
    for (uint i = 0u; i < Parts; ++i) {
        const uint d = lane + i * 32u;
        qr[i] = d < D ? q[qbase + d] : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    // 四个 SIMD group 分别处理交错的 key。
    for (uint key = sg; key < limit; key += 4u) {
        const uint base = ((batch * S + key) * H + head) * D;

        float dot = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < Parts; ++i) {
            const uint d = lane + i * 32u;
            if (d < D) {
                dot = fma(qr[i], kc[base + d], dot);
            }
        }

        const float score = simd_sum(dot) * scale;
        const float next_m = max(m, score);
        const float alpha = exp(m - next_m);
        const float weight = exp(score - next_m);

        l = fma(l, alpha, weight);

        #pragma unroll
        for (uint i = 0u; i < Parts; ++i) {
            const uint d = lane + i * 32u;
            if (d < D) {
                acc[i] = fma(weight, vc[base + d], acc[i] * alpha);
            }
        }

        m = next_m;
    }

    if (lane == 0u) {
        maxima[sg] = m;
        sums[sg] = l;
    }

    #pragma unroll
    for (uint i = 0u; i < Parts; ++i) {
        const uint d = lane + i * 32u;
        if (d < D) {
            partial[sg * Capacity + d] = acc[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < D) {
        const float global_m = max(
            max(maxima[0], maxima[1]),
            max(maxima[2], maxima[3])
        );

        float denominator = 0.0f;
        float numerator = 0.0f;

        #pragma unroll
        for (uint part = 0u; part < 4u; ++part) {
            // 很短的上下文可能使某个 SIMD group 没有 key。
            if (sums[part] > 0.0f) {
                const float factor = exp(maxima[part] - global_m);
                denominator = fma(sums[part], factor, denominator);
                numerator = fma(
                    partial[part * Capacity + tid],
                    factor,
                    numerator
                );
            }
        }

        // 输出保持 BTC，供已有 FusedLinearAddOp 使用。
        const uint dst = (batch * T + query) * C + head * D + tid;
        out[dst] = numerator / denominator;
    }
}

#define DEFINE_KV_CACHE_FA(NAME, CAPACITY)                         \
kernel void NAME(                                                 \
    device const float* q       [[buffer(0)]],                     \
    device const float* kc      [[buffer(1)]],                     \
    device const float* vc      [[buffer(2)]],                     \
    device const int* position  [[buffer(3)]],                     \
    device float* out           [[buffer(4)]],                     \
    constant uint& B            [[buffer(5)]],                     \
    constant uint& H            [[buffer(6)]],                     \
    constant uint& T            [[buffer(7)]],                     \
    constant uint& C            [[buffer(8)]],                     \
    constant uint& S            [[buffer(9)]],                     \
    uint3 group                 [[threadgroup_position_in_grid]],  \
    uint tid                    [[thread_index_in_threadgroup]],   \
    uint lane                   [[thread_index_in_simdgroup]],      \
    uint sg                     [[simdgroup_index_in_threadgroup]]) \
{                                                                 \
    threadgroup float maxima[4];                                  \
    threadgroup float sums[4];                                    \
    threadgroup float partial[4 * CAPACITY];                       \
    kv_cache_fa_impl<CAPACITY>(                                    \
        q, kc, vc, position, out, H, T, C, S,                      \
        group, tid, lane, sg, maxima, sums, partial);               \
}

DEFINE_KV_CACHE_FA(kv_cache_fa_forward_d64, 64)
DEFINE_KV_CACHE_FA(kv_cache_fa_forward_d128, 128)

#undef DEFINE_KV_CACHE_FA

kernel void kv_cache_fa_forward_rows_d64(
    device const float* q       [[buffer(0)]],
    device const float* kc      [[buffer(1)]],
    device const float* vc      [[buffer(2)]],
    device const int* position  [[buffer(3)]],
    device float* out           [[buffer(4)]],
    constant uint& B            [[buffer(5)]],
    constant uint& H            [[buffer(6)]],
    constant uint& T            [[buffer(7)]],
    constant uint& C            [[buffer(8)]],
    constant uint& S            [[buffer(9)]],
    uint3 group                 [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float maxima[4];
    threadgroup float sums[4];
    threadgroup float partial[4 * 64];
    kv_cache_fa_impl<64>(
        q, kc, vc, position + group.z / H, out, H, T, C, S,
        group, tid, lane, sg, maxima, sums, partial);
}

kernel void kv_cache_fa_forward_rows_d128(
    device const float* q       [[buffer(0)]],
    device const float* kc      [[buffer(1)]],
    device const float* vc      [[buffer(2)]],
    device const int* position  [[buffer(3)]],
    device float* out           [[buffer(4)]],
    constant uint& B            [[buffer(5)]],
    constant uint& H            [[buffer(6)]],
    constant uint& T            [[buffer(7)]],
    constant uint& C            [[buffer(8)]],
    constant uint& S            [[buffer(9)]],
    uint3 group                 [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float maxima[4];
    threadgroup float sums[4];
    threadgroup float partial[4 * 128];
    kv_cache_fa_impl<128>(
        q, kc, vc, position + group.z / H, out, H, T, C, S,
        group, tid, lane, sg, maxima, sums, partial);
}

kernel void fused_linear_ce_chunk_logits(
    device const float* x        [[buffer(0)]],
    device const float* w        [[buffer(1)]],
    device float* logits         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]],
    constant uint& C             [[buffer(4)]],
    constant uint& V             [[buffer(5)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint kBM = 64u;
    constexpr uint kBV = 96u;
    constexpr uint kBK = 16u;
    constexpr uint kFrag = 8u;
    constexpr uint kThreads = 256u;
    constexpr uint kXStride = 17u;
    constexpr uint kWStride = 97u;
    constexpr uint kC4PerBK = 4u;
    constexpr uint kAccCount = 12u;
    threadgroup float X_tg[kBM * kXStride];
    threadgroup float W_tg[kBK * kWStride];

    if (tg_pos.x != 0u)
        return;

    const uint row_block = tg_pos.y * kBM;
    const uint vocab_base = tg_pos.z * kBV;
    if (row_block >= rows || vocab_base >= V)
        return;

    const bool full_channel_tiles = (C % kBK) == 0u;
    const bool full_row_tile = row_block + kBM <= rows;
    const bool full_vocab_tile = vocab_base + kBV <= V;
    const bool full_x_tile = full_row_tile && full_channel_tiles;
    const bool full_w_tile = full_vocab_tile && full_channel_tiles;
    const uint simd_row_base = simd_id * kFrag;
    thread float2 acc[kAccCount];

    #pragma unroll
    for (uint i = 0u; i < kAccCount; ++i) {
        acc[i] = float2(0.0f);
    }
    for (uint channel_base = 0u; channel_base < C; channel_base += kBK) {
        constexpr uint kXFloat4Count = kBM * kC4PerBK;
        if (full_x_tile) {
            // 满 X tile：移除 row/C 边界判断。
            for (uint idx = tid; idx < kXFloat4Count; idx += kThreads) {
                const uint local_row = idx / kC4PerBK;
                const uint local_c4 = idx - local_row * kC4PerBK;
                const uint global_row = row_block + local_row;
                const uint global_c = channel_base + local_c4 * 4u;
                const float4 value = float4(*reinterpret_cast<device const packed_float4*>(x + global_row * C + global_c));
                *reinterpret_cast<threadgroup packed_float4*>(X_tg + local_row * kXStride + local_c4 * 4u) = packed_float4(value);
            }
        } else {
            for (uint idx = tid; idx < kXFloat4Count; idx += kThreads) {
                const uint local_row = idx / kC4PerBK;
                const uint local_c4 = idx - local_row * kC4PerBK;
                const uint global_row = row_block + local_row;
                const uint global_c = channel_base + local_c4 * 4u;
                float4 value = float4(0.0f);
                if (global_row < rows && global_c + 4u <= C) {
                    value = float4(*reinterpret_cast<device const packed_float4*>(x + global_row * C + global_c));
                } else if (global_row < rows) {
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) {
                        if (global_c + j < C) {
                            value[j] = x[global_row * C + global_c + j];
                        }
                    }
                }
                *reinterpret_cast<threadgroup packed_float4*>(X_tg + local_row * kXStride + local_c4 * 4u) = packed_float4(value);
            }
        }

        constexpr uint kWFloat4Count = kBV * kC4PerBK;
        if (full_w_tile) {
            // 满 W tile：移除 vocab/C 边界判断。
            for (uint idx = tid; idx < kWFloat4Count; idx += kThreads) {
                const uint local_vocab = idx / kC4PerBK;
                const uint local_c4 = idx - local_vocab * kC4PerBK;
                const uint global_vocab = vocab_base + local_vocab;
                const uint global_c = channel_base + local_c4 * 4u;
                const float4 value = float4(*reinterpret_cast<device const packed_float4*>(w + global_vocab * C + global_c));
                const uint local_c = local_c4 * 4u;
                W_tg[(local_c + 0u) * kWStride + local_vocab] = value[0];
                W_tg[(local_c + 1u) * kWStride + local_vocab] = value[1];
                W_tg[(local_c + 2u) * kWStride + local_vocab] = value[2];
                W_tg[(local_c + 3u) * kWStride + local_vocab] = value[3];
            }
        } else {
            for (uint idx = tid; idx < kWFloat4Count; idx += kThreads) {
                const uint local_vocab = idx / kC4PerBK;
                const uint local_c4 = idx - local_vocab * kC4PerBK;
                const uint global_vocab = vocab_base + local_vocab;
                const uint global_c = channel_base + local_c4 * 4u;
                float4 value = float4(0.0f);
                if (global_vocab < V && global_c + 4u <= C) {
                    value = float4(*reinterpret_cast<device const packed_float4*>(w + global_vocab * C + global_c));
                } else if (global_vocab < V) {
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) {
                        if (global_c + j < C) {
                            value[j] = w[global_vocab * C + global_c + j];
                        }
                    }
                }
                const uint local_c = local_c4 * 4u;
                W_tg[(local_c + 0u) * kWStride + local_vocab] = value[0];
                W_tg[(local_c + 1u) * kWStride + local_vocab] = value[1];
                W_tg[(local_c + 2u) * kWStride + local_vocab] = value[2];
                W_tg[(local_c + 3u) * kWStride + local_vocab] = value[3];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k_sub = 0u; k_sub < kBK; k_sub += kFrag) {
            const float2 x_fragment = lowreg_matrix::load_tg_8x8(X_tg, kXStride,
                ushort2(ushort(simd_row_base), ushort(k_sub)), ushort(lane));

            #pragma unroll
            for (uint i = 0u; i < kAccCount; ++i) {
                const float2 w_fragment = lowreg_matrix::load_tg_8x8(W_tg, kWStride, 
                    ushort2(ushort(k_sub), ushort(i * kFrag)), ushort(lane));
                acc[i] = lowreg_matrix::mma_8x8(x_fragment, w_fragment, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ushort2 output_position = morton_order(ushort(lane));
    const uint global_row = row_block + simd_row_base + uint(output_position.y);
    const uint local_col = uint(output_position.x);

    if (global_row >= rows)
        return;

    const uint output_row_base = global_row * V;
    if (full_vocab_tile) {
        #pragma unroll
        for (uint i = 0u; i < kAccCount; ++i) {
            const uint vocab_index = vocab_base + local_col + i * kFrag;
            *reinterpret_cast<device packed_float2*>(logits + output_row_base + vocab_index) = packed_float2(acc[i]);
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < kAccCount; ++i) {
            const uint vocab_index = vocab_base + local_col + i * kFrag;
            if (vocab_index + 1u < V) {
                *reinterpret_cast<device packed_float2*>(logits + output_row_base + vocab_index) = packed_float2(acc[i]);
            } else if (vocab_index < V) {
                logits[output_row_base + vocab_index] = acc[i][0];
            }
        }
    }
}

kernel void fused_linear_ce_logits_to_dlogits(
    device float* logits         [[buffer(0)]],
    device const int* targets    [[buffer(1)]],
    device float* row_loss       [[buffer(2)]],
    device float* lse_out        [[buffer(3)]],
    constant uint& rows          [[buffer(4)]],
    constant uint& V             [[buffer(5)]],
    constant uint& row_base      [[buffer(6)]],
    constant uint& scale_bits    [[buffer(7)]],
    constant uint& write_grad    [[buffer(8)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint  kThreads    = 256u;
    constexpr uint  kSimdgroups = kThreads / 32u;      // 8，与原版一致
    constexpr uint  kVecSpan    = kThreads * 4u;       // 1024，向量循环每轮跨度
    constexpr float kLog2E      = 1.4426950408889634f;
    threadgroup float partial_max[kSimdgroups];
    threadgroup float partial_sum[kSimdgroups];
    threadgroup float shared_max;
    threadgroup float shared_sum;
    const uint local_row = tg_pos.y;
    if (tg_pos.x != 0u || local_row >= rows) {
        return;
    }
    const uint global_row  = row_base + local_row;
    const uint logits_base = local_row * V;
    const uint vec_end     = (V / 4u) * 4u;            // 可整 float4 的前缀长度
    // ================= 第一遍：float4 无分支 online (max, sum) =================
    // 4 个分量各自维护独立的 (max, sum)，循环内零分支、零 divergence。
    float4 vmax = float4(-INFINITY);
    float4 vsum = float4(0.0f);
    for (uint base = tid * 4u; base + 4u <= V; base += kVecSpan) {
        const float4 x = float4(*reinterpret_cast<device const packed_float4*>(
            logits + logits_base + base));
        const float4 nm = max(vmax, x);
        // 首轮 vmax=-INF、x 有限：nm=x，(vmax-nm)=-INF，exp2(-INF)=0，vsum=0*0+1=1，不产生 NaN
        vsum = vsum * exp2((vmax - nm) * kLog2E) + exp2((x - nm) * kLog2E);
        vmax = nm;
    }
    // 4 个分量合并成标量 (max, sum)；sum==0 表示该分量无元素，语义与原版哨兵一致
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float om = vmax[j];
        const float os = vsum[j];
        if (os > 0.0f) {
            const float nm = max(local_max, om);
            local_sum = local_sum * exp2((local_max - nm) * kLog2E) +
                        os * exp2((om - nm) * kLog2E);
            local_max = nm;
        }
    }
    // 标量尾：V 非 4 的倍数时剩下的不足 4 个元素（V=32768 时该循环为空）
    for (uint p = vec_end + tid; p < V; p += kThreads) {
        const float x  = logits[logits_base + p];
        const float nm = max(local_max, x);
        local_sum = local_sum * exp2((local_max - nm) * kLog2E) +
                    exp2((x - nm) * kLog2E);
        local_max = nm;
    }
    // ============ 以下两级归约与原版逐字一致，未做任何改动 ============
    #pragma unroll
    for (ushort delta = 16u; delta > 0u; delta >>= 1u) {
        const float other_max = simd_shuffle_down(local_max, delta);
        const float other_sum = simd_shuffle_down(local_sum, delta);
        if (lane + uint(delta) < 32u && other_sum > 0.0f) {
            if (local_sum == 0.0f) {
                local_max = other_max;
                local_sum = other_sum;
            } else {
                const float merged_max = max(local_max, other_max);
                local_sum = local_sum * exp2((local_max - merged_max) * kLog2E) +
                    other_sum * exp2((other_max - merged_max) * kLog2E);
                local_max = merged_max;
            }
        }
    }
    if (lane == 0u) {
        partial_max[simd_id] = local_max;
        partial_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float group_max = lane < kSimdgroups ? partial_max[lane] : -INFINITY;
        float group_sum = lane < kSimdgroups ? partial_sum[lane] : 0.0f;
        #pragma unroll
        for (ushort delta = 4u; delta > 0u; delta >>= 1u) {
            const float other_max = simd_shuffle_down(group_max, delta);
            const float other_sum = simd_shuffle_down(group_sum, delta);
            if (lane + uint(delta) < kSimdgroups && other_sum > 0.0f) {
                if (group_sum == 0.0f) {
                    group_max = other_max;
                    group_sum = other_sum;
                } else {
                    const float merged_max = max(group_max, other_max);
                    group_sum = group_sum * exp2((group_max - merged_max) * kLog2E) +
                        other_sum * exp2((other_max - merged_max) * kLog2E);
                    group_max = merged_max;
                }
            }
        }
        if (lane == 0u) {
            shared_max = group_max;
            shared_sum = group_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int   target        = targets[global_row];
    const bool  target_valid  = target >= 0 && target < int(V) && shared_sum > 0.0f;
    const float inverse_sum   = target_valid ? 1.0f / shared_sum : 0.0f;
    if (tid == 0u) {
        const float target_logit = target_valid ? logits[logits_base + uint(target)] : 0.0f;
        const float lse_value    = shared_max + log(shared_sum);
        row_loss[global_row] = target_valid ? (lse_value - target_logit) : 0.0f;
        lse_out[global_row]  = target_valid ? lse_value : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // 保护 tid0 读 target logit 先于覆写
    if (write_grad == 0u) return;
    // ================= 第二遍：float4 读 + float4 写 =================
    const float scale    = as_type<float>(scale_bits);
    const uint  target_u = target_valid ? uint(target) : V;   // V 作哨兵，永不命中
    for (uint base = tid * 4u; base + 4u <= V; base += kVecSpan) {
        float4 out = float4(0.0f);
        if (target_valid) {
            const float4 x = float4(*reinterpret_cast<device const packed_float4*>(
                logits + logits_base + base));
            out = exp2((x - shared_max) * kLog2E) * inverse_sum;
            if (target_u >= base && target_u < base + 4u) {
                const uint jj = target_u - base;
                out -= float4(float(jj == 0u), float(jj == 1u),
                              float(jj == 2u), float(jj == 3u));
            }
            out *= scale;
        }
        *reinterpret_cast<device packed_float4*>(logits + logits_base + base) =
            packed_float4(out);
    }
    // 标量尾（V=32768 时为空）
    for (uint p = vec_end + tid; p < V; p += kThreads) {
        float value = 0.0f;
        if (target_valid) {
            value = exp2((logits[logits_base + p] - shared_max) * kLog2E) * inverse_sum;
            if (p == target_u) {
                value -= 1.0f;
            }
            value *= scale;
        }
        logits[logits_base + p] = value;
    }
}

kernel void fused_linear_ce_logits_to_dlogits_row_scaled(
    device float* logits         [[buffer(0)]],
    device const int* targets    [[buffer(1)]],
    device float* row_loss       [[buffer(2)]],
    device float* lse_out        [[buffer(3)]],
    device const float* row_scale [[buffer(4)]],
    constant uint& rows          [[buffer(5)]],
    constant uint& V             [[buffer(6)]],
    constant uint& row_base      [[buffer(7)]],
    constant uint& total_rows [[buffer(8)]],
    constant uint& write_grad [[buffer(9)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint  kThreads    = 256u;
    constexpr uint  kSimdgroups = kThreads / 32u;      // 8，与原版一致
    constexpr uint  kVecSpan    = kThreads * 4u;       // 1024，向量循环每轮跨度
    constexpr float kLog2E      = 1.4426950408889634f;
    threadgroup float partial_max[kSimdgroups];
    threadgroup float partial_sum[kSimdgroups];
    threadgroup float shared_max;
    threadgroup float shared_sum;
    const uint local_row = tg_pos.y;
    if (tg_pos.x != 0u || local_row >= rows) {
        return;
    }
    const uint global_row  = row_base + local_row;
    const uint logits_base = local_row * V;
    const uint vec_end     = (V / 4u) * 4u;            // 可整 float4 的前缀长度
    // ================= 第一遍：float4 无分支 online (max, sum) =================
    // 4 个分量各自维护独立的 (max, sum)，循环内零分支、零 divergence。
    float4 vmax = float4(-INFINITY);
    float4 vsum = float4(0.0f);
    for (uint base = tid * 4u; base + 4u <= V; base += kVecSpan) {
        const float4 x = float4(*reinterpret_cast<device const packed_float4*>(
            logits + logits_base + base));
        const float4 nm = max(vmax, x);
        // 首轮 vmax=-INF、x 有限：nm=x，(vmax-nm)=-INF，exp2(-INF)=0，vsum=0*0+1=1，不产生 NaN
        vsum = vsum * exp2((vmax - nm) * kLog2E) + exp2((x - nm) * kLog2E);
        vmax = nm;
    }
    // 4 个分量合并成标量 (max, sum)；sum==0 表示该分量无元素，语义与原版哨兵一致
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float om = vmax[j];
        const float os = vsum[j];
        if (os > 0.0f) {
            const float nm = max(local_max, om);
            local_sum = local_sum * exp2((local_max - nm) * kLog2E) +
                        os * exp2((om - nm) * kLog2E);
            local_max = nm;
        }
    }
    // 标量尾：V 非 4 的倍数时剩下的不足 4 个元素（V=32768 时该循环为空）
    for (uint p = vec_end + tid; p < V; p += kThreads) {
        const float x  = logits[logits_base + p];
        const float nm = max(local_max, x);
        local_sum = local_sum * exp2((local_max - nm) * kLog2E) +
                    exp2((x - nm) * kLog2E);
        local_max = nm;
    }
    // ============ 以下两级归约与原版逐字一致，未做任何改动 ============
    #pragma unroll
    for (ushort delta = 16u; delta > 0u; delta >>= 1u) {
        const float other_max = simd_shuffle_down(local_max, delta);
        const float other_sum = simd_shuffle_down(local_sum, delta);
        if (lane + uint(delta) < 32u && other_sum > 0.0f) {
            if (local_sum == 0.0f) {
                local_max = other_max;
                local_sum = other_sum;
            } else {
                const float merged_max = max(local_max, other_max);
                local_sum = local_sum * exp2((local_max - merged_max) * kLog2E) +
                    other_sum * exp2((other_max - merged_max) * kLog2E);
                local_max = merged_max;
            }
        }
    }
    if (lane == 0u) {
        partial_max[simd_id] = local_max;
        partial_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float group_max = lane < kSimdgroups ? partial_max[lane] : -INFINITY;
        float group_sum = lane < kSimdgroups ? partial_sum[lane] : 0.0f;
        #pragma unroll
        for (ushort delta = 4u; delta > 0u; delta >>= 1u) {
            const float other_max = simd_shuffle_down(group_max, delta);
            const float other_sum = simd_shuffle_down(group_sum, delta);
            if (lane + uint(delta) < kSimdgroups && other_sum > 0.0f) {
                if (group_sum == 0.0f) {
                    group_max = other_max;
                    group_sum = other_sum;
                } else {
                    const float merged_max = max(group_max, other_max);
                    group_sum = group_sum * exp2((group_max - merged_max) * kLog2E) +
                        other_sum * exp2((other_max - merged_max) * kLog2E);
                    group_max = merged_max;
                }
            }
        }
        if (lane == 0u) {
            shared_max = group_max;
            shared_sum = group_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int   target        = targets[global_row];
    const bool  target_valid  = target >= 0 && target < int(V) && shared_sum > 0.0f;
    const float inverse_sum   = target_valid ? 1.0f / shared_sum : 0.0f;
    const float scale = row_scale[global_row];
    if (tid == 0u) {
        const float target_logit = target_valid ? logits[logits_base + uint(target)] : 0.0f;
        const float lse_value    = shared_max + log(shared_sum);
        row_loss[global_row] = target_valid ? (lse_value - target_logit) * scale * float(total_rows) : 0.0f;
        lse_out[global_row]  = target_valid ? lse_value : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // 保护 tid0 读 target logit 先于覆写
    if (write_grad == 0u) return;
    // ================= 第二遍：float4 读 + float4 写 =================
    const uint  target_u = target_valid ? uint(target) : V;   // V 作哨兵，永不命中
    for (uint base = tid * 4u; base + 4u <= V; base += kVecSpan) {
        float4 out = float4(0.0f);
        if (target_valid) {
            const float4 x = float4(*reinterpret_cast<device const packed_float4*>(
                logits + logits_base + base));
            out = exp2((x - shared_max) * kLog2E) * inverse_sum;
            if (target_u >= base && target_u < base + 4u) {
                const uint jj = target_u - base;
                out -= float4(float(jj == 0u), float(jj == 1u),
                              float(jj == 2u), float(jj == 3u));
            }
            out *= scale;
        }
        *reinterpret_cast<device packed_float4*>(logits + logits_base + base) =
            packed_float4(out);
    }
    // 标量尾（V=32768 时为空）
    for (uint p = vec_end + tid; p < V; p += kThreads) {
        float value = 0.0f;
        if (target_valid) {
            value = exp2((logits[logits_base + p] - shared_max) * kLog2E) * inverse_sum;
            if (p == target_u) {
                value -= 1.0f;
            }
            value *= scale;
        }
        logits[logits_base + p] = value;
    }
}

kernel void fused_linear_ce_forward_reduce(
    device const float* row_loss [[buffer(0)]],
    device float* out_loss       [[buffer(1)]],
    constant uint& N             [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    // 单 TG 归约 N（N 通常 <= 几 K；更大时可再加 multi-TG atomic）
    threadgroup float buf[256];
    float local = 0.0f;
    for (uint i = tid; i < N; i += tpg) {
        local += row_loss[i];
    }
    buf[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) buf[tid] += buf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_loss[0] = (N > 0) ? (buf[0] / float(N)) : 0.0f;
    }
}

inline float canonicalize_float(float f) {
    if (f != f) return -INFINITY;
    return (f == 0.0f) ? 0.0f : f;
}
inline uint float_to_sortable_asc_raw(float canon_f) {
    uint bits = as_type<uint>(canon_f);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}
inline uint float_to_sortable_asc(float f) {
    return float_to_sortable_asc_raw(canonicalize_float(f));
}
inline float sortable_asc_to_float(uint key) {
    uint mask = (key & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return as_type<float>(key ^ mask);
}
inline bool key_before(float val_a, int idx_a, float val_b, int idx_b) {
    float a = canonicalize_float(val_a);
    float b = canonicalize_float(val_b);
    if (a != b) return a > b;
    return idx_a < idx_b;
}
inline bool heap_less(float a_val, int a_idx, float b_val, int b_idx) {
    return key_before(b_val, b_idx, a_val, a_idx);
}
inline void heap_sift_down(threadgroup float* vals, threadgroup int* idxs, uint n) {
    uint i = 0u;
    while (true) {
        uint l = 2u*i+1u, r = 2u*i+2u, s = i;
        if (l < n && heap_less(vals[l], idxs[l], vals[s], idxs[s])) s = l;
        if (r < n && heap_less(vals[r], idxs[r], vals[s], idxs[s])) s = r;
        if (s == i) break;
        float tv = vals[i]; vals[i] = vals[s]; vals[s] = tv;
        int   ti = idxs[i]; idxs[i] = idxs[s]; idxs[s] = ti;
        i = s;
    }
}
inline void heap_replace_if_better(threadgroup float* vals, threadgroup int* idxs, uint n, float value, int index) {
    if (n == 0u || !heap_less(vals[0], idxs[0], value, index)) return;
    vals[0] = value; idxs[0] = index;
    heap_sift_down(vals, idxs, n);
}
inline void heap_sort_descending(threadgroup float* vals, threadgroup int* idxs, uint n) {
    for (uint size = n; size > 1u; --size) {
        float tv = vals[0]; vals[0] = vals[size-1u]; vals[size-1u] = tv;
        int   ti = idxs[0]; idxs[0] = idxs[size-1u]; idxs[size-1u] = ti;
        uint m = size - 1u, i = 0u;
        while (true) {
            uint l = 2u*i+1u, r = 2u*i+2u, s = i;
            if (l < m && heap_less(vals[l], idxs[l], vals[s], idxs[s])) s = l;
            if (r < m && heap_less(vals[r], idxs[r], vals[s], idxs[s])) s = r;
            if (s == i) break;
            float tv2 = vals[i]; vals[i] = vals[s]; vals[s] = tv2;
            int   ti2 = idxs[i]; idxs[i] = idxs[s]; idxs[s] = ti2;
            i = s;
        }
    }
}
inline void idx_heap_sift_down(threadgroup int* idxs, uint n) {
    uint i = 0u;
    while (true) {
        uint l = 2u*i+1u, r = 2u*i+2u, s = i;
        if (l < n && idxs[l] > idxs[s]) s = l;
        if (r < n && idxs[r] > idxs[s]) s = r;
        if (s == i) break;
        int t = idxs[i]; idxs[i] = idxs[s]; idxs[s] = t;
        i = s;
    }
}
inline void idx_heap_replace_if_better(threadgroup int* idxs, uint n, int index) {
    if (n == 0u || index >= idxs[0]) return;
    idxs[0] = index;
    idx_heap_sift_down(idxs, n);
}
inline void idx_heap_sort_ascending(threadgroup int* idxs, uint n) {
    for (uint size = n; size > 1u; --size) {
        int t = idxs[0]; idxs[0] = idxs[size-1u]; idxs[size-1u] = t;
        uint m = size - 1u, i = 0u;
        while (true) {
            uint l = 2u*i+1u, r = 2u*i+2u, s = i;
            if (l < m && idxs[l] > idxs[s]) s = l;
            if (r < m && idxs[r] > idxs[s]) s = r;
            if (s == i) break;
            int t2 = idxs[i]; idxs[i] = idxs[s]; idxs[s] = t2;
            i = s;
        }
    }
}

inline void insertion_sort_by_key_before(threadgroup float* vals, threadgroup int* idxs, uint n) {
    for (uint i = 1u; i < n; ++i) {
        float v = vals[i]; int idx = idxs[i];
        int j = int(i) - 1;
        while (j >= 0 && key_before(v, idx, vals[j], idxs[j])) {
            vals[j+1] = vals[j]; idxs[j+1] = idxs[j]; --j;
        }
        vals[j+1] = v; idxs[j+1] = idx;
    }
}

/*
kernel void rollout_topk_select(
    device const float* logits         [[buffer(0)]],
    device float* topk_values          [[buffer(1)]],
    device int* topk_indices           [[buffer(2)]],
    device const int* runtime_top_k    [[buffer(3)]],
    constant uint& B                   [[buffer(4)]],
    constant uint& V                   [[buffer(5)]],
    constant uint& K_MAX               [[buffer(6)]],
    uint global_id                     [[thread_position_in_grid]],
    uint tid_in_tg                     [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]]
) {
    constexpr uint TG_SIZE     = 128;
    constexpr uint MAX_K       = 64;
    constexpr uint NUM_BUCKETS = 16;
    constexpr uint MAX_PASSES  = 8;
    constexpr uint AMBIG_CAP   = 256;   // 提前收敛阈值：候选桶足够小就停止切分bit

    const uint row = global_id / TG_SIZE;
    const uint tid = tid_in_tg;
    if (row >= B) return;

    device const float* row_logits = logits + (size_t)row * V;

    threadgroup atomic_uint hist[NUM_BUCKETS];
    threadgroup uint prefix_bits;
    threadgroup uint prefix_nbits;
    threadgroup uint remaining_k;
    threadgroup uint bucket_count;
    threadgroup bool  should_stop;

    const uint K = min(min(uint(runtime_top_k[0]), K_MAX), V);

    if (tid == 0u) {
        for (uint d = 0u; d < NUM_BUCKETS; ++d) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        prefix_bits = 0u; prefix_nbits = 0u; remaining_k = K; bucket_count = V; should_stop = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 阶段1：自适应基数选择，候选桶足够小就提前退出 ============
    for (uint pass = 0u; pass < MAX_PASSES; ++pass) {
        const uint shift      = (7u - pass) * 4u;
        const uint cur_prefix = prefix_bits;
        const uint cur_nbits  = prefix_nbits;

        uint local_hist[NUM_BUCKETS];
        for (uint d = 0u; d < NUM_BUCKETS; ++d) local_hist[d] = 0u;

        for (uint index = tid; index < V; index += TG_SIZE) {
            const uint key = float_to_sortable_asc(row_logits[index]);
            bool match = true;
            if (cur_nbits > 0u) {
                const uint pshift = 32u - cur_nbits;
                match = ((key >> pshift) == (cur_prefix >> pshift));
            }
            if (match) local_hist[(key >> shift) & 0xFu] += 1u;
        }
        for (uint d = 0u; d < NUM_BUCKETS; ++d) {
            uint s = simd_sum(local_hist[d]);
            if (simd_lane == 0u) atomic_fetch_add_explicit(&hist[d], s, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint rk = remaining_k, chosen_digit = 0u, chosen_count = 0u;
            for (int d = int(NUM_BUCKETS) - 1; d >= 0; --d) {
                uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (rk > c) { rk -= c; } else { chosen_digit = uint(d); chosen_count = c; break; }
            }
            remaining_k  = rk;
            prefix_bits  = cur_prefix | (chosen_digit << shift);
            prefix_nbits = cur_nbits + 4u;
            bucket_count = chosen_count;

            // ---- 提前收敛判定：候选范围已经足够小/已无需继续切分 ----
            should_stop = (rk == 0u) || (chosen_count <= AMBIG_CAP);

            if (pass + 1u < MAX_PASSES && !should_stop) {
                for (uint d = 0u; d < NUM_BUCKETS; ++d) {
                    atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (should_stop) break;
    }

    const uint final_pshift  = 32u - prefix_nbits;   // 若跑满8轮，nbits=32，pshift=0
    const uint need_definite = K - remaining_k;

    threadgroup float       definite_v[MAX_K];
    threadgroup int         definite_i[MAX_K];
    threadgroup atomic_uint definite_count;
    if (tid == 0u) atomic_store_explicit(&definite_count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 情形A：remaining_k==0，无需处理任何tie，最快路径 ============
    if (remaining_k == 0u) {
        for (uint index = tid; index < V; index += TG_SIZE) {
            const float value = canonicalize_float(row_logits[index]);
            const uint  key   = float_to_sortable_asc_raw(value);
            if ((key >> final_pshift) > (prefix_bits >> final_pshift)) {
                uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);
            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
        return;
    }

    // ============ 情形B：提前收敛，ambiguous组有界(<=AMBIG_CAP)，做精确value比较选择 ============
    if (should_stop) {
        threadgroup float       ambig_v[AMBIG_CAP];
        threadgroup int         ambig_i[AMBIG_CAP];
        threadgroup atomic_uint ambig_count;
        if (tid == 0u) atomic_store_explicit(&ambig_count, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid; index < V; index += TG_SIZE) {
            const float value = canonicalize_float(row_logits[index]);
            const uint  key   = float_to_sortable_asc_raw(value);
            const uint  a = key >> final_pshift, b = prefix_bits >> final_pshift;
            if (a > b) {
                uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
            } else if (a == b) {
                uint pos = atomic_fetch_add_explicit(&ambig_count, 1u, memory_order_relaxed);
                if (pos < AMBIG_CAP) { ambig_v[pos] = value; ambig_i[pos] = int(index); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);

            threadgroup float tie_v[MAX_K];
            threadgroup int   tie_i[MAX_K];
            for (uint k = 0u; k < remaining_k; ++k) { tie_v[k] = -INFINITY; tie_i[k] = 0; }
            uint use_n = min(atomic_load_explicit(&ambig_count, memory_order_relaxed), (uint)AMBIG_CAP);
            for (uint i = 0u; i < use_n; ++i) {
                heap_replace_if_better(tie_v, tie_i, remaining_k, ambig_v[i], ambig_i[i]);
            }
            heap_sort_descending(tie_v, tie_i, remaining_k);

            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else if (k < K) {
                    topk_values[row*K_MAX+k] = tie_v[k-need_definite]; topk_indices[row*K_MAX+k] = tie_i[k-need_definite];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
        return;
    }

    // ============ 情形C：极端全等值兜底（跑满8轮仍未收敛，说明存在海量位级完全相同的值）============
    {
        constexpr uint ITEMS     = 8u;
        constexpr uint TILE_SIZE = TG_SIZE * ITEMS;
        const uint boundary_key  = prefix_bits;

        threadgroup int tie_idx[MAX_K];
        if (tid == 0u) { for (uint k = 0u; k < MAX_K; ++k) tie_idx[k] = 0x7FFFFFFF; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float surv_v[TG_SIZE * ITEMS];
        threadgroup int   surv_i[TG_SIZE * ITEMS];
        threadgroup uint  surv_n[TG_SIZE];

        for (uint tile = 0u; tile < V; tile += TILE_SIZE) {
            uint n = 0u;
            for (uint j = 0u; j < ITEMS; ++j) {
                const uint index = tile + j*TG_SIZE + tid;
                if (index < V) {
                    const float value = canonicalize_float(row_logits[index]);
                    const uint  key   = float_to_sortable_asc_raw(value);
                    if (key > boundary_key) {
                        uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                        if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
                    } else if (key == boundary_key) {
                        surv_v[tid+n*TG_SIZE] = value; surv_i[tid+n*TG_SIZE] = int(index); ++n;
                    }
                }
            }
            surv_n[tid] = n;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint t = 0u; t < TG_SIZE; ++t) {
                    uint nn = surv_n[t];
                    for (uint s = 0u; s < nn; ++s) idx_heap_replace_if_better(tie_idx, remaining_k, surv_i[t+s*TG_SIZE]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);
            idx_heap_sort_ascending(tie_idx, remaining_k);
            const float tie_value = sortable_asc_to_float(boundary_key);
            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else if (k < K) {
                    topk_values[row*K_MAX+k] = tie_value; topk_indices[row*K_MAX+k] = tie_idx[k-need_definite];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
    }
}
*/
kernel void rollout_topk_select(
    device const float* logits          [[buffer(0)]],
    device float* topk_values           [[buffer(1)]],
    device int* topk_indices            [[buffer(2)]],
    device const int* runtime_top_k     [[buffer(3)]],
    device const int* logits_position   [[buffer(4)]],
    constant uint& B                    [[buffer(5)]],
    constant uint& V                    [[buffer(6)]],
    constant uint& K_MAX                [[buffer(7)]],
    constant uint& mode_id              [[buffer(8)]],
    constant uint& T                    [[buffer(9)]],
    uint global_id                      [[thread_position_in_grid]],
    uint tid_in_tg                      [[thread_position_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]]
) {
    constexpr uint TG_SIZE     = 128;
    constexpr uint MAX_K       = 64;
    constexpr uint NUM_BUCKETS = 16;
    constexpr uint MAX_PASSES  = 8;
    constexpr uint AMBIG_CAP   = 256;   // 提前收敛阈值：候选桶足够小就停止切分bit

    const uint row = global_id / TG_SIZE;
    const uint tid = tid_in_tg;
    if (row >= B) return;

    const uint requested_position = 0u;

    if (requested_position >= T) {
        if (tid == 0u) {
            for (uint k = 0u; k < K_MAX; ++k) {
                topk_values[row * K_MAX + k] = -INFINITY;
                topk_indices[row * K_MAX + k] = 0;
            }
        }
        return;
    }

    const size_t row_offset = (row * T + requested_position) * V;
    device const float* row_logits = logits + row_offset;

    threadgroup atomic_uint hist[NUM_BUCKETS];
    threadgroup uint prefix_bits;
    threadgroup uint prefix_nbits;
    threadgroup uint remaining_k;
    threadgroup uint bucket_count;
    threadgroup bool  should_stop;

    const uint K = min(min(uint(max(runtime_top_k[0], 1)), K_MAX), V);

    if (tid == 0u) {
        for (uint d = 0u; d < NUM_BUCKETS; ++d) atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
        prefix_bits = 0u; prefix_nbits = 0u; remaining_k = K; bucket_count = V; should_stop = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 阶段1：自适应基数选择，候选桶足够小就提前退出 ============
    for (uint pass = 0u; pass < MAX_PASSES; ++pass) {
        const uint shift      = (7u - pass) * 4u;
        const uint cur_prefix = prefix_bits;
        const uint cur_nbits  = prefix_nbits;

        uint local_hist[NUM_BUCKETS];
        for (uint d = 0u; d < NUM_BUCKETS; ++d) local_hist[d] = 0u;

        for (uint index = tid; index < V; index += TG_SIZE) {
            const uint key = float_to_sortable_asc(row_logits[index]);
            bool match = true;
            if (cur_nbits > 0u) {
                const uint pshift = 32u - cur_nbits;
                match = ((key >> pshift) == (cur_prefix >> pshift));
            }
            if (match) local_hist[(key >> shift) & 0xFu] += 1u;
        }
        for (uint d = 0u; d < NUM_BUCKETS; ++d) {
            uint s = simd_sum(local_hist[d]);
            if (simd_lane == 0u) atomic_fetch_add_explicit(&hist[d], s, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint rk = remaining_k, chosen_digit = 0u, chosen_count = 0u;
            for (int d = int(NUM_BUCKETS) - 1; d >= 0; --d) {
                uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (rk > c) { rk -= c; } else { chosen_digit = uint(d); chosen_count = c; break; }
            }
            remaining_k  = rk;
            prefix_bits  = cur_prefix | (chosen_digit << shift);
            prefix_nbits = cur_nbits + 4u;
            bucket_count = chosen_count;

            // ---- 提前收敛判定：候选范围已经足够小/已无需继续切分 ----
            should_stop = (rk == 0u) || (chosen_count <= AMBIG_CAP);

            if (pass + 1u < MAX_PASSES && !should_stop) {
                for (uint d = 0u; d < NUM_BUCKETS; ++d) {
                    atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (should_stop) break;
    }

    const uint final_pshift  = 32u - prefix_nbits;   // 若跑满8轮，nbits=32，pshift=0
    const uint need_definite = K - remaining_k;

    threadgroup float       definite_v[MAX_K];
    threadgroup int         definite_i[MAX_K];
    threadgroup atomic_uint definite_count;
    if (tid == 0u) atomic_store_explicit(&definite_count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 情形A：remaining_k==0，无需处理任何tie，最快路径 ============
    if (remaining_k == 0u) {
        for (uint index = tid; index < V; index += TG_SIZE) {
            const float value = canonicalize_float(row_logits[index]);
            const uint  key   = float_to_sortable_asc_raw(value);
            if ((key >> final_pshift) > (prefix_bits >> final_pshift)) {
                uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);
            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
        return;
    }

    // ============ 情形B：提前收敛，ambiguous组有界(<=AMBIG_CAP)，做精确value比较选择 ============
    if (should_stop) {
        threadgroup float       ambig_v[AMBIG_CAP];
        threadgroup int         ambig_i[AMBIG_CAP];
        threadgroup atomic_uint ambig_count;
        if (tid == 0u) atomic_store_explicit(&ambig_count, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid; index < V; index += TG_SIZE) {
            const float value = canonicalize_float(row_logits[index]);
            const uint  key   = float_to_sortable_asc_raw(value);
            const uint  a = key >> final_pshift, b = prefix_bits >> final_pshift;
            if (a > b) {
                uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
            } else if (a == b) {
                uint pos = atomic_fetch_add_explicit(&ambig_count, 1u, memory_order_relaxed);
                if (pos < AMBIG_CAP) { ambig_v[pos] = value; ambig_i[pos] = int(index); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);

            threadgroup float tie_v[MAX_K];
            threadgroup int   tie_i[MAX_K];
            for (uint k = 0u; k < remaining_k; ++k) { tie_v[k] = -INFINITY; tie_i[k] = 0; }
            uint use_n = min(atomic_load_explicit(&ambig_count, memory_order_relaxed), (uint)AMBIG_CAP);
            for (uint i = 0u; i < use_n; ++i) {
                heap_replace_if_better(tie_v, tie_i, remaining_k, ambig_v[i], ambig_i[i]);
            }
            heap_sort_descending(tie_v, tie_i, remaining_k);

            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else if (k < K) {
                    topk_values[row*K_MAX+k] = tie_v[k-need_definite]; topk_indices[row*K_MAX+k] = tie_i[k-need_definite];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
        return;
    }

    // ============ 情形C：极端全等值兜底（跑满8轮仍未收敛，说明存在海量位级完全相同的值）============
    {
        constexpr uint ITEMS     = 8u;
        constexpr uint TILE_SIZE = TG_SIZE * ITEMS;
        const uint boundary_key  = prefix_bits;

        threadgroup int tie_idx[MAX_K];
        if (tid == 0u) { for (uint k = 0u; k < MAX_K; ++k) tie_idx[k] = 0x7FFFFFFF; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float surv_v[TG_SIZE * ITEMS];
        threadgroup int   surv_i[TG_SIZE * ITEMS];
        threadgroup uint  surv_n[TG_SIZE];

        for (uint tile = 0u; tile < V; tile += TILE_SIZE) {
            uint n = 0u;
            for (uint j = 0u; j < ITEMS; ++j) {
                const uint index = tile + j*TG_SIZE + tid;
                if (index < V) {
                    const float value = canonicalize_float(row_logits[index]);
                    const uint  key   = float_to_sortable_asc_raw(value);
                    if (key > boundary_key) {
                        uint pos = atomic_fetch_add_explicit(&definite_count, 1u, memory_order_relaxed);
                        if (pos < need_definite) { definite_v[pos] = value; definite_i[pos] = int(index); }
                    } else if (key == boundary_key) {
                        surv_v[tid+n*TG_SIZE] = value; surv_i[tid+n*TG_SIZE] = int(index); ++n;
                    }
                }
            }
            surv_n[tid] = n;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint t = 0u; t < TG_SIZE; ++t) {
                    uint nn = surv_n[t];
                    for (uint s = 0u; s < nn; ++s) idx_heap_replace_if_better(tie_idx, remaining_k, surv_i[t+s*TG_SIZE]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            insertion_sort_by_key_before(definite_v, definite_i, need_definite);
            idx_heap_sort_ascending(tie_idx, remaining_k);
            const float tie_value = sortable_asc_to_float(boundary_key);
            for (uint k = 0u; k < K_MAX; ++k) {
                if (k < need_definite) {
                    topk_values[row*K_MAX+k] = definite_v[k]; topk_indices[row*K_MAX+k] = definite_i[k];
                } else if (k < K) {
                    topk_values[row*K_MAX+k] = tie_value; topk_indices[row*K_MAX+k] = tie_idx[k-need_definite];
                } else {
                    topk_values[row*K_MAX+k] = -INFINITY; topk_indices[row*K_MAX+k] = 0;
                }
            }
        }
    }
}


inline uint rollout_xorshift32(thread uint& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

inline float rollout_uniform01(thread uint& state) {
    uint value = rollout_xorshift32(state);
    return (float(value) + 1.0f) / 4294967297.0f;
}

/*
kernel void rollout_sample_from_topk(
    device const float* topk_values  [[buffer(0)]],
    device const int* topk_indices   [[buffer(1)]],
    device uint* rng_state           [[buffer(2)]],
    device const int* finished       [[buffer(3)]],
    device int* next_tokens          [[buffer(4)]],
    device const float* temperature  [[buffer(5)]],
    device const int* runtime_top_k  [[buffer(6)]],
    constant uint& B                 [[buffer(7)]],
    constant uint& K_MAX             [[buffer(8)]],
    constant int& eos_token          [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= B) {
        return;
    }

    if (finished[gid] != 0) {
        next_tokens[gid] = eos_token;
        return;
    }

    const uint K = min(uint(runtime_top_k[0]), K_MAX);
    const float temp = temperature[0];

    device const float* values =
        topk_values + gid * K_MAX;
    device const int* indices =
        topk_indices + gid * K_MAX;

    if (temp <= 0.0f) {
        next_tokens[gid] = indices[0];
        return;
    }

    float maximum = -INFINITY;

    for (uint k = 0u; k < K; ++k) {
        maximum = max(maximum, values[k] / temp);
    }

    float total = 0.0f;

    for (uint k = 0u; k < K; ++k) {
        total += exp(values[k] / temp - maximum);
    }

    uint state = rng_state[gid * 4u];

    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;

    rng_state[gid * 4u] = state;

    const float u =
        (float(state) + 1.0f) / 4294967297.0f;

    const float threshold = u * total;

    float cumulative = 0.0f;
    int selected = indices[K - 1u];

    for (uint k = 0u; k < K; ++k) {
        cumulative += exp(values[k] / temp - maximum);

        if (cumulative >= threshold) {
            selected = indices[k];
            break;
        }
    }

    next_tokens[gid] = selected;
}
*/
inline uint2 rollout_philox_mulhilo(
    uint a,
    uint b
) {
    ulong product = ulong(a) * ulong(b);

    return uint2(
        uint(product >> 32),
        uint(product & 0xffffffffu)
    );
}

inline uint4 rollout_philox4x32_10(
    ulong counter,
    uint key0,
    uint key1
) {
    uint c0 = uint(counter & 0xfffffffful);
    uint c1 = uint(counter >> 32);
    uint c2 = 0u;
    uint c3 = 0u;

    constexpr uint M0 = 0xD2511F53u;
    constexpr uint M1 = 0xCD9E8D57u;
    constexpr uint W0 = 0x9E3779B9u;
    constexpr uint W1 = 0xBB67AE85u;

    for (uint round = 0u; round < 10u; ++round) {
        uint2 p0 = rollout_philox_mulhilo(M0, c0);
        uint2 p1 = rollout_philox_mulhilo(M1, c2);

        uint next_c0 = p1.x ^ c1 ^ key0;
        uint next_c1 = p1.y;
        uint next_c2 = p0.x ^ c3 ^ key1;
        uint next_c3 = p0.y;

        c0 = next_c0;
        c1 = next_c1;
        c2 = next_c2;
        c3 = next_c3;

        key0 += W0;
        key1 += W1;
    }

    return uint4(c0, c1, c2, c3);
}

inline uint rollout_philox_stream_uint32(
    uint generation_step,
    uint row,
    uint batch_size,
    uint seed_lo,
    uint seed_hi
) {
    ulong linear_index =
        ulong(generation_step) * ulong(batch_size)
        + ulong(row);

    ulong counter = linear_index / 4ul;
    uint lane = uint(linear_index & 3ul);

    uint4 block = rollout_philox4x32_10(
        counter,
        seed_lo,
        seed_hi
    );

    return block[lane];
}

inline float rollout_philox_uniform(
    uint raw
) {
    return float(raw & 0x7fffffffu)
        * 4.6566127342e-10f;
}

kernel void rollout_sample_from_topk(
    device const float* topk_values [[buffer(0)]],
    device const int* topk_indices [[buffer(1)]],
    device const uint* rng_state [[buffer(2)]], 
    device const uint* rng_raw [[buffer(3)]],
    device uchar* finished [[buffer(4)]], 
    device int* next_tokens [[buffer(5)]],
    device int* generated_ids [[buffer(6)]], 
    device int* lengths [[buffer(7)]],
    device int* position [[buffer(8)]], 
    device int* step [[buffer(9)]],
    device int* finished_all [[buffer(10)]],
    device const float* temperature [[buffer(11)]], 
    device const int* runtime_top_k [[buffer(12)]],
    device const float* logits [[buffer(13)]], 
    device const int* logits_position [[buffer(14)]],
    device float* sort_values [[buffer(15)]], 
    device int* sort_indices [[buffer(16)]],
    constant uint& B [[buffer(17)]], 
    constant uint& K_MAX [[buffer(18)]],
    constant uint& mode_id [[buffer(19)]], 
    constant uint& max_tokens [[buffer(20)]],
    constant int& eos_token [[buffer(21)]], 
    constant int& bos_token [[buffer(22)]],
    constant uint& V [[buffer(23)]], 
    constant uint& T [[buffer(24)]],
    constant uint& P [[buffer(25)]],
    uint row [[threadgroup_position_in_grid]], 
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= B) return;
    const uint source = mode_id == 0u ? 0u : row;
    const float temp = temperature[0];
    // T=0 is global argmax for every K; the TopK stage keeps at least one value.
    if (temp == 0.0f) {
        if (tid == 0u) next_tokens[row] = topk_indices[size_t(source) * K_MAX];
        return;
    }
    const int requested = runtime_top_k[0];
    const bool sorted = requested > 0;
    const uint K = sorted ? min(uint(requested), V) : V;
    const bool large_k = sorted && K > K_MAX;
    const uint pos = 0u;
    device const float* row_logits = logits + (size_t(source) * T + pos) * V;
    device float* workspace = sort_values + size_t(row) * P;
    device int* order = sort_indices + size_t(row) * P;

    // One group owns each row's planned scratch. No cross-group barriers.
    // Small K uses the existing TopK result; K<=0 uses token-ID order directly.
    if (large_k) {
        for (uint i=tid; i<P; i+=256u) {
            workspace[i] = i<V ? row_logits[i] : -INFINITY;
            order[i] = i<V ? int(i) : 0x7fffffff;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint size=2u; size<=P; size<<=1u) {
            for (uint stride=size>>1u; stride>0u; stride>>=1u) {
                for (uint i=tid; i<P; i+=256u) {
                    uint j=i^stride;
                    if (j>i) {
                        float a=workspace[i], b=workspace[j];
                        int ia=order[i], ib=order[j];
                        bool before=key_before(a,ia,b,ib);
                        if (before != ((i&size)==0u)) {
                            workspace[i]=b; workspace[j]=a;
                            order[i]=ib; order[j]=ia;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
    }
    device const float* values = !sorted ? row_logits :
        (large_k ? workspace : topk_values + size_t(source) * K_MAX);
    device const int* indices = large_k ? order : topk_indices + size_t(source) * K_MAX;
    threadgroup float reduce[256], prefix[256];
    threadgroup float maximum, threshold, carried;
    threadgroup atomic_uint winner;
    float local=-INFINITY;
    for (uint i=tid; i<K; i+=256u) local=max(local,values[i]/temp);
    reduce[tid]=local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n=128u; n>0u; n>>=1u) {
        if (tid<n) reduce[tid]=max(reduce[tid],reduce[tid+n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid==0u) maximum=reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local=0.0f;
    for (uint i=tid; i<K; i+=256u) local+=exp(values[i]/temp-maximum);
    reduce[tid]=local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n=128u; n>0u; n>>=1u) {
        if (tid<n) reduce[tid]+=reduce[tid+n];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid==0u) {
        uint raw=rollout_philox_stream_uint32(uint(step[mode_id == 0u ? 0u : row]),row,B,rng_state[0],rng_state[1]);
        threshold=rollout_philox_uniform(raw)*reduce[0];
        carried=0.0f;
        atomic_store_explicit(&winner,K,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base=0u; base<K; base+=256u) {
        uint i=base+tid;
        prefix[tid]=i<K ? exp(values[i]/temp-maximum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride=1u; stride<256u; stride<<=1u) {
            float add=tid>=stride ? prefix[tid-stride] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[tid]+=add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (i<K && carried+prefix[tid]>threshold)
            atomic_fetch_min_explicit(&winner,i,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&winner,memory_order_relaxed)<K) break;
        if (tid==0u) carried+=prefix[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid==0u) {
        uint chosen=min(atomic_load_explicit(&winner,memory_order_relaxed),K-1u);
        next_tokens[row]=sorted ? indices[chosen] : int(chosen);
    }
    // Sample every row. Existing Engine policy applies forced queues/masks and
    // commits completion/position/step once after the GPU pass has completed.
}

kernel void rollout_decode_state(
    device int* tokens [[buffer(0)]], 
    device uchar* masks [[buffer(1)]],
    device const int* metadata [[buffer(2)]], 
    device const uchar* finished [[buffer(3)]],
    device const int* config [[buffer(4)]], 
    device int* positions [[buffer(5)]],
    device int* steps [[buffer(6)]], 
    device const int* candidates [[buffer(7)]],
    device int* next_tokens [[buffer(8)]], 
    device const int* global_step [[buffer(9)]],
    device int* out [[buffer(10)]], 
    constant uint& B [[buffer(11)]],
    constant uint& S [[buffer(12)]], 
    constant uint& phase [[buffer(13)]],
    constant uint& offset [[buffer(14)]], 
    uint row [[thread_position_in_grid]]) 
{
    if (row >= B) return;
    int start = metadata[2 * row];
    if (finished[row] && config[2] == 1) start = global_step[0];
    int i = start + int(offset);
    // Bound every dummy access, including KV writes after the generation budget.
    int safe_i = clamp(i, 1, config[1] - 1);
    if (phase == 0) {
        positions[row] = config[0] + safe_i - 1;
        steps[row] = safe_i;
        out[row] = tokens[row * S + uint(safe_i - 1)];
    } else {
        int raw = candidates[row];
        next_tokens[row] = raw;
        if (i < config[1] && i < int(S)) {
            bool forced = i < metadata[2 * row + 1];
            int token = forced ? tokens[row * S + uint(i)] : raw;
            tokens[row * S + uint(i)] = token;
            masks[row * S + uint(i)] = forced ? uchar(0) : uchar(1);
            out[row] = token;
        } else {
            out[row] = raw;
        }
    }
}

kernel void rollout_decode_finished(
    device const uchar* finished [[buffer(0)]],
    device const int* step     [[buffer(1)]],
    device const int* config   [[buffer(2)]],
    device int* out            [[buffer(3)]],
    constant uint& B           [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    int done = 1;
    for (uint row = 0u; row < B; ++row) {
        if (finished[row] == 0) {
            done = 0;
            break;
        }
    }
    out[0] = done;
}

kernel void grad_norm_sq(
    device const uchar* g          [[buffer(0)]],
    device float* partial          [[buffer(1)]],
    constant uint& numel           [[buffer(2)]],
    constant uint& grad_bf16       [[buffer(3)]],
    uint tid   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]]
) {
    threadgroup float shared[256];

    float val = 0.0f;
    if (tid < numel) {
        float x;
        if (grad_bf16 != 0u) {
            const ushort bits = reinterpret_cast<device const ushort*>(g)[tid];
            x = as_type<float>(uint(bits) << 16u);
        } else {
            x = reinterpret_cast<device const float*>(g)[tid];
        }
        val = x * x;
    }
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial[tgid] = shared[0];
    }
}

kernel void adamw_step(
    device float* p               [[buffer(0)]],
    device const uchar* g         [[buffer(1)]],
    device float* m               [[buffer(2)]],
    device float* v               [[buffer(3)]],
    constant float& decay_factor  [[buffer(4)]],
    constant float& update_lr     [[buffer(5)]],
    constant float& b1            [[buffer(6)]],
    constant float& b2            [[buffer(7)]],
    constant float& eps           [[buffer(8)]],
    constant uint& numel          [[buffer(9)]],
    constant uint& decay_numel    [[buffer(10)]],
    constant float& clip_coef     [[buffer(11)]],
    constant float& inv_sqrt_bc2   [[buffer(12)]],
    constant uint& grad_bf16       [[buffer(13)]],
    uint id                       [[thread_position_in_grid]]
) {
    const uint base = id * 4u;
    const float wd = base < decay_numel ? decay_factor : 1.0f;
    const float one_minus_b1 = 1.0f - b1;
    const float one_minus_b2 = 1.0f - b2;

    if (base + 4u <= numel) {
        device packed_float4* p4 = (device packed_float4*)(p + base);
        device const packed_float4* g4 = (device const packed_float4*)(g + base);
        device packed_float4* m4 = (device packed_float4*)(m + base);
        device packed_float4* v4 = (device packed_float4*)(v + base);

        float4 pv = float4(*p4);
        float4 mv = float4(*m4);
        float4 vv = float4(*v4);

        float4 gv;
        if (grad_bf16 != 0u) {
            device const ushort* gb = reinterpret_cast<device const ushort*>(g);
            const uint4 bits = uint4(gb[base], gb[base + 1u], gb[base + 2u], gb[base + 3u]);
            gv = as_type<float4>(bits << 16u);
        } else {
            device const float* gf = reinterpret_cast<device const float*>(g);
            gv = float4(*reinterpret_cast<device const packed_float4*>(gf + base));
        }
        gv *= clip_coef;

        pv *= wd;
        volatile float4 dm = (gv - mv) * one_minus_b1;
        mv = mv + dm;

        const float4 gv_squared = gv * gv;
        volatile float4 dv = (gv_squared - vv) * one_minus_b2;
        vv = vv + dv;

        const float4 denominator = sqrt(max(vv, float4(0.0f))) * inv_sqrt_bc2 + float4(eps);
        volatile float4 dp = -(mv/denominator) * update_lr;
        pv = pv + dp;
        *p4 = packed_float4(pv);
        *m4 = packed_float4(mv);
        *v4 = packed_float4(vv);
        return;
    }

    for (uint i = base; i < numel; ++i) {
        float p_val = p[i];
        float m_val = m[i];
        float v_val = v[i];

        float g_val;
        if (grad_bf16 != 0u) {
            const ushort bits = reinterpret_cast<device const ushort*>(g)[i];
            g_val = as_type<float>(uint(bits) << 16u);
        } else {
            g_val = reinterpret_cast<device const float*>(g)[i];
        }
        g_val *= clip_coef;

        p_val *= wd;
        volatile float dm = (g_val - m_val) * one_minus_b1;
        m_val = m_val + dm;
        const float g_squared = g_val * g_val;
        volatile float dv = (g_squared - v_val) * one_minus_b2;
        v_val = v_val + dv;
        const float denominator = sqrt(max(v_val, 0.0f)) * inv_sqrt_bc2 + eps;
        volatile float dp = -(m_val / denominator) * update_lr;
        p_val = p_val + dp;

        p[i] = p_val;
        m[i] = m_val;
        v[i] = v_val;
    }
}

kernel void adamw8bit_step(
    device float* p [[buffer(0)]],
    device const uchar* g [[buffer(1)]],
    device uchar* qm [[buffer(2)]],
    device uchar* qv [[buffer(3)]],
    device float* absmax_m [[buffer(4)]],
    device float* absmax_v [[buffer(5)]],
    device const float* qmap_m [[buffer(6)]],
    device const float* qmap_v [[buffer(7)]],
    constant float& decay [[buffer(8)]],
    constant float& lr_bc1 [[buffer(9)]],
    constant float& b1 [[buffer(10)]],
    constant float& b2 [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    constant uint& numel [[buffer(13)]],
    constant uint& decay_numel [[buffer(14)]],
    constant uint& first_step [[buffer(15)]],
    constant float& inv_sqrt_bc2 [[buffer(16)]],
    constant uint& grad_bf16 [[buffer(17)]],
    constant float& clip_coef [[buffer(18)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    constexpr uint kBlockSize = 256u;

    threadgroup float local_max_m[kBlockSize];
    threadgroup float local_max_v[kBlockSize];

    const uint i = block * kBlockSize + lane;
    const bool valid = i < numel;

    float m = 0.0f;
    float v = 0.0f;

    if (valid && !first_step) {
        m = absmax_m[block] * qmap_m[uint(qm[i])];
        v = absmax_v[block] * qmap_v[uint(qv[i])];
    }

    if (valid) {
        float p_value = p[i];
        float g_value;
        if (grad_bf16 != 0u) {
            const ushort bits = reinterpret_cast<device const ushort*>(g)[i];
            g_value = as_type<float>(uint(bits) << 16u);
        } else {
            g_value = reinterpret_cast<device const float*>(g)[i];
        }

        g_value *= clip_coef;

        p_value *= (i < decay_numel) ? decay : 1.0f;
        volatile float dm = (g_value - m) * (1.0f - b1);
        m = m + dm;

        const float g_squared = g_value * g_value;
        volatile float dv = (g_squared - v) * (1.0f - b2);
        v = v + dv;

        const float denominator = sqrt(fmax(v, 0.0f)) * inv_sqrt_bc2 + eps;
        volatile float dp = -(m / denominator) * lr_bc1;
        p[i] = p_value + dp;
    }

    local_max_m[lane] = valid ? fabs(m) : 0.0f;
    local_max_v[lane] = valid ? fmax(v, 0.0f) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            local_max_m[lane] = fmax(
                local_max_m[lane],
                local_max_m[lane + stride]
            );
            local_max_v[lane] = fmax(
                local_max_v[lane],
                local_max_v[lane + stride]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float new_scale_m = fmax(local_max_m[0], 1.0e-30f);
    const float new_scale_v = fmax(local_max_v[0], 1.0e-30f);

    if (lane == 0u) {
        absmax_m[block] = new_scale_m;
        absmax_v[block] = new_scale_v;
    }

    if (!valid) {
        return;
    }

    const float m_normalized = clamp(m / new_scale_m, -1.0f, 1.0f);

    const float v_normalized = clamp(fmax(v, 0.0f) / new_scale_v, 0.0f, 1.0f);

    uint lo_m = 0u;
    uint hi_m = 255u;

    for (uint iter = 0u; iter < 8u; ++iter) {
        const uint mid = (lo_m + hi_m) >> 1u;
        if (qmap_m[mid] < m_normalized) {
            lo_m = mid + 1u;
        } else {
            hi_m = mid;
        }
    }

    const uint m_upper = min(lo_m, 255u);
    const uint m_lower = m_upper > 0u ? m_upper - 1u : 0u;

    qm[i] = uchar(fabs(qmap_m[m_upper] - m_normalized) < fabs(qmap_m[m_lower] - m_normalized) ? m_upper : m_lower);

    uint lo_v = 0u;
    uint hi_v = 255u;

    for (uint iter = 0u; iter < 8u; ++iter) {
        const uint mid = (lo_v + hi_v) >> 1u;
        if (qmap_v[mid] < v_normalized) {
            lo_v = mid + 1u;
        } else {
            hi_v = mid;
        }
    }

    const uint v_upper = min(lo_v, 255u);
    const uint v_lower = v_upper > 0u ? v_upper - 1u : 0u;

    qv[i] = uchar(fabs(qmap_v[v_upper] - v_normalized) < fabs(qmap_v[v_lower] - v_normalized) ? v_upper : v_lower);
}

kernel void mini_update_v(
    device const uchar* g     [[buffer(0)]],
    device float* v           [[buffer(1)]],
    constant float& b2        [[buffer(2)]],
    constant float& grad_scale [[buffer(3)]], 
    constant uint& numel      [[buffer(4)]],
    constant uint& block_elems [[buffer(5)]],
    constant uint& cols       [[buffer(6)]],
    constant uint& head_cols  [[buffer(7)]],
    constant uint& first_step [[buffer(8)]],
    constant uint& grad_bf16  [[buffer(9)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane  [[thread_position_in_threadgroup]]
) {
    constexpr uint kThreads = 256u;
    threadgroup float partial[kThreads];
    const bool by_head = head_cols != 0u;
    const uint group_count = by_head ? cols / head_cols : 1u + (numel - 1u) / block_elems;

    if (block >= group_count) {
        return;
    }

    uint start = 0u;
    uint count = block_elems;

    if (!by_head) {
        start = block * block_elems;
        count = min(block_elems, numel - start);
    }

    float sum_sq = 0.0f;
    if (by_head) {
        const uint col_begin = block * head_cols;
        for (uint j = lane; j < count; j += kThreads) {
            const uint row = j / head_cols;
            const uint col_in_head = j % head_cols;
            const uint i = row * cols + col_begin + col_in_head;
            if (grad_bf16 != 0u) {
                const ushort bits = reinterpret_cast<device const ushort*>(g)[i];
                const float value = as_type<float>(uint(bits) << 16) * grad_scale;
                sum_sq = fma(value, value, sum_sq);
            } else {
                const float value = reinterpret_cast<device const float*>(g)[i] * grad_scale;
                sum_sq = fma(value, value, sum_sq);
            }
        }
    } else {
        for (uint j = lane; j < count; j += kThreads) {
            if (grad_bf16) {
                const ushort bits = reinterpret_cast<device const ushort*>(g)[start+j];
                const float value = as_type<float>(uint(bits) << 16) * grad_scale;
                sum_sq = fma(value, value, sum_sq);
            } else {
                const float value = reinterpret_cast<device const float*>(g)[start+j] * grad_scale;
                sum_sq = fma(value, value, sum_sq);
            }
        }
    }

    partial[lane] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const float mean_sq = partial[0] / float(count);
        const float old_v = first_step ? 0.0f : v[block];
        // v_t = beta2 * v_(t-1) + (1-beta2) * mean(g^2)
        v[block] = fma(mean_sq - old_v, 1.0f - b2, old_v);
    }
}

inline float pcg_hash_unit_float(uint value) {
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    const uint result = (word >> 22u) ^ word;
    // Every integer in [0, 2^24 - 1] is exactly representable in FP32.
    // Result is strictly less than 1.
    return float(result >> 8u) * (1.0f / 16777216.0f);
}

kernel void adamw_mini8bit_step(
    device float* p [[buffer(0)]],
    device const uchar* g [[buffer(1)]],
    device uchar* m_state [[buffer(2)]],
    device float* meta [[buffer(3)]],
    device const float* v [[buffer(4)]],
    constant float& decay [[buffer(5)]],
    constant float& lr_bc1 [[buffer(6)]],
    constant float& inv_sqrt_bc2 [[buffer(7)]],
    constant float& b1 [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant float& grad_scale [[buffer(10)]], 
    constant uint& n [[buffer(11)]],
    constant uint& decay_n [[buffer(12)]],
    constant uint& block_elems [[buffer(13)]],
    constant uint& cols [[buffer(14)]],
    constant uint& head_cols [[buffer(15)]],
    constant uint& first [[buffer(16)]],
    constant uint& seed [[buffer(17)]],
    constant uint& stochastic [[buffer(18)]],
    constant uint& transposed [[buffer(19)]],
    constant uint& quantized [[buffer(20)]],
    constant uint& grad_bf16 [[buffer(21)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    const uint j = block * 256u + lane;
    const bool valid = j < n;
    const uint rows = n / cols;
    // j: logical [out, in]; i: physical [in, out].
    const uint i = transposed ? (j % rows) * cols + j / rows : j;
    threadgroup float low[256], high[256];
    threadgroup float old_scale, old_zero;

    if (lane == 0u) {
        old_scale = (!first && quantized) ? meta[2u * block] : 0.0f;
        old_zero = (!first && quantized) ? meta[2u * block + 1u] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = 0.0f;
    if (valid) {
        if (!first) {
            m = quantized ? (float(m_state[j]) - old_zero) * old_scale : reinterpret_cast<device float*>(m_state)[j];
        }
        if (grad_bf16) {
            const ushort bits = reinterpret_cast<device const ushort*>(g)[i];
            const float grad = as_type<float>(uint(bits) << 16);
            m = fma(grad * grad_scale - m, 1.0f - b1, m);
        } else {
            const float grad = reinterpret_cast<device const float*>(g)[i];
            m = fma(grad * grad_scale - m, 1.0f - b1, m);
        }
        const uint vg = head_cols ? (i % cols) / head_cols : i / block_elems;
        const float denom = sqrt(v[vg]) * inv_sqrt_bc2 + eps;
        const float decayed = i < decay_n ? p[i] * decay : p[i];
        // Update with FP32 momentum before quantization.
        p[i] = decayed - lr_bc1 * m / denom;
    }

    // quantized is uniform across the threadgroup.
    if (!quantized) {
        if (valid) {
            reinterpret_cast<device float*>(m_state)[j] = m;
        }
        return;
    }

    low[lane] = valid ? m : INFINITY;
    high[lane] = valid ? m : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride; stride >>= 1u) {
        if (lane < stride) {
            low[lane] = fmin(low[lane], low[lane + stride]);
            high[lane] = fmax(high[lane], high[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Same affine formula as the reference implementation.
    const float scale =
        fmax(high[0] - low[0], 1.0e-5f) / 255.0f;
    const float zero =
        clamp(-rint(low[0] / scale), 0.0f, 255.0f);

    if (lane == 0u) {
        meta[2u * block] = scale;
        meta[2u * block + 1u] = zero;
    }

    if (valid) {
        const float x = m / scale;
        float rounded = rint(x);

        if (stochastic) {
            const float down = floor(x);
            const float u = pcg_hash_unit_float(
                seed ^ (j * 0x9E3779B9u) ^ (j >> 15u)
            );
            rounded = down + float(u < x - down);
        }

        m_state[j] =
            uchar(clamp(rounded + zero, 0.0f, 255.0f));
    }
}

inline float2 gqa_load_weight_pair(device const float* w, uint row,
                                   uint col, uint rows, uint cols) {
    if (row >= rows || col >= cols) return float2(0.0f);
    if (col + 1u < cols)
        return float2(*reinterpret_cast<device const packed_float2*>(w + row*cols + col));
    return float2(w[row*cols + col], 0.0f);
}
inline void gqa_store_pair(device float* out, float2 value,
                          uint b, uint t, uint T, uint heads, uint D, uint col) {
    for (uint j=0u; j<2u; ++j) {
        const uint c=col+j;
        if (c < heads*D) out[((b*heads+c/D)*T+t)*D+c%D]=value[j];
    }
}


template <bool kChecked> inline void gqa_mma_sweep(
    threadgroup const float* Z,
    device const float* wq, device const float* wk, device const float* wv,
    uint C, uint CKV, uint input_base,
    uint matrix_x, uint matrix_y,
    uint weight_col0, uint weight_col1,
    thread float2& q00, thread float2& q01, thread float2& q10, thread float2& q11,
    thread float2& k00, thread float2& k01, thread float2& k10, thread float2& k11,
    thread float2& v00, thread float2& v01, thread float2& v10, thread float2& v11)
{
    constexpr uint kFq2BM      = 16u;
    constexpr uint kFq2BK      = 128u;              // 可调: 128 / 192 / 256
    constexpr uint kFq2BN      = 128u;              // 输出 supertile, 不变
    constexpr uint kFq2Frag    = 8u;
    constexpr uint kFq2ZStride = kFq2BK + 4u;       // 132: 行长 528B, 16B 对齐
    constexpr uint kFq2Fill    = kFq2BK / 64u;      // 每线程每次 fill 的 float4 数 = 2

    uint z_off0 = matrix_y * kFq2ZStride + matrix_x;
    uint z_off1 = z_off0 + 8u * kFq2ZStride;
    uint w_row  = input_base + matrix_y;
    uint w_off  = w_row * C + weight_col0;
    const uint w_step = kFq2Frag * C;

    for (uint ks = 0u; ks < kFq2BK; ks += kFq2Frag) {
        const float2 z0 = float2(*reinterpret_cast<threadgroup const packed_float2*>(Z + z_off0));
        const float2 z1 = float2(*reinterpret_cast<threadgroup const packed_float2*>(Z + z_off1));

        float2 wqf0, wqf1;
        if (kChecked) {
            wqf0 = gqa_load_weight_pair(wq, w_row, weight_col0, C, C);
            wqf1 = gqa_load_weight_pair(wq, w_row, weight_col1, C, C);
        } else {
            wqf0 = float2(*reinterpret_cast<device const packed_float2*>(wq+w_off));
            wqf1 = float2(*reinterpret_cast<device const packed_float2*>(wq+w_off+8u));
        }

        q00 = lowreg_matrix::mma_8x8(z0, wqf0, q00);
        q10 = lowreg_matrix::mma_8x8(z1, wqf0, q10);
        q01 = lowreg_matrix::mma_8x8(z0, wqf1, q01);
        q11 = lowreg_matrix::mma_8x8(z1, wqf1, q11);
        // Uniform within each SIMD group; Q-only tiles skip K/V MMAs.
        if (weight_col0 - matrix_x < CKV) {
            const float2 wkf0 = gqa_load_weight_pair(wk, w_row, weight_col0, C, CKV);
            const float2 wkf1 = gqa_load_weight_pair(wk, w_row, weight_col1, C, CKV);
            const float2 wvf0 = gqa_load_weight_pair(wv, w_row, weight_col0, C, CKV);
            const float2 wvf1 = gqa_load_weight_pair(wv, w_row, weight_col1, C, CKV);
        k00 = lowreg_matrix::mma_8x8(z0, wkf0, k00);
        k10 = lowreg_matrix::mma_8x8(z1, wkf0, k10);
        k01 = lowreg_matrix::mma_8x8(z0, wkf1, k01);
        k11 = lowreg_matrix::mma_8x8(z1, wkf1, k11);
        v00 = lowreg_matrix::mma_8x8(z0, wvf0, v00);
        v10 = lowreg_matrix::mma_8x8(z1, wvf0, v10);
        v01 = lowreg_matrix::mma_8x8(z0, wvf1, v01);
        v11 = lowreg_matrix::mma_8x8(z1, wvf1, v11);
        }

        z_off0 += kFq2Frag;
        z_off1 += kFq2Frag;
        w_row  += kFq2Frag;
        w_off  += w_step;
    }
}


kernel void gqa_rms_qkv(
    device const float* x        [[buffer(0)]],
    device const float* wq       [[buffer(1)]],
    device const float* bq       [[buffer(2)]],
    device const float* wk       [[buffer(3)]],
    device const float* bk       [[buffer(4)]],
    device const float* wv       [[buffer(5)]],
    device const float* bv       [[buffer(6)]],
    device float* q_cache        [[buffer(7)]],
    device float* k_cache        [[buffer(8)]],
    device float* v_cache        [[buffer(9)]],
    device const float* gamma [[buffer(10)]],
    constant uint& B             [[buffer(11)]],
    constant uint& T             [[buffer(12)]],
    constant uint& C             [[buffer(13)]],
    constant uint& H             [[buffer(14)]],
    constant uint& eps_bits      [[buffer(15)]],
    constant uint& HKV [[buffer(16)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 thread_pos             [[thread_position_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  simd_id                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kFq2BM      = 16u;
    constexpr uint kFq2BK      = 128u;              // 可调: 128 / 192 / 256
    constexpr uint kFq2BN      = 128u;              // 输出 supertile, 不变
    constexpr uint kFq2Frag    = 8u;
    constexpr uint kFq2ZStride = kFq2BK + 4u;       // 132: 行长 528B, 16B 对齐
    constexpr uint kFq2Fill    = kFq2BK / 64u;      // 每线程每次 fill 的 float4 数 = 2

    threadgroup float4 Z4[kFq2BM][kFq2ZStride / 4u];   // 8.25KB (kBK=128)
    threadgroup float  inv_rms_tg[kFq2BM];
    threadgroup float* Z = reinterpret_cast<threadgroup float*>(&Z4[0][0]);

    const uint tid = thread_pos.x;
    if (tg_pos.x != 0u || tg_pos.z != 0u ||
        B == 0u || T == 0u || C == 0u || H == 0u || HKV == 0u || H % HKV != 0u || (C & 3u) != 0u || C % H != 0u) {
        return;
    }
    const uint D = C / H;
    const uint CKV = HKV * D;
    if (D == 0u || D > 128u) return;

    const uint token_count = B * T;
    const uint row_base = tg_pos.y * kFq2BM;
    if (row_base >= token_count) return;

    const float eps = as_type<float>(eps_bits);

    // ---- fill 坐标: 16 线程一行, f_row == simd_id*2 + (lane>>4) ----
    const uint f_row  = tid >> 4;
    const uint f_c16  = tid & 15u;
    const uint f_grow = row_base + f_row;
    const bool f_valid = f_grow < token_count;
    device const float* x_row = x + f_grow * C;

    // ---- 输出坐标(与原版一致) ----
    const ushort2 mo = morton_order(ushort(lane));
    const uint matrix_x = uint(mo.x);
    const uint matrix_y = uint(mo.y);
    const uint simd_col_base = simd_id * 16u;
    const uint out_grow0 = row_base + matrix_y;
    const uint out_grow1 = out_grow0 + 8u;
    const bool out_row0_valid = out_grow0 < token_count;
    const bool out_row1_valid = out_grow1 < token_count;
    uint ob0 = 0u, ot0 = 0u, ob1 = 0u, ot1 = 0u;
    if (out_row0_valid) { ob0 = out_grow0 / T; ot0 = out_grow0 - ob0 * T; }
    if (out_row1_valid) { ob1 = out_grow1 / T; ot1 = out_grow1 - ob1 * T; }

    // ---- 首次填充: 原始 x(无乘法), 顺手累加 x^2 ----
    float sumsq = 0.0f;
    for (uint i = 0u; i < kFq2Fill; ++i) {
        const uint col = (f_c16 + i * 16u) * 4u;
        const float4 v = f_valid ? fq2_load_x4(x_row, col, C) : float4(0.0f);
        Z4[f_row][f_c16 + i * 16u] = v * fq2_load_x4(gamma, col, C);
        fq2_acc_sq(v, sumsq);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool sumsq_pending = true;
    const bool z_resident = (C <= kFq2BK);   // 小 C 时 Z 常驻, 后续 output tile 零 refill

    for (uint output_base = 0u; output_base < C; output_base += kFq2BN) {
        float2 q00 = float2(0.0f), q01 = float2(0.0f), q10 = float2(0.0f), q11 = float2(0.0f);
        float2 k00 = float2(0.0f), k01 = float2(0.0f), k10 = float2(0.0f), k11 = float2(0.0f);
        float2 v00 = float2(0.0f), v01 = float2(0.0f), v10 = float2(0.0f), v11 = float2(0.0f);

        const uint weight_col0 = output_base + simd_col_base + matrix_x;
        const uint weight_col1 = weight_col0 + 8u;
        const bool full_output_tile = output_base + simd_col_base + 16u <= C;

        for (uint input_base = 0u; input_base < C; input_base += kFq2BK) {
            // 1) 先发射下一块的 device load 到寄存器, 与 MMA 重叠
            const bool has_next_input = input_base + kFq2BK < C;
            const bool prep_next_out  = (!z_resident) && (!has_next_input) &&
                                        (output_base + kFq2BN < C);
            const bool do_refill = has_next_input || prep_next_out;
            const uint next_base = has_next_input ? (input_base + kFq2BK) : 0u;

            float4 pre[kFq2Fill];
            if (do_refill) {
                for (uint i = 0u; i < kFq2Fill; ++i) {
                    const uint col = next_base + (f_c16 + i * 16u) * 4u;
                    pre[i] = f_valid ? fq2_load_x4(x_row, col, C) : float4(0.0f);
                }
            }

            // 2) MMA: 分支已提出内层循环
            if (full_output_tile && (input_base + kFq2BK <= C)) {
                gqa_mma_sweep<false>(Z, wq, wk, wv, C, CKV, input_base,
                    matrix_x, matrix_y, weight_col0, weight_col1,
                    q00, q01, q10, q11, k00, k01, k10, k11, v00, v01, v10, v11);
            } else {
                gqa_mma_sweep<true>(Z, wq, wk, wv, C, CKV, input_base,
                    matrix_x, matrix_y, weight_col0, weight_col1,
                    q00, q01, q10, q11, k00, k01, k10, k11, v00, v01, v10, v11);
            }

            // 3) 写回 Z; 第一遍扫描期间继续累加 x^2 (prep_next_out 重装 block0 不重复累加)
            if (do_refill) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0u; i < kFq2Fill; ++i) {
                    const uint col = next_base + (f_c16 + i * 16u) * 4u;
                    Z4[f_row][f_c16 + i * 16u] = pre[i] * fq2_load_x4(gamma, col, C);
                }
                if (sumsq_pending && has_next_input) {
                    for (uint i = 0u; i < kFq2Fill; ++i) fq2_acc_sq(pre[i], sumsq);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // ---- 第一遍 K 扫描结束: sumsq 完整, 归约一次, 全程复用 ----
        if (sumsq_pending) {
            const float s = fq2_reduce16(sumsq);
            if ((lane & 15u) == 0u) {
                inv_rms_tg[f_row] = f_valid ? rsqrt(s / float(C) + eps) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sumsq_pending = false;
        }

        // ---- 出口: s·acc, 再由 store helper 加 bias(与原版一致) ----
        const float s0 = inv_rms_tg[matrix_y];
        const float s1 = inv_rms_tg[8u + matrix_y];

        const uint output_channel0 = output_base + simd_col_base + matrix_x;
        const uint output_channel1 = output_channel0 + 8u;
        const float2 bq0 = fa_qkv_load_bias_pair_bm16(bq, output_channel0, C);
        const float2 bq1 = fa_qkv_load_bias_pair_bm16(bq, output_channel1, C);
        const float2 bk0 = fa_qkv_load_bias_pair_bm16(bk, output_channel0, CKV);
        const float2 bk1 = fa_qkv_load_bias_pair_bm16(bk, output_channel1, CKV);
        const float2 bv0 = fa_qkv_load_bias_pair_bm16(bv, output_channel0, CKV);
        const float2 bv1 = fa_qkv_load_bias_pair_bm16(bv, output_channel1, CKV);
        if (out_row0_valid) {
            gqa_store_pair(q_cache,q00*s0+bq0,ob0,ot0,T,H,D,output_channel0);
            gqa_store_pair(q_cache,q01*s0+bq1,ob0,ot0,T,H,D,output_channel1);
            gqa_store_pair(k_cache,k00*s0+bk0,ob0,ot0,T,HKV,D,output_channel0);
            gqa_store_pair(k_cache,k01*s0+bk1,ob0,ot0,T,HKV,D,output_channel1);
            gqa_store_pair(v_cache,v00*s0+bv0,ob0,ot0,T,HKV,D,output_channel0);
            gqa_store_pair(v_cache,v01*s0+bv1,ob0,ot0,T,HKV,D,output_channel1);
        }
        if (out_row1_valid) {
            gqa_store_pair(q_cache,q10*s1+bq0,ob1,ot1,T,H,D,output_channel0);
            gqa_store_pair(q_cache,q11*s1+bq1,ob1,ot1,T,H,D,output_channel1);
            gqa_store_pair(k_cache,k10*s1+bk0,ob1,ot1,T,HKV,D,output_channel0);
            gqa_store_pair(k_cache,k11*s1+bk1,ob1,ot1,T,HKV,D,output_channel1);
            gqa_store_pair(v_cache,v10*s1+bv0,ob1,ot1,T,HKV,D,output_channel0);
            gqa_store_pair(v_cache,v11*s1+bv1,ob1,ot1,T,HKV,D,output_channel1);
        }
    }
}


kernel void gqa_from_qkv_d64(
    device const float* q_cache     [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device float* lse_out           [[buffer(4)]],
    constant uint& B                [[buffer(5)]],
    constant uint& H                [[buffer(6)]],
    constant uint& T                [[buffer(7)]],
    constant uint& C                [[buffer(8)]],
    constant uint& is_causal_int    [[buffer(9)]],
    constant uint& write_lse        [[buffer(10)]],
    constant uint& HKV [[buffer(11)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]])
{
    constexpr uint kBQ         = 8u;
    constexpr uint kBK         = 32u;
    constexpr uint kD          = 64u;          // tile 宽度，编译期固定
    constexpr uint kFrag       = 8u;
    constexpr uint kThreads    = 128u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kQStride    = kD + 1u;      // 65
    constexpr uint kKTStride   = kFrag + 1u;   // 9
    constexpr uint kPStride    = kBK + 1u;     // 33
    constexpr uint kKTGroup    = kD * kKTStride;   // 576
    constexpr uint kVSub       = kBK / kFrag;      // 4

    threadgroup float Q_tg[kBQ][kQStride];                  // 520 f
    threadgroup float KT_all[kSimdGroups * kKTGroup];       // 2304 f
    threadgroup float P_tg[kBQ][kPStride];                  // 264 f
    threadgroup float running_max_tg[kBQ];
    threadgroup float running_sum_tg[kBQ];
    threadgroup float alpha_tg[kBQ];

    const uint tid = thread_pos.x;

    // 统一退出条件，避免 barrier 死锁
    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u ||
        simd_lane_count != 32u || tg_pos.x != 0u ||
        B == 0u || H == 0u || HKV == 0u || H % HKV != 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;                       // 真实 head_dim
    if (D == 0u || D > kD) return;

    const uint bh = tg_pos.z;
    if (bh >= B * H) return;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint bkv = b * HKV + h / (H / HKV);
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) return;

    const bool  is_causal   = (is_causal_int != 0u);
    const bool  d_full      = (D == kD);        // 整个 threadgroup 一致的 uniform 分支
    const float score_scale = rsqrt(float(D));  // 必须用真实 D

    // D=64 只需要两组 accumulator：d = simd_id*8+x 与 (simd_id+4)*8+x
    float2 acc0 = float2(0.0f);
    float2 acc1 = float2(0.0f);

    if (tid < kBQ) {
        running_max_tg[tid] = -INFINITY;
        running_sum_tg[tid] = 0.0f;
        alpha_tg[tid]       = 0.0f;
    }

    // ---------------- Q：整个 kernel 只加载一次 ----------------
    // 8 行 × 16 个 float4 = 128 = kThreads，每线程恰好一个 float4
    {
        const uint local_q = tid >> 4;            // 0..7
        const uint d0      = (tid & 15u) << 2;    // 0,4,...,60
        const uint query   = query_base + local_q;
        float4 value = float4(0.0f);
        if (query < T) {
            device const float* row = q_cache + (bh * T + query) * D;
            if (d_full) {
                // fast path：无任何 D 边界判断
                value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
            } else if (d0 + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
            } else {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    if (d0 + c < D) value[c] = row[d0 + c];
                }
            }
        }
        Q_tg[local_q][d0 + 0u] = value.x;
        Q_tg[local_q][d0 + 1u] = value.y;
        Q_tg[local_q][d0 + 2u] = value.z;
        Q_tg[local_q][d0 + 3u] = value.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* KT_tg = KT_all + simd_id * kKTGroup;   // 每个 simdgroup 私有
    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;

    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {

        // 完整 tile / 尾 tile 判定（threadgroup 内一致）
        const bool tile_full     = (key_base + kBK) <= T;
        const bool tile_nocausal = (!is_causal) || (key_base + kBK <= query_base + 1u);
        const bool tile_fast     = tile_full && tile_nocausal;

        // ---------------- K 加载 + 转置：KT_tg[d][key] ----------------
        // 8 key × 16 float4 = 128 元素，32 lane × 4 次
        if (d_full && tile_full) {
            // fast path：既无 D 判断，也无 key<T 判断
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                const uint index     = lane + i * 32u;         // 0..127
                const uint local_key = index >> 4;             // 0..7
                const uint d0        = (index & 15u) << 2;     // 0..60
                const uint key       = key_base + simd_id * kFrag + local_key;
                const float4 value = float4(*reinterpret_cast<device const packed_float4*>(
                        k_cache + (bkv * T + key) * kD + d0));
                KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                const uint index     = lane + i * 32u;
                const uint local_key = index >> 4;
                const uint d0        = (index & 15u) << 2;
                const uint key       = key_base + simd_id * kFrag + local_key;
                float4 value = float4(0.0f);
                if (key < T) {
                    device const float* row = k_cache + (bkv * T + key) * D;
                    if (d0 + 3u < D) {
                        value = float4(*reinterpret_cast<device const packed_float4*>(row + d0));
                    } else if (d0 < D) {
                        #pragma unroll
                        for (uint c = 0u; c < 4u; ++c) {
                            if (d0 + c < D) value[c] = row[d0 + c];
                        }
                    }
                }
                KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // KT_tg 为 simdgroup 私有

        // ---------------- QK^T ----------------
        float2 score = float2(0.0f);
        #pragma unroll
        for (uint d_sub = 0u; d_sub < kD; d_sub += kFrag) {
            const float2 q_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(&Q_tg[0][0]),
                    kQStride, ushort2(ushort(0u), ushort(d_sub)), ushort(lane));
            const float2 k_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(KT_tg),
                    kKTStride, ushort2(ushort(d_sub), ushort(0u)), ushort(lane));
            score = lowreg_matrix::mma_8x8(q_frag, k_frag, score);
        }

        // ---------------- 提前发射 V 的 device load，掩盖延迟 ----------------
        const ushort2 v_off  = morton_order(ushort(lane));
        const uint    v_row  = uint(v_off.y);
        const uint    v_col0 = simd_id * kFrag + uint(v_off.x);
        const uint    v_col1 = (simd_id + kSimdGroups) * kFrag + uint(v_off.x);
        float2 v_frag0[kVSub];
        float2 v_frag1[kVSub];
        if (d_full && tile_full) {
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                const uint key = key_base + v_row + sub * kFrag;
                device const float* row = v_cache + (bkv * T + key) * kD;
                v_frag0[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                v_frag1[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
            }
        } else {
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                float2 f0 = float2(0.0f);
                float2 f1 = float2(0.0f);
                const uint key = key_base + v_row + sub * kFrag;
                if (key < T) {
                    device const float* row = v_cache + (bkv * T + key) * D;
                    if (v_col0 + 1u < D) {
                        f0 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                    } else if (v_col0 < D) {
                        f0[0] = row[v_col0];
                    }
                    if (v_col1 + 1u < D) {
                        f1 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                    } else if (v_col1 < D) {
                        f1[0] = row[v_col1];
                    }
                }
                v_frag0[sub] = f0;
                v_frag1[sub] = f1;
            }
        }

        // ---------------- scale + mask 写入 P ----------------
        const ushort2 s_off = morton_order(ushort(lane));
        const uint local_q  = uint(s_off.y);
        const uint local_k0 = simd_id * kFrag + uint(s_off.x);
        if (tile_fast) {
            // fast path：无 causal 判断、无 key<T 判断
            // （query >= T 的行 Q 已填 0，score=0，不影响任何被写出的行）
            P_tg[local_q][local_k0]      = score[0] * score_scale;
            P_tg[local_q][local_k0 + 1u] = score[1] * score_scale;
        } else {
            const uint query = query_base + local_q;
            const uint key0  = key_base + local_k0;
            const uint key1  = key0 + 1u;
            const bool ok0 = (query < T) && (key0 < T) && (!is_causal || key0 <= query);
            const bool ok1 = (query < T) && (key1 < T) && (!is_causal || key1 <= query);
            P_tg[local_q][local_k0]      = ok0 ? score[0] * score_scale : -INFINITY;
            P_tg[local_q][local_k0 + 1u] = ok1 ? score[1] * score_scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 32-key 在线 softmax ----------------
        const uint row_half     = lane >> 4u;
        const uint half_lane    = lane & 15u;
        const uint softmax_row  = simd_id * 2u + row_half;
        const uint p_col0       = half_lane;
        const uint p_col1       = half_lane + 16u;
        const float score0 = P_tg[softmax_row][p_col0];
        const float score1 = P_tg[softmax_row][p_col1];

        float local_max = max(score0, score1);
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(8u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(4u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(2u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(1u)));
        const ushort leader_lane = ushort(row_half * 16u);
        const float tile_max = simd_shuffle(local_max, leader_lane);

        const float old_max   = running_max_tg[softmax_row];
        const float new_max   = max(old_max, tile_max);
        const float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        const float prob0     = isfinite(score0) ? exp(score0 - new_max) : 0.0f;
        const float prob1     = isfinite(score1) ? exp(score1 - new_max) : 0.0f;

        float sum = prob0 + prob1;
        sum += simd_shuffle_down(sum, ushort(8u));
        sum += simd_shuffle_down(sum, ushort(4u));
        sum += simd_shuffle_down(sum, ushort(2u));
        sum += simd_shuffle_down(sum, ushort(1u));
        const float tile_sum = simd_shuffle(sum, leader_lane);

        P_tg[softmax_row][p_col0] = prob0;
        P_tg[softmax_row][p_col1] = prob1;
        if (half_lane == 0u) {
            running_max_tg[softmax_row] = new_max;
            running_sum_tg[softmax_row] = running_sum_tg[softmax_row] * old_scale + tile_sum;
            alpha_tg[softmax_row]       = old_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 旧 numerator 缩放 ----------------
        const float rescale = alpha_tg[uint(morton_order(ushort(lane)).y)];
        acc0 *= rescale;
        acc1 *= rescale;

        // ---------------- P × V ----------------
        #pragma unroll
        for (uint sub = 0u; sub < kVSub; ++sub) {
            const float2 p_frag = lowreg_matrix::load_tg_8x8(
                    reinterpret_cast<threadgroup const float*>(&P_tg[0][0]),
                    kPStride, ushort2(ushort(0u), ushort(sub * kFrag)), ushort(lane));
            acc0 = lowreg_matrix::mma_8x8(p_frag, v_frag0[sub], acc0);
            acc1 = lowreg_matrix::mma_8x8(p_frag, v_frag1[sub], acc1);
        }

        // 下一个 key tile 会覆写 P_tg，所有 simdgroup 必须先读完
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------------- 归一化并写出 [B,T,C] ----------------
    const ushort2 f_off = morton_order(ushort(lane));
    const uint f_lq  = uint(f_off.y);
    const uint f_q   = query_base + f_lq;
    if (f_q < T) {
        const float denom = running_sum_tg[f_lq];
        const float inv   = denom > 0.0f ? 1.0f / denom : 0.0f;
        const uint  x     = uint(f_off.x);

        if (write_lse != 0u && simd_id == 0u && x == 0u) {
            lse_out[bh * T + f_q] = denom > 0.0f ? running_max_tg[f_lq] + log(denom) : 0.0f;
        }

        device float* orow = out + (b * T + f_q) * C + h * D;
        const uint d0 = simd_id * kFrag + x;
        const uint d1 = (simd_id + kSimdGroups) * kFrag + x;
        if (d0        < D) orow[d0]        = acc0[0] * inv;
        if (d0 + 1u   < D) orow[d0 + 1u]   = acc0[1] * inv;
        if (d1        < D) orow[d1]        = acc1[0] * inv;
        if (d1 + 1u   < D) orow[d1 + 1u]   = acc1[1] * inv;
    }
}


kernel void gqa_from_qkv_d128(
    device const float* q_cache     [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* out               [[buffer(3)]],
    device float* lse_out           [[buffer(4)]],
    constant uint& B                [[buffer(5)]],
    constant uint& H                [[buffer(6)]],
    constant uint& T                [[buffer(7)]],
    constant uint& C                [[buffer(8)]],
    constant uint& is_causal_int    [[buffer(9)]],
    constant uint& write_lse        [[buffer(10)]],
    constant uint& HKV [[buffer(11)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint3 thread_pos                [[thread_position_in_threadgroup]],
    uint3 threads_per_tg            [[threads_per_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane_count            [[threads_per_simdgroup]])
{
    constexpr uint kBQ         = 8u;
    constexpr uint kBK         = 32u;
    constexpr uint kBD         = 64u;          // 单个 d 子块
    constexpr uint kD          = 128u;         // tile 总宽度
    constexpr uint kChunks     = kD / kBD;     // 2
    constexpr uint kFrag       = 8u;
    constexpr uint kThreads    = 128u;
    constexpr uint kSimdGroups = 4u;
    constexpr uint kQStride    = kBD + 1u;     // 65
    constexpr uint kKTStride   = kFrag + 1u;   // 9
    constexpr uint kPStride    = kBK + 1u;     // 33
    constexpr uint kKTGroup    = kBD * kKTStride;
    constexpr uint kVSub       = kBK / kFrag;  // 4

    threadgroup float Q_tg[kChunks][kBQ][kQStride];          // 1040 f
    threadgroup float KT_all[kSimdGroups * kKTGroup];        // 2304 f
    threadgroup float P_tg[kBQ][kPStride];
    threadgroup float running_max_tg[kBQ];
    threadgroup float running_sum_tg[kBQ];
    threadgroup float alpha_tg[kBQ];

    const uint tid = thread_pos.x;

    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u || threads_per_tg.z != 1u ||
        simd_lane_count != 32u || tg_pos.x != 0u ||
        B == 0u || H == 0u || HKV == 0u || H % HKV != 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D <= kBD || D > kD) return;          // 该 kernel 只处理 64 < D <= 128

    const uint bh = tg_pos.z;
    if (bh >= B * H) return;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint bkv = b * HKV + h / (H / HKV);
    const uint query_base = tg_pos.y * kBQ;
    if (query_base >= T) return;

    const bool  is_causal   = (is_causal_int != 0u);
    const bool  d_full      = (D == kD);
    const float score_scale = rsqrt(float(D));

    float2 acc[2u * kChunks];
    #pragma unroll
    for (uint i = 0u; i < 2u * kChunks; ++i) acc[i] = float2(0.0f);

    if (tid < kBQ) {
        running_max_tg[tid] = -INFINITY;
        running_sum_tg[tid] = 0.0f;
        alpha_tg[tid]       = 0.0f;
    }

    // ---------------- Q：加载一次，2 chunk × 8 行 × 16 float4 = 256 ----------------
    #pragma unroll
    for (uint it = 0u; it < 2u; ++it) {
        const uint index   = tid + it * kThreads;   // 0..255
        const uint chunk   = index >> 7;            // 0 / 1
        const uint rem     = index & 127u;
        const uint local_q = rem >> 4;
        const uint d0      = (rem & 15u) << 2;
        const uint gd      = chunk * kBD + d0;
        const uint query   = query_base + local_q;
        float4 value = float4(0.0f);
        if (query < T) {
            device const float* row = q_cache + (bh * T + query) * D;
            if (d_full) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
            } else if (gd + 3u < D) {
                value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
            } else if (gd < D) {
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                    if (gd + c < D) value[c] = row[gd + c];
                }
            }
        }
        Q_tg[chunk][local_q][d0 + 0u] = value.x;
        Q_tg[chunk][local_q][d0 + 1u] = value.y;
        Q_tg[chunk][local_q][d0 + 2u] = value.z;
        Q_tg[chunk][local_q][d0 + 3u] = value.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* KT_tg = KT_all + simd_id * kKTGroup;
    const uint key_limit = is_causal ? min(T, query_base + kBQ) : T;

    for (uint key_base = 0u; key_base < key_limit; key_base += kBK) {

        const bool tile_full     = (key_base + kBK) <= T;
        const bool tile_nocausal = (!is_causal) || (key_base + kBK <= query_base + 1u);
        const bool tile_fast     = tile_full && tile_nocausal;

        // ---------------- QK^T：两个 d chunk 复用同一块 KT ----------------
        float2 score = float2(0.0f);
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d_base = chunk * kBD;

            if (chunk != 0u) {
                simdgroup_barrier(mem_flags::mem_threadgroup);   // 复用 KT_tg 前
            }

            if (d_full && tile_full) {
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    const uint index     = lane + i * 32u;
                    const uint local_key = index >> 4;
                    const uint d0        = (index & 15u) << 2;
                    const uint key       = key_base + simd_id * kFrag + local_key;
                    const float4 value = float4(*reinterpret_cast<device const packed_float4*>(
                            k_cache + (bkv * T + key) * kD + d_base + d0));
                    KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                    KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                    KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                    KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
                }
            } else {
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    const uint index     = lane + i * 32u;
                    const uint local_key = index >> 4;
                    const uint d0        = (index & 15u) << 2;
                    const uint gd        = d_base + d0;
                    const uint key       = key_base + simd_id * kFrag + local_key;
                    float4 value = float4(0.0f);
                    if (key < T) {
                        device const float* row = k_cache + (bkv * T + key) * D;
                        if (gd + 3u < D) {
                            value = float4(*reinterpret_cast<device const packed_float4*>(row + gd));
                        } else if (gd < D) {
                            #pragma unroll
                            for (uint c = 0u; c < 4u; ++c) {
                                if (gd + c < D) value[c] = row[gd + c];
                            }
                        }
                    }
                    KT_tg[(d0 + 0u) * kKTStride + local_key] = value.x;
                    KT_tg[(d0 + 1u) * kKTStride + local_key] = value.y;
                    KT_tg[(d0 + 2u) * kKTStride + local_key] = value.z;
                    KT_tg[(d0 + 3u) * kKTStride + local_key] = value.w;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint d_sub = 0u; d_sub < kBD; d_sub += kFrag) {
                const float2 q_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(&Q_tg[chunk][0][0]),
                        kQStride, ushort2(ushort(0u), ushort(d_sub)), ushort(lane));
                const float2 k_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(KT_tg),
                        kKTStride, ushort2(ushort(d_sub), ushort(0u)), ushort(lane));
                score = lowreg_matrix::mma_8x8(q_frag, k_frag, score);
            }
        }

        // ---------------- scale + mask ----------------
        const ushort2 s_off = morton_order(ushort(lane));
        const uint local_q  = uint(s_off.y);
        const uint local_k0 = simd_id * kFrag + uint(s_off.x);
        if (tile_fast) {
            P_tg[local_q][local_k0]      = score[0] * score_scale;
            P_tg[local_q][local_k0 + 1u] = score[1] * score_scale;
        } else {
            const uint query = query_base + local_q;
            const uint key0  = key_base + local_k0;
            const uint key1  = key0 + 1u;
            const bool ok0 = (query < T) && (key0 < T) && (!is_causal || key0 <= query);
            const bool ok1 = (query < T) && (key1 < T) && (!is_causal || key1 <= query);
            P_tg[local_q][local_k0]      = ok0 ? score[0] * score_scale : -INFINITY;
            P_tg[local_q][local_k0 + 1u] = ok1 ? score[1] * score_scale : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------- 在线 softmax ----------------
        const uint row_half    = lane >> 4u;
        const uint half_lane   = lane & 15u;
        const uint softmax_row = simd_id * 2u + row_half;
        const float score0 = P_tg[softmax_row][half_lane];
        const float score1 = P_tg[softmax_row][half_lane + 16u];

        float local_max = max(score0, score1);
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(8u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(4u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(2u)));
        local_max = max(local_max, simd_shuffle_down(local_max, ushort(1u)));
        const ushort leader_lane = ushort(row_half * 16u);
        const float tile_max = simd_shuffle(local_max, leader_lane);

        const float old_max   = running_max_tg[softmax_row];
        const float new_max   = max(old_max, tile_max);
        const float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
        const float prob0     = isfinite(score0) ? exp(score0 - new_max) : 0.0f;
        const float prob1     = isfinite(score1) ? exp(score1 - new_max) : 0.0f;

        float sum = prob0 + prob1;
        sum += simd_shuffle_down(sum, ushort(8u));
        sum += simd_shuffle_down(sum, ushort(4u));
        sum += simd_shuffle_down(sum, ushort(2u));
        sum += simd_shuffle_down(sum, ushort(1u));
        const float tile_sum = simd_shuffle(sum, leader_lane);

        P_tg[softmax_row][half_lane]       = prob0;
        P_tg[softmax_row][half_lane + 16u] = prob1;
        if (half_lane == 0u) {
            running_max_tg[softmax_row] = new_max;
            running_sum_tg[softmax_row] = running_sum_tg[softmax_row] * old_scale + tile_sum;
            alpha_tg[softmax_row]       = old_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rescale = alpha_tg[uint(morton_order(ushort(lane)).y)];
        #pragma unroll
        for (uint i = 0u; i < 2u * kChunks; ++i) acc[i] *= rescale;

        // ---------------- P × V ----------------
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d_base = chunk * kBD;
            const ushort2 v_off = morton_order(ushort(lane));
            const uint v_row  = uint(v_off.y);
            const uint v_col0 = d_base + simd_id * kFrag + uint(v_off.x);
            const uint v_col1 = d_base + (simd_id + kSimdGroups) * kFrag + uint(v_off.x);

            float2 v_frag0[kVSub];
            float2 v_frag1[kVSub];
            if (d_full && tile_full) {
                #pragma unroll
                for (uint sub = 0u; sub < kVSub; ++sub) {
                    const uint key = key_base + v_row + sub * kFrag;
                    device const float* row = v_cache + (bkv * T + key) * kD;
                    v_frag0[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                    v_frag1[sub] = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                }
            } else {
                #pragma unroll
                for (uint sub = 0u; sub < kVSub; ++sub) {
                    float2 f0 = float2(0.0f);
                    float2 f1 = float2(0.0f);
                    const uint key = key_base + v_row + sub * kFrag;
                    if (key < T) {
                        device const float* row = v_cache + (bkv * T + key) * D;
                        if (v_col0 + 1u < D)      f0 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col0));
                        else if (v_col0 < D)      f0[0] = row[v_col0];
                        if (v_col1 + 1u < D)      f1 = float2(*reinterpret_cast<device const packed_float2*>(row + v_col1));
                        else if (v_col1 < D)      f1[0] = row[v_col1];
                    }
                    v_frag0[sub] = f0;
                    v_frag1[sub] = f1;
                }
            }

            float2 a0 = acc[chunk * 2u + 0u];
            float2 a1 = acc[chunk * 2u + 1u];
            #pragma unroll
            for (uint sub = 0u; sub < kVSub; ++sub) {
                const float2 p_frag = lowreg_matrix::load_tg_8x8(
                        reinterpret_cast<threadgroup const float*>(&P_tg[0][0]),
                        kPStride, ushort2(ushort(0u), ushort(sub * kFrag)), ushort(lane));
                a0 = lowreg_matrix::mma_8x8(p_frag, v_frag0[sub], a0);
                a1 = lowreg_matrix::mma_8x8(p_frag, v_frag1[sub], a1);
            }
            acc[chunk * 2u + 0u] = a0;
            acc[chunk * 2u + 1u] = a1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------------- 写出 ----------------
    const ushort2 f_off = morton_order(ushort(lane));
    const uint f_lq = uint(f_off.y);
    const uint f_q  = query_base + f_lq;
    if (f_q < T) {
        const float denom = running_sum_tg[f_lq];
        const float inv   = denom > 0.0f ? 1.0f / denom : 0.0f;
        const uint  x     = uint(f_off.x);

        if (write_lse != 0u && simd_id == 0u && x == 0u) {
            lse_out[bh * T + f_q] = denom > 0.0f ? running_max_tg[f_lq] + log(denom) : 0.0f;
        }

        device float* orow = out + (b * T + f_q) * C + h * D;
        #pragma unroll
        for (uint chunk = 0u; chunk < kChunks; ++chunk) {
            const uint d0 = chunk * kBD + simd_id * kFrag + x;
            const uint d1 = chunk * kBD + (simd_id + kSimdGroups) * kFrag + x;
            const float2 a0 = acc[chunk * 2u + 0u];
            const float2 a1 = acc[chunk * 2u + 1u];
            if (d0      < D) orow[d0]      = a0[0] * inv;
            if (d0 + 1u < D) orow[d0 + 1u] = a0[1] * inv;
            if (d1      < D) orow[d1]      = a1[0] * inv;
            if (d1 + 1u < D) orow[d1 + 1u] = a1[1] * inv;
        }
    }
}


template <uint kSimdGroups, uint kBQ, uint kBK, uint kDTile>
METAL_FUNC static void gqa_bwd_dq_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dQ,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint HKV,
    uint3 tg_pos,
    uint3 thread_pos,
    uint3 threads_per_tg,
    uint lane,
    uint sg,
    uint simd_lane_count,
    threadgroup float* Ks,
    threadgroup float* Vs
) {
    constexpr uint kFrag = 8u;
    constexpr uint kKVStride = kDTile + 4u;
    constexpr uint kKSub = kBK / kFrag;
    constexpr uint kDSub = kDTile / kFrag;
    constexpr float kLog2E = 1.4426950408889634f;

    const uint expected_threads = kSimdGroups * 32u;
    if (threads_per_tg.x != expected_threads || threads_per_tg.y != 1u ||
        threads_per_tg.z != 1u || simd_lane_count != 32u ||
        B == 0u || H == 0u || T == 0u || C == 0u || C % H != 0u) {
        return;
    }

    const uint D = C / H;
    const uint bh = tg_pos.z;
    if (D == 0u || D > 128u || bh >= B * H) {
        return;
    }

    const uint d_chunks = (D + kDTile - 1u) / kDTile;
    const uint output_chunk = tg_pos.x;
    if (output_chunk >= d_chunks) {
        return;
    }

    const uint query_block = tg_pos.y * kBQ;
    if (query_block >= T) {
        return;
    }

    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint head_base = bh * T * D;
    const uint kv_base = (b * HKV + h / (H / HKV)) * T * D;
    const uint stat_base = bh * T;
    const bool is_causal = is_causal_int != 0u;
    const uint key_limit = is_causal ? min(T, query_block + kBQ) : T;
    const float scale = rsqrt(float(D));
    const float scale_log2e = scale * kLog2E;
    const uint query_frag_base = query_block + sg * kFrag;
    const ushort2 frag_offset = morton_order(ushort(lane));

    thread float2 dq_accum[kDSub];
    #pragma unroll
    for (uint i = 0u; i < kDSub; ++i) {
        dq_accum[i] = float2(0.0f);
    }

    for (uint key_start = 0u; key_start < key_limit; key_start += kBK) {
        thread float2 score_frag[kKSub];
        thread float2 dp_frag[kKSub];
        #pragma unroll
        for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
            score_frag[key_sub] = float2(0.0f);
            dp_frag[key_sub] = float2(0.0f);
        }

        // Visit every D tile exactly once, but schedule this threadgroup's
        // output tile last.  That leaves its K tile resident for dS*K below.
        for (uint pass = 0u; pass < d_chunks; ++pass) {
            const uint d_chunk = pass + (pass >= output_chunk ? 1u : 0u);
            const uint scheduled_chunk = pass + 1u == d_chunks
                ? output_chunk : d_chunk;
            const uint d_base = scheduled_chunk * kDTile;
            const uint pair_count = kBK * (kDTile / 2u);

            for (uint pair = thread_pos.x; pair < pair_count;
                 pair += expected_threads) {
                const uint local_key = pair / (kDTile / 2u);
                const uint local_d = (pair - local_key * (kDTile / 2u)) << 1u;
                const uint key = key_start + local_key;
                const uint d = d_base + local_d;
                float2 kv = float2(0.0f);
                float2 vv = float2(0.0f);
                if (key < key_limit && d + 1u < D) {
                    const uint src = kv_base + key * D + d;
                    kv = float2(*reinterpret_cast<device const packed_float2*>(
                        k_cache + src));
                    vv = float2(*reinterpret_cast<device const packed_float2*>(
                        v_cache + src));
                } else if (key < key_limit && d < D) {
                    const uint src = kv_base + key * D + d;
                    kv[0] = k_cache[src];
                    vv[0] = v_cache[src];
                }
                *reinterpret_cast<threadgroup packed_float2*>(
                    Ks + local_key * kKVStride + local_d) = packed_float2(kv);
                *reinterpret_cast<threadgroup packed_float2*>(
                    Vs + local_key * kKVStride + local_d) = packed_float2(vv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint d_index = 0u; d_index < kDSub; ++d_index) {
                const uint global_d = d_base + d_index * kFrag;
                const uint q_row = query_frag_base + uint(frag_offset.y);
                const uint q_col = global_d + uint(frag_offset.x);
                float2 q_frag = float2(0.0f);
                float2 do_frag = float2(0.0f);
                if (q_row < T && q_col + 1u < D) {
                    q_frag = float2(*reinterpret_cast<device const packed_float2*>(
                        q_cache + head_base + q_row * D + q_col));
                    do_frag = float2(*reinterpret_cast<device const packed_float2*>(
                        dO + (b * T + q_row) * C + h * D + q_col));
                } else if (q_row < T && q_col < D) {
                    q_frag[0] = q_cache[head_base + q_row * D + q_col];
                    do_frag[0] = dO[(b * T + q_row) * C + h * D + q_col];
                }

                #pragma unroll
                for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
                    const ushort2 origin = ushort2(
                        ushort(d_index * kFrag), ushort(key_sub * kFrag));
                    const float2 k_transposed = lowreg_matrix::load_tg_8x8(
                        Ks, kKVStride, origin, ushort(lane), true);
                    const float2 v_transposed = lowreg_matrix::load_tg_8x8(
                        Vs, kKVStride, origin, ushort(lane), true);
                    score_frag[key_sub] = lowreg_matrix::mma_8x8(
                        q_frag, k_transposed, score_frag[key_sub]);
                    dp_frag[key_sub] = lowreg_matrix::mma_8x8(
                        do_frag, v_transposed, dp_frag[key_sub]);
                }
            }

            // All SIMD groups must finish reading this tile before another
            // pass overwrites it.  No barrier is needed after the last pass:
            // that tile is intentionally retained for the output MMA.
            if (pass + 1u < d_chunks) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        const uint query = query_frag_base + uint(frag_offset.y);
        const bool query_valid = query < T;
        const float row_lse_log2 = query_valid
            ? lse[stat_base + query] * kLog2E : 0.0f;
        const float row_d = query_valid ? d_val[stat_base + query] : 0.0f;

        #pragma unroll
        for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
            const uint key0 = key_start + key_sub * kFrag + uint(frag_offset.x);
            const uint key1 = key0 + 1u;
            const bool valid0 = query_valid && key0 < key_limit &&
                (!is_causal || key0 <= query);
            const bool valid1 = query_valid && key1 < key_limit &&
                (!is_causal || key1 <= query);
            const float p0 = valid0
                ? exp2(score_frag[key_sub][0] * scale_log2e - row_lse_log2)
                : 0.0f;
            const float p1 = valid1
                ? exp2(score_frag[key_sub][1] * scale_log2e - row_lse_log2)
                : 0.0f;
            score_frag[key_sub] = float2(
                scale * p0 * (dp_frag[key_sub][0] - row_d),
                scale * p1 * (dp_frag[key_sub][1] - row_d));
        }

        #pragma unroll
        for (uint d_index = 0u; d_index < kDSub; ++d_index) {
            #pragma unroll
            for (uint key_sub = 0u; key_sub < kKSub; ++key_sub) {
                const ushort2 origin = ushort2(
                    ushort(key_sub * kFrag), ushort(d_index * kFrag));
                const float2 k_frag = lowreg_matrix::load_tg_8x8(
                    Ks, kKVStride, origin, ushort(lane));
                dq_accum[d_index] = lowreg_matrix::mma_8x8(
                    score_frag[key_sub], k_frag, dq_accum[d_index]);
            }
        }

        if (key_start + kBK < key_limit) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint query = query_frag_base + uint(frag_offset.y);
    if (query < T) {
        #pragma unroll
        for (uint d_index = 0u; d_index < kDSub; ++d_index) {
            const uint d_col = output_chunk * kDTile + d_index * kFrag
                + uint(frag_offset.x);
            if (d_col >= D) {
                continue;
            }
            const uint out_base = (b * T + query) * C + h * D + d_col;
            if (d_col + 1u < D) {
                *reinterpret_cast<device packed_float2*>(dQ + out_base) =
                    packed_float2(dq_accum[d_index]);
            } else {
                dQ[out_base] = dq_accum[d_index][0];
            }
        }
    }
}


kernel void gqa_bwd_dq(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* dO [[buffer(3)]],
    device float* dQ [[buffer(4)]], device const float* lse [[buffer(5)]],
    device const float* dval [[buffer(6)]],
    constant uint& B [[buffer(7)]], constant uint& H [[buffer(8)]],
    constant uint& T [[buffer(9)]], constant uint& C [[buffer(10)]],
    constant uint& causal [[buffer(11)]], constant uint& HKV [[buffer(12)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
    uint3 nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float Ks[32u*68u], Vs[32u*68u];
    gqa_bwd_dq_impl<4u,32u,32u,64u>(q,k,v,dO,dQ,lse,dval,B,H,T,C,causal,HKV,
                                   tg,tp,nt,lane,sg,sw,Ks,Vs);
}


template <uint kMaxD, uint kSimdGroups, uint kBQ, uint kDOutChunks>
METAL_FUNC static void gqa_bwd_dkv_impl(
    device const float* q_cache,
    device const float* k_cache,
    device const float* v_cache,
    device const float* dO,
    device float* dK,
    device float* dV,
    device const float* lse,
    device const float* d_val,
    uint B,
    uint H,
    uint T,
    uint C,
    uint is_causal_int,
    uint HKV,
    uint3 tg_pos,
    uint3 thread_pos,
    uint3 threads_per_tg,
    uint lane,
    uint sg,
    uint simd_lane_count,
    threadgroup float* Qs,
    threadgroup float* dOs,
    threadgroup float* lse_tile,
    threadgroup float* dval_tile
) {
    constexpr uint kFrag = 8u;
    constexpr uint kDTile = 64u;
    constexpr uint kDSubPerTile = kDTile / kFrag;
    constexpr uint kQSub = kBQ / kFrag;
    constexpr uint kThreads = kSimdGroups * 32u;
    constexpr uint kBK = kSimdGroups * kFrag;
    constexpr uint kQStride = kDTile + 4u;
    constexpr uint kMaxChunks = (kMaxD + kDTile - 1u) / kDTile;
    constexpr uint kDOutSub = kDOutChunks * kDSubPerTile;
    static_assert((kBQ % kFrag) == 0u, "kBQ must be a multiple of 8");
    static_assert(kDOutChunks > 0u, "at least one output D tile is required");

    if (threads_per_tg.x != kThreads || threads_per_tg.y != 1u ||
        threads_per_tg.z != 1u || simd_lane_count != 32u ||
        B == 0u || H == 0u || T == 0u || C == 0u || (C % H) != 0u) {
        return;
    }
    const uint D = C / H;
    if (D == 0u || D > kMaxD) {
        return;
    }
    const uint d_chunks = (D + kDTile - 1u) / kDTile;
    const uint output_chunk_base = tg_pos.x * kDOutChunks;
    if (output_chunk_base >= d_chunks) {
        return;
    }

    const uint bh = tg_pos.z;
    const uint key_block = tg_pos.y * kBK;
    if (bh >= B * HKV || key_block >= T) {
        return;
    }

    const uint b = bh / HKV;
    const uint kv_h = bh - b * HKV;
    const uint kv_base = bh * T * D;
    const uint CKV = HKV * D;
    const uint group = H / HKV;
    const bool is_causal = is_causal_int != 0u;
    const float scale = rsqrt(float(D));
    const float scale_log2e = scale * 1.4426950408889634f;
    const uint key_frag_base = key_block + sg * kFrag;
    const ushort2 frag_offset = morton_order(ushort(lane));
    const uint key_row = key_frag_base + uint(frag_offset.y);

    // Each float2 accumulator is one lane's portion of an 8x8 output fragment.
    // kDOutChunks=1 consumes 32 floats/thread for dK+dV; =2 consumes 64.
    thread float2 dk_accum[kDOutSub];
    thread float2 dv_accum[kDOutSub];
    #pragma unroll
    for (uint i = 0u; i < kDOutSub; ++i) {
        dk_accum[i] = float2(0.0f);
        dv_accum[i] = float2(0.0f);
    }

    // One owner per KV-head/key tile; sum all query heads in registers.
    for (uint g=0u; g<group; ++g) {
    const uint h = kv_h * group + g;
    const uint head_base = (b * H + h) * T * D;
    const uint stat_base = (b * H + h) * T;
    const uint query_begin = is_causal ? key_block : 0u;
    uint q_start = query_begin;
    uint q_tile_base = head_base + q_start * D;
    uint do_tile_base = (b * T + q_start) * C + h * D;
    uint stat_tile_base = stat_base + q_start;

    for (; q_start < T; q_start += kBQ,
         q_tile_base += kBQ * D,
         do_tile_base += kBQ * C,
         stat_tile_base += kBQ) {
        // Protect the shared tile from the previous query iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_pos.x < kBQ) {
            const uint query = q_start + thread_pos.x;
            lse_tile[thread_pos.x] = query < T
                ? lse[stat_tile_base + thread_pos.x] * 1.4426950408889634f
                : 0.0f;
            dval_tile[thread_pos.x] = query < T
                ? d_val[stat_tile_base + thread_pos.x]
                : 0.0f;
        }

        thread float2 score_frag[kQSub];
        thread float2 dp_frag[kQSub];
        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            score_frag[q_sub] = float2(0.0f);
            dp_frag[q_sub] = float2(0.0f);
        }

        // Pass 1: stream all 64-column D tiles to form complete score and dP.
        for (uint d_chunk = 0u; d_chunk < kMaxChunks; ++d_chunk) {
            if (d_chunk >= d_chunks) {
                break;
            }
            if (d_chunk != 0u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const uint d_base = d_chunk * kDTile;
            for (uint local_q = sg; local_q < kBQ; local_q += kSimdGroups) {
                const uint query = q_start + local_q;
                const bool row_valid = query < T;
                const uint q_row = q_tile_base + local_q * D;
                const uint do_row = do_tile_base + local_q * C;
                for (uint local_d = lane; local_d < kDTile; local_d += 32u) {
                    const uint d = d_base + local_d;
                    float q_value = 0.0f;
                    float do_value = 0.0f;
                    if (row_valid && d < D) {
                        q_value = q_cache[q_row + d];
                        do_value = dO[do_row + d];
                    }
                    Qs[local_q * kQStride + local_d] = q_value;
                    dOs[local_q * kQStride + local_d] = do_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stream one K/V fragment at a time.  Keeping all eight fragments
            // live would add 32 floats/thread and reduce occupancy for out=2.
            #pragma unroll
            for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                const uint global_d = d_base + d_index * kFrag;
                const float2 k_frag = load_device_frag_8x8(
                    k_cache + kv_base, D, key_frag_base, global_d,
                    T, D, ushort(lane));
                const float2 v_frag = load_device_frag_8x8(
                    v_cache + kv_base, D, key_frag_base, global_d,
                    T, D, ushort(lane));
                #pragma unroll
                for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
                    const ushort2 origin = ushort2(
                        ushort(d_index * kFrag), ushort(q_sub * kFrag));
                    const float2 q_transposed = lowreg_matrix::load_tg_8x8(
                        Qs, kQStride, origin, ushort(lane), true);
                    const float2 do_transposed = lowreg_matrix::load_tg_8x8(
                        dOs, kQStride, origin, ushort(lane), true);
                    score_frag[q_sub] = lowreg_matrix::mma_8x8(
                        k_frag, q_transposed, score_frag[q_sub]);
                    dp_frag[q_sub] = lowreg_matrix::mma_8x8(
                        v_frag, do_transposed, dp_frag[q_sub]);
                }
            }
        }

        // Reuse score/dP registers for P/dS so no additional fragment arrays
        // become live alongside the output accumulators.
        #pragma unroll
        for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
            const uint q_local0 = q_sub * kFrag + uint(frag_offset.x);
            const uint q_local1 = q_local0 + 1u;
            const uint query0 = q_start + q_local0;
            const uint query1 = q_start + q_local1;
            const bool valid0 = query0 < T && (!is_causal || query0 >= key_row);
            const bool valid1 = query1 < T && (!is_causal || query1 >= key_row);
            const float p0 = valid0
                ? exp2(score_frag[q_sub][0] * scale_log2e - lse_tile[q_local0])
                : 0.0f;
            const float p1 = valid1
                ? exp2(score_frag[q_sub][1] * scale_log2e - lse_tile[q_local1])
                : 0.0f;
            const float dp0 = dp_frag[q_sub][0];
            const float dp1 = dp_frag[q_sub][1];
            score_frag[q_sub] = float2(p0, p1);
            dp_frag[q_sub] = float2(
                scale * p0 * (dp0 - dval_tile[q_local0]),
                scale * p1 * (dp1 - dval_tile[q_local1]));
        }

        // Pass 2: reload only this threadgroup's output D tiles and accumulate.
        #pragma unroll
        for (uint out_chunk = 0u; out_chunk < kDOutChunks; ++out_chunk) {
            const uint d_chunk = output_chunk_base + out_chunk;
            if (d_chunk >= d_chunks) {
                break;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint d_base = d_chunk * kDTile;
            for (uint local_q = sg; local_q < kBQ; local_q += kSimdGroups) {
                const uint query = q_start + local_q;
                const bool row_valid = query < T;
                const uint q_row = q_tile_base + local_q * D;
                const uint do_row = do_tile_base + local_q * C;
                for (uint local_d = lane; local_d < kDTile; local_d += 32u) {
                    const uint d = d_base + local_d;
                    float q_value = 0.0f;
                    float do_value = 0.0f;
                    if (row_valid && d < D) {
                        q_value = q_cache[q_row + d];
                        do_value = dO[do_row + d];
                    }
                    Qs[local_q * kQStride + local_d] = q_value;
                    dOs[local_q * kQStride + local_d] = do_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint q_sub = 0u; q_sub < kQSub; ++q_sub) {
                #pragma unroll
                for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                    const ushort2 origin = ushort2(
                        ushort(q_sub * kFrag), ushort(d_index * kFrag));
                    const float2 q_value = lowreg_matrix::load_tg_8x8(
                        Qs, kQStride, origin, ushort(lane));
                    const float2 do_value = lowreg_matrix::load_tg_8x8(
                        dOs, kQStride, origin, ushort(lane));
                    const uint accum_index = out_chunk * kDSubPerTile + d_index;
                    dk_accum[accum_index] = lowreg_matrix::mma_8x8(
                        dp_frag[q_sub], q_value, dk_accum[accum_index]);
                    dv_accum[accum_index] = lowreg_matrix::mma_8x8(
                        score_frag[q_sub], do_value, dv_accum[accum_index]);
                }
            }
        }
    }

    } // GQA group

    if (key_row < T) {
        #pragma unroll
        for (uint out_chunk = 0u; out_chunk < kDOutChunks; ++out_chunk) {
            const uint d_chunk = output_chunk_base + out_chunk;
            if (d_chunk >= d_chunks) {
                break;
            }
            #pragma unroll
            for (uint d_index = 0u; d_index < kDSubPerTile; ++d_index) {
                const uint d_col = d_chunk * kDTile + d_index * kFrag
                    + uint(frag_offset.x);
                if (d_col >= D) {
                    continue;
                }
                const uint accum_index = out_chunk * kDSubPerTile + d_index;
                const uint out_base = (b * T + key_row) * CKV + kv_h * D + d_col;
                if (d_col + 1u < D) {
                    *reinterpret_cast<device packed_float2*>(dK + out_base) =
                        packed_float2(dk_accum[accum_index]);
                    *reinterpret_cast<device packed_float2*>(dV + out_base) =
                        packed_float2(dv_accum[accum_index]);
                } else {
                    dK[out_base] = dk_accum[accum_index][0];
                    dV[out_base] = dv_accum[accum_index][0];
                }
            }
        }
    }
}


kernel void gqa_bwd_dkv(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* dO [[buffer(3)]],
    device float* dK [[buffer(4)]], device float* dV [[buffer(5)]],
    device const float* lse [[buffer(6)]], device const float* dval [[buffer(7)]],
    constant uint& B [[buffer(8)]], constant uint& H [[buffer(9)]],
    constant uint& T [[buffer(10)]], constant uint& C [[buffer(11)]],
    constant uint& causal [[buffer(12)]], constant uint& HKV [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
    uint3 nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float Qs[24u*68u], dOs[24u*68u], lse_tile[24u], dval_tile[24u];
    gqa_bwd_dkv_impl<128u,4u,24u,2u>(q,k,v,dO,dK,dV,lse,dval,B,H,T,C,causal,HKV,
                                    tg,tp,nt,lane,sg,sw,Qs,dOs,lse_tile,dval_tile);
}


// A separate launch for Q and K prevents repeated writes to a shared KV head.
// layout=0: BHTD (forward), layout=1: BTHD (projection gradients).
// negative_sin is independent of layout; backward uses the opposite sign.
kernel void gqa_rope_inplace(
    device float* data [[buffer(0)]], device const float* cs [[buffer(1)]],
    device const float* sn [[buffer(2)]], constant uint& B [[buffer(3)]],
    constant uint& H [[buffer(4)]], constant uint& T [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& layout [[buffer(7)]],
    constant uint& negative_sin [[buffer(8)]], uint3 gid [[thread_position_in_grid]]) {
    uint j=gid.x, t=gid.y, bh=gid.z, half_dim=D/2u;
    if (j>=half_dim || t>=T || bh>=B*H) return;
    uint b=bh/H, h=bh%H;
    uint base=layout==0u ? (bh*T+t)*D : ((b*T+t)*H+h)*D;
    float c=cs[t*half_dim+j], s=sn[t*half_dim+j];
    if (negative_sin) s=-s;
    float a=data[base+j], v=data[base+j+half_dim];
    data[base+j]=fma(a,c,-v*s);
    data[base+j+half_dim]=fma(a,s,v*c);
}
// Recompute weighted normalized activations after attention derivatives are done.
kernel void gqa_rms_recompute(
    device const float* x [[buffer(0)]], device const float* gamma [[buffer(1)]],
    device float* z [[buffer(2)]], device float* inv [[buffer(3)]],
    constant uint& N [[buffer(4)]], constant uint& C [[buffer(5)]],
    constant uint& eps_bits [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=tg.y*4u+sg;
    if (row>=N) return;
    float ss=0.0f;
    for (uint c=lane;c<C;c+=32u) { float a=x[row*C+c]; ss=fma(a,a,ss); }
    float r=rsqrt(simd_sum(ss)/float(C)+as_type<float>(eps_bits));
    if (lane==0u) inv[row]=r;
    for (uint c=lane;c<C;c+=32u) z[row*C+c]=x[row*C+c]*r*gamma[c];
}
kernel void gqa_rms_dgamma(
    device const float* x [[buffer(0)]], device const float* dz [[buffer(1)]],
    device const float* inv [[buffer(2)]], device float* dg [[buffer(3)]],
    constant uint& N [[buffer(4)]], constant uint& C [[buffer(5)]],
    constant uint& accumulate [[buffer(6)]], uint c [[thread_position_in_grid]]) {
    if (c>=C) return;
    float a=0.0f;
    for (uint row=0u;row<N;++row) a=fma(dz[row*C+c],x[row*C+c]*inv[row],a);
    dg[c]=accumulate ? dg[c]+a : a;
}
kernel void gqa_rms_dx(
    device const float* x [[buffer(0)]], device const float* dz [[buffer(1)]],
    device const float* gamma [[buffer(2)]], device const float* inv [[buffer(3)]],
    device float* dx [[buffer(4)]], constant uint& N [[buffer(5)]],
    constant uint& C [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row=tg.y;
    if (row>=N) return;
    float r=inv[row], dotval=0.0f;
    for (uint c=lane;c<C;c+=32u)
        dotval=fma(dz[row*C+c]*gamma[c],x[row*C+c]*r,dotval);
    float mean=simd_sum(dotval)/float(C);
    for (uint c=lane;c<C;c+=32u)
        dx[row*C+c]+=r*(dz[row*C+c]*gamma[c]-x[row*C+c]*r*mean);
}

kernel void fused_rmsnorm_linear_ce_dx(
    device const float* x       [[buffer(0)]],
    device const float* dz      [[buffer(1)]],
    device const float* gamma   [[buffer(2)]],
    device const float* inv     [[buffer(3)]],
    device float* dx            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    constant uint& C            [[buffer(6)]],
    uint3 tg                    [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]]
) {
    const uint row = tg.y;
    if (row >= N) {
        return;
    }

    const uint base = row * C;
    const float r = inv[row];
    float dotval = 0.0f;

    for (uint c = lane; c < C; c += 32u) {
        const float x_hat = x[base + c] * r;
        const float dy = dz[base + c] * gamma[c];
        dotval = fma(dy, x_hat, dotval);
    }

    const float mean_dot = simd_sum(dotval) / float(C);

    for (uint c = lane; c < C; c += 32u) {
        const float x_hat = x[base + c] * r;
        const float dy = dz[base + c] * gamma[c];
        dx[base + c] = r * (dy - x_hat * mean_dot);
    }
}

kernel void fused_rmsnorm_linear_ce_restore_target(
    device const float* z [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const int* targets [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device const float* row_scale [[buffer(4)]],
    device float* dlogits [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& V [[buffer(8)]],
    constant uint& row_base [[buffer(9)]],
    constant uint& scale_bits [[buffer(10)]],
    constant uint& has_row_scale [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg.y;

    if (row >= rows) {
        return;
    }

    const uint global_row = row_base + row;
    const int target = targets[global_row];

    if (target < 0 || target >= int(V)) {
        return;
    }

    float dot = 0.0f;

    for (uint c = lane; c < C; c += 32u) {
        dot = fma(
            z[row * C + c],
            w[uint(target) * C + c],
            dot
        );
    }

    const float target_logit = simd_sum(dot);

    if (lane == 0u) {
        const float scale =
            has_row_scale != 0u
            ? row_scale[global_row]
            : as_type<float>(scale_bits);

        dlogits[row * V + uint(target)] =
            scale * exp(target_logit - lse[global_row]);
    }
}

kernel void fused_rmsnorm_linear_ce_subtract_target(
    device const float* w [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device const float* row_scale [[buffer(2)]],
    device float* dz [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& V [[buffer(6)]],
    constant uint& row_base [[buffer(7)]],
    constant uint& scale_bits [[buffer(8)]],
    constant uint& has_row_scale [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint row = gid.y;

    if (row >= rows || c >= C) {
        return;
    }

    const uint global_row = row_base + row;
    const int target = targets[global_row];

    if (target < 0 || target >= int(V)) {
        return;
    }

    const float scale =
        has_row_scale != 0u
        ? row_scale[global_row]
        : as_type<float>(scale_bits);

    dz[row * C + c] = fma(
        -scale,
        w[uint(target) * C + c],
        dz[row * C + c]
    );
}

kernel void fused_rmsnorm_linear_ce_dw_tail(
    device const float* dlogits [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device float* dw [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& V [[buffer(5)]],
    constant uint& accumulate [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c0 = gid.x * 4u;
    const uint v = gid.y;

    if (c0 >= C || v >= V) {
        return;
    }

    if (c0 + 3u < C) {
        float4 sum = float4(0.0f);

        for (uint row = 0u; row < rows; ++row) {
            const float4 z_value =
                float4(*reinterpret_cast<device const packed_float4*>(
                    z + row * C + c0
                ));

            const float dlogit = dlogits[row * V + v];
            sum = fma(float4(dlogit), z_value, sum);
        }

        if (accumulate != 0u) {
            sum += float4(*reinterpret_cast<device const packed_float4*>(
                dw + v * C + c0
            ));
        }

        *reinterpret_cast<device packed_float4*>(
            dw + v * C + c0
        ) = packed_float4(sum);
    } else {
        for (uint c = c0; c < C; ++c) {
            float sum = 0.0f;

            for (uint row = 0u; row < rows; ++row) {
                sum = fma(
                    dlogits[row * V + v],
                    z[row * C + c],
                    sum
                );
            }

            dw[v * C + c] =
                accumulate != 0u
                ? dw[v * C + c] + sum
                : sum;
        }
    }
}

kernel void ce_tile_logits_to_dlogits(
    device float* tile              [[buffer(0)]],
    device const float* lse         [[buffer(1)]],
    device const int* targets       [[buffer(2)]],
    constant uint& tile_v           [[buffer(3)]],
    constant uint& vocab_base       [[buffer(4)]],
    constant uint& V                [[buffer(5)]],
    constant uint& row_base         [[buffer(6)]],
    constant uint& scale_bits       [[buffer(7)]],
    uint id                        [[thread_position_in_grid]],
    uint count                     [[threads_per_grid]]
) {
    if (id >= count) {
        return;
    }

    const uint row = row_base + id / tile_v;
    const uint vocab = vocab_base + id % tile_v;
    const int target = targets[row];

    // 与当前原有 CE 内核的处理保持一致。
    if (target < 0 || target >= int(V)) {
        tile[id] = 0.0f;
        return;
    }

    const float p = exp2(
        (tile[id] - lse[row]) * 1.4426950408889634f
    );

    tile[id] = (
        p - float(vocab == uint(target))
    ) * as_type<float>(scale_bits);
}


kernel void fp32_to_bf16(
    device const float* src         [[buffer(0)]],
    device ushort* dst              [[buffer(1)]],
    constant uint& accumulate       [[buffer(2)]],
    uint id                         [[thread_position_in_grid]],
    uint count                      [[threads_per_grid]]
) {
    if (id >= count) {
        return;
    }

    float value = src[id];

    if (accumulate != 0u) {
        value += as_type<float>(uint(dst[id]) << 16);
    }

    const uint bits = as_type<uint>(value);

    if ((bits & 0x7f800000u) == 0x7f800000u) {
        // 保留 Inf，确保 NaN 截断后仍然为 NaN。
        dst[id] = ushort(
            (bits >> 16)
            | ((bits & 0x007fffffu) != 0u ? 0x0040u : 0u)
        );
    } else {
        // Round-to-nearest, ties-to-even。
        dst[id] = ushort(
            (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16
        );
    }
}

kernel void swiglu_forward_tile_kernel(
    device const float* gu       [[buffer(0)]],
    device float* hidden_tile    [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& H            [[buffer(3)]],
    constant uint& h0           [[buffer(4)]],
    constant uint& h            [[buffer(5)]],
    uint gid                    [[thread_position_in_grid]]
) {
    const uint VEC = 4u;
    const uint vectors_per_row = (h + VEC - 1u) / VEC;

    if (vectors_per_row == 0u) {
        return;
    }

    const uint row = gid / vectors_per_row;
    const uint col = (gid % vectors_per_row) * VEC;

    if (row >= N || col >= h) {
        return;
    }

    const uint gu_base = row * (2u * H);
    const uint hidden_base = row * h;
    const uint lanes = min(VEC, h - col);

    float4 gate = float4(0.0f);
    float4 up = float4(0.0f);

    for (uint i = 0u; i < lanes; ++i) {
        const uint src_col = h0 + col + i;
        gate[i] = gu[gu_base + src_col];
        up[i] = gu[gu_base + H + src_col];
    }

    // 与原 swiglu_forward_kernel 保持相同计算表达式。
    const float4 sigmoid_gate =
        1.0f / (1.0f + exp(-gate));

    const float4 result = gate * sigmoid_gate * up;

    for (uint i = 0u; i < lanes; ++i) {
        hidden_tile[hidden_base + col + i] = result[i];
    }
}

inline ushort fla_float_to_bf16_rne(float value)
{
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return ushort((bits >> 16) | ((bits & 0x007fffffu) != 0u ? 0x0040u : 0u));
    }

    // Round-to-nearest, ties-to-even。
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

kernel void bias_grad_reduce_bf16(
    device const float* grad_out [[buffer(0)]],
    device ushort* grad_bias    [[buffer(1)]],
    constant uint& rows         [[buffer(2)]],
    constant uint& C            [[buffer(3)]],
    constant uint& accumulate   [[buffer(4)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kThreads = 256u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kSimdGroups = kThreads / kSimdWidth;

    constexpr uint kBN = 16u;
    constexpr uint kVecWidth = 4u;
    constexpr uint kVecsPerBlock = kBN / kVecWidth;

    const uint col0 = tg_pos.y * kBN;
    const uint valid_cols = min(kBN, C - col0);
    const uint full_vecs = valid_cols / kVecWidth;
    const uint tail_cols = valid_cols % kVecWidth;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    for (uint row = tid; row < rows; row += kThreads) {
        const device float* src = grad_out + row * C + col0;

        if (full_vecs >= 1u) {
            acc0 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 0u
                )
            );
        }

        if (full_vecs >= 2u) {
            acc1 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 4u
                )
            );
        }

        if (full_vecs >= 3u) {
            acc2 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 8u
                )
            );
        }

        if (full_vecs >= 4u) {
            acc3 += float4(
                *reinterpret_cast<device const packed_float4*>(
                    src + 12u
                )
            );
        }

        if (tail_cols != 0u) {
            const device float* tail_src =
                src + full_vecs * kVecWidth;

            float4 tail = float4(0.0f);

            if (tail_cols >= 1u) {
                tail.x = tail_src[0];
            }
            if (tail_cols >= 2u) {
                tail.y = tail_src[1];
            }
            if (tail_cols >= 3u) {
                tail.z = tail_src[2];
            }

            if (full_vecs == 0u) {
                acc0 += tail;
            } else if (full_vecs == 1u) {
                acc1 += tail;
            } else if (full_vecs == 2u) {
                acc2 += tail;
            } else {
                acc3 += tail;
            }
        }
    }

    const float4 simd_sum0 = float4(
        simd_sum(acc0.x),
        simd_sum(acc0.y),
        simd_sum(acc0.z),
        simd_sum(acc0.w)
    );

    const float4 simd_sum1 = float4(
        simd_sum(acc1.x),
        simd_sum(acc1.y),
        simd_sum(acc1.z),
        simd_sum(acc1.w)
    );

    const float4 simd_sum2 = float4(
        simd_sum(acc2.x),
        simd_sum(acc2.y),
        simd_sum(acc2.z),
        simd_sum(acc2.w)
    );

    const float4 simd_sum3 = float4(
        simd_sum(acc3.x),
        simd_sum(acc3.y),
        simd_sum(acc3.z),
        simd_sum(acc3.w)
    );

    threadgroup float4 tg_partial[kSimdGroups][kVecsPerBlock];

    if (lane == 0u) {
        tg_partial[simd_id][0] = simd_sum0;
        tg_partial[simd_id][1] = simd_sum1;
        tg_partial[simd_id][2] = simd_sum2;
        tg_partial[simd_id][3] = simd_sum3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float4 sums[kVecsPerBlock];

        sums[0] = float4(0.0f);
        sums[1] = float4(0.0f);
        sums[2] = float4(0.0f);
        sums[3] = float4(0.0f);

        for (uint group = 0u; group < kSimdGroups; ++group) {
            sums[0] += tg_partial[group][0];
            sums[1] += tg_partial[group][1];
            sums[2] += tg_partial[group][2];
            sums[3] += tg_partial[group][3];
        }

        device ushort* dst = grad_bias + col0;
        for (uint col = 0u; col < valid_cols; ++col) {
            float value = sums[col / kVecWidth][col % kVecWidth];
            if (accumulate != 0u) {
                value += as_type<float>(uint(dst[col]) << 16);
            }
            dst[col] = fla_float_to_bf16_rne(value);
        }
    }
}

inline void fa_store_bias_grad_mixed(
    device uchar* dst,
    uint index,
    float reduced,
    uint accumulate,
    bool is_bf16
) {
    if (is_bf16) {
        device ushort* out = reinterpret_cast<device ushort*>(dst);

        float value = reduced;
        if (accumulate != 0u) {
            value = as_type<float>(uint(out[index]) << 16) + reduced;
        }

        out[index] = fla_float_to_bf16_rne(value);
    } else {
        device float* out = reinterpret_cast<device float*>(dst);
        out[index] = accumulate != 0u
            ? out[index] + reduced
            : reduced;
    }
}

kernel void fused_qkv_bias_grad_bf16(
    device const float* dQ       [[buffer(0)]],
    device const float* dK       [[buffer(1)]],
    device const float* dV       [[buffer(2)]],
    device uchar* dbq            [[buffer(3)]],
    device uchar* dbk            [[buffer(4)]],
    device uchar* dbv            [[buffer(5)]],
    constant uint& rows          [[buffer(6)]],
    constant uint& C             [[buffer(7)]],
    constant uint& accumulate    [[buffer(8)]],
    constant uint& enabled_mask  [[buffer(9)]],
    constant uint& bf16_mask     [[buffer(10)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 thread_pos             [[thread_position_in_threadgroup]]
) {
    constexpr uint kThreads = 128;
    constexpr uint kChannelTile = 4;

    const uint tid = thread_pos.x;
    if (tid >= kThreads || tg_pos.x != 0) {
        return;
    }
    if (rows == 0 || C == 0) {
        return;
    }

    const uint c_base = tg_pos.y * kChannelTile;
    if (c_base >= C) {
        return;
    }
    const uint tile = min(kChannelTile, C - c_base);

    float sum_q[kChannelTile];
    float sum_k[kChannelTile];
    float sum_v[kChannelTile];

    for (uint i = 0; i < kChannelTile; ++i) {
        sum_q[i] = 0.0f;
        sum_k[i] = 0.0f;
        sum_v[i] = 0.0f;
    }

    for (uint row = tid; row < rows; row += kThreads) {
        const uint base = row * C + c_base;
        for (uint i = 0; i < tile; ++i) {
            sum_q[i] += dQ[base + i];
            sum_k[i] += dK[base + i];
            sum_v[i] += dV[base + i];
        }
    }

    threadgroup float reduce_buf[kThreads];

    for (uint i = 0; i < tile; ++i) {
        // dbq
        reduce_buf[tid] = sum_q[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0 && (enabled_mask & 1u) != 0u) {
            fa_store_bias_grad_mixed(
                dbq, c_base + i, reduce_buf[0],
                accumulate, (bf16_mask & 1u) != 0u
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dbk
        reduce_buf[tid] = sum_k[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0 && (enabled_mask & 2u) != 0u) {
            fa_store_bias_grad_mixed(
                dbk, c_base + i, reduce_buf[0],
                accumulate, (bf16_mask & 2u) != 0u
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dbv
        reduce_buf[tid] = sum_v[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0 && (enabled_mask & 4u) != 0u) {
            fa_store_bias_grad_mixed(
                dbv, c_base + i, reduce_buf[0],
                accumulate, (bf16_mask & 4u) != 0u
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gather_backward_bf16_kernel(
    device const int* indices       [[buffer(0)]],
    device const float* grad_out    [[buffer(1)]],
    device ushort* grad_weight      [[buffer(2)]],
    constant uint& rows             [[buffer(3)]],
    constant uint& C                [[buffer(4)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    constexpr uint kSimdWidth = 32u;
    constexpr uint kChannelsPerThread = 4u;
    constexpr uint kChannelTile = kSimdWidth * kChannelsPerThread;

    const uint source_row = tg_pos.z;
    const int token = indices[source_row];

    bool appeared_before = false;
    for (uint row = lane; row < source_row; row += kSimdWidth) {
        if (indices[row] == token) {
            appeared_before = true;
            break;
        }
    }

    if (simd_any(appeared_before)) {
        return;
    }

    const uint channel = tg_pos.y * kChannelTile + lane * kChannelsPerThread;

    if (channel >= C) {
        return;
    }

    const uint valid_channels = min(kChannelsPerThread, C - channel);
    float4 sum = float4(0.0f);
    for (uint row = source_row; row < rows; ++row) {
        if (indices[row] != token) {
            continue;
        }
        const ulong src_base = ulong(row) * ulong(C) + ulong(channel);
        for (uint i = 0u; i < valid_channels; ++i) {
            sum[i] += grad_out[src_base + i];
        }
    }

    const ulong dst_base = ulong(uint(token)) * ulong(C) + ulong(channel);
    for (uint i = 0u; i < valid_channels; ++i) {
        const float old_value = as_type<float>(uint(grad_weight[dst_base + i]) << 16);
        grad_weight[dst_base + i] = fla_float_to_bf16_rne(old_value + sum[i]);
    }
}

kernel void fused_rmsnorm_linear_ce_dw_bf16(
    device const float* dlogits [[buffer(0)]],
    device const float* z       [[buffer(1)]],
    device ushort* dw          [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& C           [[buffer(4)]],
    constant uint& V           [[buffer(5)]],
    constant uint& accumulate  [[buffer(6)]],
    uint3 gid                  [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint v = gid.y;

    if (c >= C || v >= V) {
        return;
    }

    float sum = 0.0f;
    for (uint row = 0u; row < rows; ++row) {
        sum = fma(dlogits[row * V + v], z[row * C + c], sum);
    }

    const size_t index = size_t(v) * size_t(C) + size_t(c);
    if (accumulate != 0u) {
        sum += as_type<float>(uint(dw[index]) << 16);
    }
    dw[index] = fla_float_to_bf16_rne(sum);
}

kernel void gqa_rms_dgamma_bf16(
    device const float* x      [[buffer(0)]],
    device const float* dz     [[buffer(1)]],
    device const float* inv    [[buffer(2)]],
    device ushort* dg         [[buffer(3)]],
    device float* acc         [[buffer(4)]],
    constant uint& N          [[buffer(5)]],
    constant uint& C          [[buffer(6)]],
    constant uint& accumulate [[buffer(7)]],
    constant uint& first      [[buffer(8)]],
    constant uint& last       [[buffer(9)]],
    uint c                    [[thread_position_in_grid]]
) {
    if (c >= C) {
        return;
    }

    float sum = first != 0u ? 0.0f : acc[c];
    for (uint row = 0u; row < N; ++row) {
        const size_t index = size_t(row) * size_t(C) + size_t(c);
        sum = fma(dz[index], x[index] * inv[row], sum);
    }

    if (last == 0u) {
        acc[c] = sum;
    } else {
        if (accumulate != 0u) {
            sum += as_type<float>(uint(dg[c]) << 16);
        }
        dg[c] = fla_float_to_bf16_rne(sum);
    }
}

////////////////////////////////////////////////////////////////////////////
///                               量化
////////////////////////////////////////////////////////////////////////////
METAL_FUNC uint fp16_quant_abs_bits(float x) {
    return as_type<uint>(x) & 0x7fffffffu;
}

METAL_FUNC float fp16_quant_scale(uint max_bits) {
    if (max_bits == 0u || max_bits >= 0x7f800000u) {
        return 1.0f;
    }

    const float amax = as_type<float>(max_bits);
    // FP16 最大有限值为65504。scale下限取FP32最小正规数，避免倒数溢出。
    return max(amax / 65504.0f, 0x1.0p-126f);
}

METAL_FUNC half fp16_quant_value(float x, float inv_scale, bool invalid) 
{
    if (invalid) {
        return as_type<half>(ushort(0x7e00u));
    }
    const float scaled = clamp(x * inv_scale, -65504.0f, 65504.0f);
    return half(scaled);
}

kernel void per_token_cast_to_fp16(
    device const float* src       [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device float* scales          [[buffer(2)]],
    constant uint& M              [[buffer(3)]],
    constant uint& K              [[buffer(4)]],
    constant uint& src_row_stride [[buffer(5)]],
    constant uint& src_col_stride [[buffer(6)]],
    uint3 tg                      [[threadgroup_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tg.z * 4u + sg;
    const uint tile_k = tg.y;
    if (row >= M) {
        return;
    }

    const uint groups_k = K / 128u + uint(K % 128u != 0u);
    const uint k0 = tile_k * 128u;
    float values[4];
    uint max_bits = 0u;

    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint k = k0 + lane + j * 32u;

        float x = 0.0f;
        if (k < K) {
            const ulong index = ulong(row) * ulong(src_row_stride) + ulong(k) * ulong(src_col_stride);
            x = src[index];
        }
        values[j] = x;
        max_bits = max(max_bits, fp16_quant_abs_bits(x));
    }

    max_bits = simd_max(max_bits);
    const bool invalid = max_bits >= 0x7f800000u;
    const float scale = fp16_quant_scale(max_bits);
    const float inv_scale = 1.0f / scale;

    if (lane == 0u) {
        scales[ulong(row) * ulong(groups_k) + ulong(tile_k)] = invalid ? as_type<float>(0x7fc00000u) : scale;
    }

    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint k = k0 + lane + j * 32u;
        if (k < K) {
            dst[ulong(row) * ulong(K) + ulong(k)] = fp16_quant_value(values[j], inv_scale, invalid);
        }
    }
}

kernel void per_block_cast_to_fp16(
    device const float* src       [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device float* scales          [[buffer(2)]],
    constant uint& N              [[buffer(3)]],
    constant uint& K              [[buffer(4)]],
    constant uint& src_row_stride [[buffer(5)]],
    constant uint& src_col_stride [[buffer(6)]],
    uint3 tg                      [[threadgroup_position_in_grid]],
    uint3 tid3                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint block_k = tg.y;
    const uint block_n = tg.z;
    const uint n0 = block_n * 128u;
    const uint k0 = block_k * 128u;
    const uint groups_k = K / 128u + uint(K % 128u != 0u);

    threadgroup uint group_max_bits[8];

    uint max_bits = 0u;
    // 第一遍：每线程最多读取64个元素，只保留局部最大值。
    for (uint p = tid; p < 128u * 128u; p += 256u) {
        const uint n = n0 + p / 128u;
        const uint k = k0 + p % 128u;
        if (n < N && k < K) {
            const ulong index = ulong(n) * ulong(src_row_stride) + ulong(k) * ulong(src_col_stride);
            max_bits = max(max_bits, fp16_quant_abs_bits(src[index]));
        }
    }
    max_bits = simd_max(max_bits);
    if (lane == 0u) {
        group_max_bits[sg] = max_bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个SIMD-group都归约同样的8个组结果，
    // 因而所有线程得到相同block最大值，无需第二次barrier。
    uint block_max_bits = 0u;
    if (lane < 8u) {
        block_max_bits = group_max_bits[lane];
    }
    block_max_bits = simd_max(block_max_bits);

    const bool invalid = block_max_bits >= 0x7f800000u;
    const float scale = fp16_quant_scale(block_max_bits);
    const float inv_scale = 1.0f / scale;
    if (tid == 0u) {
        scales[ulong(block_n) * ulong(groups_k) + ulong(block_k)] = invalid ? as_type<float>(0x7fc00000u) : scale;
    }

    // 第二遍：重新读取原输入，写出连续FP16结果。
    for (uint p = tid; p < 128u * 128u; p += 256u) {
        const uint n = n0 + p / 128u;
        const uint k = k0 + p % 128u;
        if (n < N && k < K) {
            const ulong src_index = ulong(n) * ulong(src_row_stride) + ulong(k) * ulong(src_col_stride);
            const ulong dst_index = ulong(n) * ulong(K) + ulong(k);
            dst[dst_index] = fp16_quant_value(src[src_index], inv_scale, invalid);
        }
    }
}

METAL_FUNC uint ceil_div(uint a, uint b) {
    return (a + b - 1u) / b;
}

template <uint BLOCK_M, uint BLOCK_N>
METAL_FUNC uint get_num_1d_blocks_per_group(uint P) {
    uint best_h = 8u;
    uint min_usage = 8u * BLOCK_M + ceil_div(P, 8u) * BLOCK_N;
    const uint usage_16 = 16u * BLOCK_M + ceil_div(P, 16u) * BLOCK_N;
    if (usage_16 < min_usage) {
        best_h = 16u;
    }
    return best_h;
}

METAL_FUNC void get_swizzled_block_idx(
    uint num_1d_blocks_per_group,
    uint num_m_blocks,
    uint num_n_blocks,
    uint q,
    thread uint& m_block_idx,
    thread uint& n_block_idx
) {
    const uint h = num_1d_blocks_per_group;
    const uint num_blocks_per_group = h * num_n_blocks;
    const uint g = q / num_blocks_per_group;
    const uint m0 = g * h;
    const uint r = q % num_blocks_per_group;
    const uint h_actual = min(h, num_m_blocks - m0);
    m_block_idx = m0 + r % h_actual;
    n_block_idx = r / h_actual;
}

METAL_FUNC bool get_next_block(
    thread int& current_iter,
    uint persistent_blocks,
    uint block_idx,
    uint num_blocks,
    uint num_1d_blocks_per_group,
    uint num_m_blocks,
    uint num_n_blocks,
    thread uint& m_block_idx,
    thread uint& n_block_idx
) {
    const uint t = uint(++current_iter);
    const uint q = t * persistent_blocks + block_idx;
    if (q >= num_blocks) {
        return false;
    }
    get_swizzled_block_idx(num_1d_blocks_per_group, num_m_blocks, num_n_blocks, q, m_block_idx, n_block_idx);
    return true;
}

METAL_FUNC uint get_global_idx(uint block_idx, uint block_size) {
    return block_idx * block_size;
}

template<uint BK>
METAL_FUNC uint fp16_gemm_ab_index(uint row, uint k) {
    constexpr uint ROW_MASK = BK / 8u - 1u;
    return row * BK + (k ^ ((row & ROW_MASK) << 3u));
}

template<uint BLOCK_M, uint BLOCK_N, uint BLOCK_K, uint NUM_STAGES, bool WITH_ACCUMULATION>
METAL_FUNC void fp16_gemm_impl(
    device const half* a,
    device const half* b,
    device const float* sa,
    device const float* sb,
    device float* d,
    device const float* c,
    constant uint& M,
    constant uint& N,
    constant uint& K,
    constant uint& num_m_blocks,
    constant uint& num_n_blocks,
    uint3 tg,
    uint3 tg_grid,
    uint tid,
    threadgroup uchar* smem_buffer
) {
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint NUM_MATH_THREADS = BLOCK_M*BLOCK_N <= 64u*64u ? 128u : 512u;
    constexpr uint NUM_MATH_SIMDGROUPS = NUM_MATH_THREADS / SIMD_WIDTH;
    constexpr uint NUM_LOADER_SIMDGROUPS = 2u;
    constexpr uint NUM_THREADS = NUM_MATH_THREADS + NUM_LOADER_SIMDGROUPS * SIMD_WIDTH;

    const uint sg = tid / SIMD_WIDTH;
    const uint lane = tid % SIMD_WIDTH;
    const bool is_loader = sg < NUM_LOADER_SIMDGROUPS;
    const bool is_math = sg >= NUM_LOADER_SIMDGROUPS;
    const uint math_sg = sg - NUM_LOADER_SIMDGROUPS;

    constexpr uint SCALE_GROUP = 128u;
    constexpr uint STAGES_PER_SCALE_GROUP = SCALE_GROUP / BLOCK_K;
    constexpr uint SMEM_CAPACITY = 32768u;
    constexpr uint SMEM_SYNC_BYTES = 256u;

    // constexpr uint SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
    constexpr uint SMEM_D_SIZE = 0u;
    constexpr uint SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(half);
    constexpr uint SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(half);
    constexpr uint SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    constexpr uint NUM_SFB_ROWS = (BLOCK_K % BLOCK_N == 0u) ? 1u : 2u;
    constexpr uint SMEM_A_OFFSET = SMEM_D_SIZE;
    constexpr uint SMEM_B_OFFSET = SMEM_A_OFFSET + NUM_STAGES * SMEM_A_SIZE_PER_STAGE;
    constexpr uint SMEM_SFA_OFFSET = SMEM_B_OFFSET + NUM_STAGES * SMEM_B_SIZE_PER_STAGE;

    const uint group_k = (K + SCALE_GROUP - 1u) / SCALE_GROUP;
    const uint group_n = (N + SCALE_GROUP - 1u) / SCALE_GROUP;
    // Scale-B 是当前输出 tile 沿完整 K 方向所需的 scale。
    const uint smem_sfb_bytes = ((group_k * NUM_SFB_ROWS * sizeof(float) + 15u) / 16u) * 16u;
    const uint smem_sfb_offset = SMEM_SFA_OFFSET + NUM_STAGES * SMEM_SFA_SIZE_PER_STAGE;
    const uint smem_sync_offset = smem_sfb_offset + smem_sfb_bytes;

    // 每个 stage: 
    //  full_phase, empty_phase, empty_count; 
    //  math barrier: math_count, math_phase,;
    //  loader barrier: loader_count, loader_phase
    constexpr uint sync_bytes = (3u * NUM_STAGES + 4u) * sizeof(atomic_uint);
    constexpr uint sync_bytes_aligned = (sync_bytes + 15u) & ~15u;
    constexpr uint mma_scratch_bytes = NUM_MATH_SIMDGROUPS * 8u * 8u * sizeof(float);
    const uint smem_mma_scratch_offset = smem_sync_offset + sync_bytes_aligned;

    // threadgroup float* smem_d = reinterpret_cast<threadgroup float*>(smem_buffer);
    threadgroup half* smem_a = reinterpret_cast<threadgroup half*>(smem_buffer + SMEM_A_OFFSET);
    threadgroup half* smem_b = reinterpret_cast<threadgroup half*>(smem_buffer + SMEM_B_OFFSET);
    threadgroup float* smem_sfa = reinterpret_cast<threadgroup float*>(smem_buffer + SMEM_SFA_OFFSET);
    threadgroup float* smem_sfb = reinterpret_cast<threadgroup float*>(smem_buffer + smem_sfb_offset);
    threadgroup float* mma_scratch = reinterpret_cast<threadgroup float*>(smem_buffer + smem_mma_scratch_offset);

    // 每个 stage 两个 atomic_uint：full_seq、empty_count
    threadgroup atomic_uint* full_phase = reinterpret_cast<threadgroup atomic_uint*>(smem_buffer + smem_sync_offset);
    threadgroup atomic_uint* empty_phase = full_phase + NUM_STAGES;
    threadgroup atomic_uint* empty_count = empty_phase + NUM_STAGES;
    threadgroup atomic_uint* math_count = empty_count + NUM_STAGES;
    threadgroup atomic_uint* math_phase = math_count + 1;

    threadgroup atomic_uint* loader_count = math_phase + 1;
    threadgroup atomic_uint* loader_phase = loader_count + 1;

    if (tid == 0u) {
        for (uint stage = 0; stage < NUM_STAGES; ++stage) {
            atomic_store_explicit(full_phase + stage, 0u, memory_order_relaxed);
            atomic_store_explicit(empty_phase + stage, 0u, memory_order_relaxed);
            atomic_store_explicit(empty_count + stage, 0u, memory_order_relaxed);
        }
        atomic_store_explicit(math_count, 0u, memory_order_relaxed);
        atomic_store_explicit(math_phase, 0u, memory_order_relaxed);
        atomic_store_explicit(loader_count, 0u, memory_order_relaxed);
        atomic_store_explicit(loader_phase, 0u, memory_order_relaxed);
    }

    // 只在初始化时使用全 threadgroup barrier
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint P = tg_grid.x;
    const uint block_idx = tg.x;
    const uint num_blocks = num_m_blocks * num_n_blocks;
    const uint num_1d_blocks_per_group = get_num_1d_blocks_per_group<BLOCK_M, BLOCK_N>(P);
    const uint num_k_blocks = (K + BLOCK_K - 1u) / BLOCK_K;

    // 两条路径各自保存 scheduler 和 pipeline 状态。
    uint stage_idx = 0u;
    uint phase = 0u;

    if (is_loader) { // Producer
        uint loader_local_phase = 0u; // 每次完成一个K tile的搬运翻转
        uint m_block_idx;
        uint n_block_idx;
        int current_iter = -1;
        while (get_next_block(current_iter, P, block_idx, num_blocks, num_1d_blocks_per_group, 
                              num_m_blocks, num_n_blocks, m_block_idx, n_block_idx)) {
            const uint m_start = get_global_idx(m_block_idx, BLOCK_M);
            const uint n_start = get_global_idx(n_block_idx, BLOCK_N);

            for (uint k_block_idx = 0u; k_block_idx < num_k_blocks; ++k_block_idx) {
                const uint stage = stage_idx;
                const uint k_start = get_global_idx(k_block_idx, BLOCK_K);

                if (lane == 0u) { // producer等待empty, 轮询当前stage是否可复用
                    while (atomic_load_explicit(empty_phase + stage, memory_order_relaxed) == (phase ^ 1u)) {
                    }
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    if (sg == 0u) { // 由loader 0复位
                        atomic_store_explicit(empty_count + stage, 0u, memory_order_relaxed);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup half* stage_a = smem_a + stage * BLOCK_M * BLOCK_K;
                threadgroup half* stage_b = smem_b + stage * BLOCK_N * BLOCK_K;
                threadgroup float* stage_sfa = smem_sfa + stage * BLOCK_M;

                const uint vector_k = min(BLOCK_K, K - k_start) & ~3u;
                if (sg == 0u) {
                    // A：[M,K]
                    for (uint index = lane; index < BLOCK_M * (vector_k / 4u); index += SIMD_WIDTH) {
                        const uint local_m = index / (vector_k/4u);
                        const uint local_k = index % (vector_k/4u)*4u;
                        const uint global_m = m_start + local_m;
                        const uint global_k = k_start + local_k;
                        half4 value = half4(0.0h);
                        if (global_m < M) {
                            const ulong offset = ulong(global_m) * K + global_k;
                            if ((offset & 3ul) == 0ul) {
                                value = *reinterpret_cast<device const half4*>(a + offset);
                            } else { // K 非四元素对齐时，部分行的源地址不对齐。
                                value = half4(a[offset], a[offset + 1u], a[offset + 2u], a[offset + 3u]);
                            }
                        }
                        *reinterpret_cast<threadgroup half4*>(stage_a + local_m * BLOCK_K + local_k) = value;
                    }
                    for (uint index = lane; index < BLOCK_M * (BLOCK_K - vector_k); index += SIMD_WIDTH) {
                        const uint local_m = index / (BLOCK_K - vector_k);
                        const uint local_k = vector_k + index % (BLOCK_K - vector_k);
                        const uint global_m = m_start + local_m;
                        const uint global_k = k_start + local_k;
                        stage_a[local_m * BLOCK_K + local_k] = global_m < M && global_k < K ? a[ulong(global_m) * K + global_k] : half(0.0h);
                    }
                    // Scale-A：[M, ceil(K/128)]
                    if (k_block_idx % STAGES_PER_SCALE_GROUP == 0u) {
                        const uint scale_k = k_start / SCALE_GROUP;
                        for (uint local_m = lane * 4u; local_m + 3u < BLOCK_M; local_m += SIMD_WIDTH * 4u) {
                            const uint global_m = m_start + local_m;
                            const float4 value = float4(global_m < M ? sa[ulong(global_m) * group_k + scale_k] : 0.0f,
                                global_m + 1u < M ? sa[ulong(global_m + 1u) * group_k + scale_k] : 0.0f,
                                global_m + 2u < M ? sa[ulong(global_m + 2u) * group_k + scale_k] : 0.0f,
                                global_m + 3u < M ? sa[ulong(global_m + 3u) * group_k + scale_k] : 0.0f);
                            *reinterpret_cast<threadgroup float4*>(stage_sfa + local_m) = value;
                        }
                        for (uint local_m = (BLOCK_M & ~3u) + lane; local_m < BLOCK_M; local_m += SIMD_WIDTH) {
                            const uint global_m = m_start + local_m;
                            stage_sfa[local_m] = global_m < M ? sa[ulong(global_m) * group_k + scale_k] : 0.0f;
                        }
                    }
                } else {
                    // B：[N,K]，保持不转置。
                    for (uint index = lane; index < BLOCK_N * (vector_k/4u); index += SIMD_WIDTH) {
                        const uint local_n = index / (vector_k/4u);
                        const uint local_k = index % (vector_k/4u)*4u;
                        const uint global_n = n_start + local_n;
                        const uint global_k = k_start + local_k;
                        half4 value = half4(0.0h);
                        if (global_n < N) {
                            const ulong offset = ulong(global_n) * K + global_k;
                            if ((offset & 3ul) == 0ul) {
                                value = *reinterpret_cast<device const half4*>(b + offset);
                            } else {
                                value = half4(b[offset], b[offset + 1u], b[offset + 2u], b[offset + 3u]);
                            }
                        }
                        *reinterpret_cast<threadgroup half4*>(stage_b + local_n * BLOCK_K + local_k) = value;
                    }
                    for (uint index = lane; index < BLOCK_N * (BLOCK_K - vector_k); index += SIMD_WIDTH) {
                        const uint local_n = index / (BLOCK_K - vector_k);
                        const uint local_k = vector_k + index % (BLOCK_K - vector_k);
                        const uint global_n = n_start + local_n;
                        const uint global_k = k_start + local_k;
                        stage_b[local_n * BLOCK_K + local_k] = global_n < N && global_k < K ? b[ulong(global_n) * K + global_k] : half(0.0h);
                    }
                }

                // producer 完成 shared-memory 写入后发布 full。
                loader_local_phase ^= 1u;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) { // producer发布full, 所有搬运线程完成A/B/Scale-A写入后，发布就绪序号
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    const uint previous_count = atomic_fetch_add_explicit(loader_count, 1u, memory_order_relaxed);
                    if (previous_count == NUM_LOADER_SIMDGROUPS - 1u) {
                        atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                        atomic_store_explicit(loader_count, 0u, memory_order_relaxed);
                        atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                        atomic_store_explicit(loader_phase, loader_local_phase, memory_order_relaxed);
                    } else {
                        while (atomic_load_explicit(loader_phase, memory_order_relaxed) != loader_local_phase) {
                        }
                    }
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // 只有 loader 0 的 lane 0 发布 full。
                if (sg == 0u && lane == 0u) {
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    atomic_store_explicit(full_phase + stage, phase ^ 1u, memory_order_relaxed);
                }

                ++stage_idx;
                if (stage_idx == NUM_STAGES) {
                    stage_idx = 0u;
                    phase ^= 1u;
                }
            }
        }
        return;
    }

    if (is_math) { // Math Consumer: SIMD-group 2...NUM_MATH_SIMDGROUP
        constexpr uint FRAGMENT_M = 8u;
        constexpr uint FRAGMENT_N = 8u;
        constexpr uint NUM_N_FRAGMENTS = BLOCK_N / FRAGMENT_N;
        constexpr uint SIMDGROUPS_PER_MATH_WG = 4u;
        constexpr uint NUM_MATH_WARPGROUPS = NUM_MATH_SIMDGROUPS / SIMDGROUPS_PER_MATH_WG;

        constexpr uint M_ROWS_PER_WG_PER_WAVE = BLOCK_M / NUM_MATH_WARPGROUPS;
        constexpr uint WAVE_BLOCK_M = M_ROWS_PER_WG_PER_WAVE * NUM_MATH_WARPGROUPS;
        constexpr uint NUM_M_WAVES = BLOCK_M / WAVE_BLOCK_M;

        constexpr uint FRAGMENTS_PER_WG_PER_WAVE = (M_ROWS_PER_WG_PER_WAVE / FRAGMENT_M) * NUM_N_FRAGMENTS;
        constexpr uint FRAGMENTS_PER_SIMDGROUP_PER_WAVE = FRAGMENTS_PER_WG_PER_WAVE / SIMDGROUPS_PER_MATH_WG;
        constexpr uint FRAGMENTS_PER_MATH_GROUP = NUM_M_WAVES * FRAGMENTS_PER_SIMDGROUP_PER_WAVE;

        // math SIMD-group 间同步序号
        uint math_local_phase = 0u;

        // math_sg=0..3 -> math_wg_idx=0, math_sg=4..7 -> math_wg_idx=1
        const uint math_wg_idx = math_sg / SIMDGROUPS_PER_MATH_WG;
        const uint simdgroup_in_math_wg = math_sg % SIMDGROUPS_PER_MATH_WG;

        uint m_block_idx;
        uint n_block_idx;
        int current_iter = -1;
        while (get_next_block(current_iter, P, block_idx, num_blocks, num_1d_blocks_per_group, 
                              num_m_blocks, num_n_blocks, m_block_idx, n_block_idx)) {
            const uint m_start = get_global_idx(m_block_idx, BLOCK_M);
            const uint n_start = get_global_idx(n_block_idx, BLOCK_N);

            math_local_phase ^= 1u;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                const uint previous_count = atomic_fetch_add_explicit(math_count, 1u, memory_order_relaxed);

                if (previous_count == NUM_MATH_SIMDGROUPS - 1u) {
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    atomic_store_explicit(math_count, 0u, memory_order_relaxed);
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    atomic_store_explicit(math_phase, math_local_phase, memory_order_relaxed);
                } else {
                    while (atomic_load_explicit(math_phase, memory_order_relaxed) != math_local_phase) {
                    }
                }
                atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Math线程加载当前输出tile的Scale-B. 第一个math SIMD-group对应tid [32, 64)
            const uint first_scale_n = n_start / SCALE_GROUP;
            const uint scale_n_rows = NUM_SFB_ROWS;
            if (math_sg == 0u) {
                const uint load_thread = lane;
                constexpr uint NUM_SFB_LOAD_THREADS = SIMD_WIDTH;
                for (uint index = load_thread; index < group_k * scale_n_rows / 4u; index += NUM_SFB_LOAD_THREADS) {
                    const uint local_n_scale = index*4u / group_k;
                    const uint local_k_scale = index*4u % group_k;
                    const uint global_n_scale = first_scale_n + local_n_scale;

                    float4 value = float4(0.0f);
                    if (global_n_scale < group_n) {
                        const ulong offset = ulong(global_n_scale) * group_k + local_k_scale;
                        if ((offset & 3ul) == 0ul && offset + 3ul < ulong(group_n) * group_k) {
                            value = *reinterpret_cast<device const float4*>(sb + offset);
                        } else { // 源地址不对齐或跨过最后一个有效 scale：保留有效元素，其余补零。
                            #pragma unroll
                            for (uint j = 0u; j < 4u; ++j) {
                                if (offset + j < ulong(group_n) * group_k) {
                                    value[j] = sb[offset + j];
                                }
                            }
                        }
                    }
                    *reinterpret_cast<threadgroup float4*>(smem_sfb + index * 4u) = value;
                }
                for (uint index = ((group_k * scale_n_rows) & ~3u) + load_thread; index < group_k * scale_n_rows; index += NUM_SFB_LOAD_THREADS) {
                    const uint local_n_scale = index / group_k;
                    const uint local_k_scale = index % group_k;
                    const uint global_n_scale = first_scale_n + local_n_scale;
                    smem_sfb[index] = global_n_scale < group_n ? sb[ulong(global_n_scale) * group_k + local_k_scale] : 0.0f;
                }
            }

            // math-only barrier: 所有 math SIMD-group 完成 Scale-B 加载后继续。
            math_local_phase ^= 1u;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                const uint previous_count = atomic_fetch_add_explicit(math_count, 1u, memory_order_relaxed);

                if (previous_count == NUM_MATH_SIMDGROUPS - 1u) {
                    // 最后一个到达者汇集其他 SIMD-group 发布的写入。
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    // 必须先复位计数器，再发布本轮完成。
                    atomic_store_explicit(math_count, 0u, memory_order_relaxed);
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    atomic_store_explicit(math_phase, math_local_phase, memory_order_relaxed);
                } else {
                    while (atomic_load_explicit(math_phase, memory_order_relaxed) != math_local_phase) {
                    }
                }

                // 观察完成后，获取所有 math SIMD-group 的共享内存写入。
                atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            float2 final_accum[FRAGMENTS_PER_MATH_GROUP];
            float scale_a_group[FRAGMENTS_PER_MATH_GROUP];
            simdgroup_float8x8 acc_frag[FRAGMENTS_PER_MATH_GROUP];
            #pragma unroll
            for (uint f = 0u; f < FRAGMENTS_PER_MATH_GROUP; ++f) {
                final_accum[f] = float2(0.0f);
                scale_a_group[f] = 0.0f;
                acc_frag[f] = simdgroup_float8x8(0.0f);
            }
            float scale_b_group0 = 0.0f;
            float scale_b_group1 = 0.0f;
            const ushort2 lane_offset = morton_order(ushort(lane));

            // K 分组计算。math group 同时完成 MMA、释放 stage、promotion。
            for (uint k_block_idx = 0u; k_block_idx < num_k_blocks; ++k_block_idx) {
                const uint stage = stage_idx;
                const uint k_start = get_global_idx(k_block_idx, BLOCK_K);
                const bool scale_group_start = (k_block_idx % STAGES_PER_SCALE_GROUP) == 0u;
                const bool scale_group_end = ((k_block_idx + 1u) % STAGES_PER_SCALE_GROUP) == 0u || (k_block_idx + 1u == num_k_blocks);

                if (scale_group_start) {
                    const uint scale_k = k_start / SCALE_GROUP;
                    // Scale-B 只在每个 scale group 边界读取一次。
                    scale_b_group0 = smem_sfb[scale_k];
                    scale_b_group1 = scale_b_group0;
                    if constexpr (NUM_SFB_ROWS > 1u) {
                        scale_b_group1 = smem_sfb[group_k + scale_k];
                    }
                }

                if (lane == 0u) { // consumer 等待当前 stage 的 A/B/Scale-A 就绪。
                    while (atomic_load_explicit(full_phase + stage, memory_order_relaxed) == phase) {
                    }
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                threadgroup const half* stage_a = smem_a + stage * BLOCK_M * BLOCK_K;
                threadgroup const half* stage_b = smem_b + stage * BLOCK_N * BLOCK_K;
                threadgroup const float* stage_sfa = smem_sfa + stage * BLOCK_M;
                #pragma unroll
                for (uint f = 0u; f < FRAGMENTS_PER_MATH_GROUP; ++f) {
                    const uint m_wave = f / FRAGMENTS_PER_SIMDGROUP_PER_WAVE;
                    const uint fragment_in_wg = (f % FRAGMENTS_PER_SIMDGROUP_PER_WAVE) * SIMDGROUPS_PER_MATH_WG + simdgroup_in_math_wg;
                    const uint fragment_m = m_wave * WAVE_BLOCK_M + math_wg_idx * M_ROWS_PER_WG_PER_WAVE 
                                            + (fragment_in_wg / NUM_N_FRAGMENTS) * FRAGMENT_M;
                    const uint fragment_n = (fragment_in_wg % NUM_N_FRAGMENTS) * FRAGMENT_N;

                    const uint local_m = fragment_m + uint(lane_offset.y);
                    const uint local_n = fragment_n + uint(lane_offset.x);

                    // 每个 scale group 只从 Scale-A 的第一个 stage 读取一次。
                    if (scale_group_start) {
                        scale_a_group[f] = stage_sfa[local_m];
                    }

                    #pragma unroll
                    for (uint local_k = 0u; local_k < BLOCK_K; local_k += 8u) {
                        simdgroup_half8x8 a_frag;
                        simdgroup_half8x8 b_frag;
                        // A: [BLOCK_M, BLOCK_K] 行主序, 行跨度 = BLOCK_K
                        simdgroup_load(a_frag, stage_a + fragment_m * BLOCK_K + local_k, BLOCK_K);
                        // B 物理存储为 [BLOCK_N, BLOCK_K]（row=n, col=k），
                        // 需要 transpose_matrix=true 读出 K x N 视角的片段
                        simdgroup_load(b_frag, stage_b + fragment_n * BLOCK_K + local_k,
                                       BLOCK_K, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(acc_frag[f], a_frag, b_frag, acc_frag[f]);
                    }
                }

                // 当前 stage 完成后仍然必须报告 empty。
                simdgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0u) {
                    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                    const uint previous_count = atomic_fetch_add_explicit(empty_count + stage, 1u, memory_order_relaxed);
                    if (previous_count == NUM_MATH_SIMDGROUPS - 1u) {
                        atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
                        atomic_store_explicit(empty_phase + stage, phase ^ 1u, memory_order_relaxed);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                // 只在 scale group 结束时进行一次 store/barrier/load/promotion。
                if (scale_group_end) {
                    threadgroup float* acc_scratch = mma_scratch + math_sg * 64u;

                    #pragma unroll
                    for (uint f = 0u; f < FRAGMENTS_PER_MATH_GROUP; ++f) {
                        simdgroup_store(acc_frag[f], acc_scratch, 8, 0, false);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        const uint scratch_row = uint(lane_offset.y);
                        const uint scratch_col = uint(lane_offset.x);
                        const float2 accum = float2(acc_scratch[scratch_row * 8u + scratch_col], acc_scratch[scratch_row * 8u + scratch_col + 1u]);
                        const uint m_wave = f / FRAGMENTS_PER_SIMDGROUP_PER_WAVE;
                        const uint fragment_in_wg = (f % FRAGMENTS_PER_SIMDGROUP_PER_WAVE) * SIMDGROUPS_PER_MATH_WG + simdgroup_in_math_wg;
                        const uint fragment_m = m_wave * WAVE_BLOCK_M + math_wg_idx * M_ROWS_PER_WG_PER_WAVE
                                                 + (fragment_in_wg / NUM_N_FRAGMENTS) * FRAGMENT_M;
                        const uint fragment_n = (fragment_in_wg % NUM_N_FRAGMENTS) * FRAGMENT_N;
                        const uint local_n = fragment_n + uint(lane_offset.x);
                        const uint scale_b_row0 = (n_start + local_n) / SCALE_GROUP - first_scale_n;
                        const uint scale_b_row1 = (n_start + local_n + 1u) / SCALE_GROUP - first_scale_n;
                        const float scale_b0 = scale_b_row0 == 0u ? scale_b_group0 : scale_b_group1;
                        const float scale_b1 = scale_b_row1 == 0u ? scale_b_group0 : scale_b_group1;
                        final_accum[f] += accum * float2(scale_a_group[f] * scale_b0, scale_a_group[f] * scale_b1);
                        acc_frag[f] = simdgroup_float8x8(0.0f);
                    }
                }
                ++stage_idx;
                if (stage_idx == NUM_STAGES) {
                    stage_idx = 0u;
                    phase ^= 1u;
                }
            }
            // 当前 Metal 路径保持 FP32 输出。这里对应 DeepGEMM 的输出 store 阶段。
            for (uint f = 0u; f < FRAGMENTS_PER_MATH_GROUP; ++f) {
                const uint m_wave = f / FRAGMENTS_PER_SIMDGROUP_PER_WAVE;
                const uint fragment_in_wg = (f % FRAGMENTS_PER_SIMDGROUP_PER_WAVE) * SIMDGROUPS_PER_MATH_WG + simdgroup_in_math_wg;
                const uint fragment_m = m_wave * WAVE_BLOCK_M + math_wg_idx * M_ROWS_PER_WG_PER_WAVE 
                                        + (fragment_in_wg / NUM_N_FRAGMENTS) * FRAGMENT_M;
                const uint fragment_n = (fragment_in_wg % NUM_N_FRAGMENTS) * FRAGMENT_N;

                const uint local_m = fragment_m + uint(lane_offset.y);
                const uint local_n = fragment_n + uint(lane_offset.x);
                const uint global_m = m_start + local_m;
                const uint global_n = n_start + local_n;

                if (global_m >= M) continue;

                const ulong global_idx = ulong(global_m) * ulong(N) + ulong(global_n);
                const float2 value = final_accum[f];

                if (global_n < (N & ~1u)) {
                    float2 out_value = value;
                    if constexpr (WITH_ACCUMULATION) {
                        out_value += float2(*reinterpret_cast<device const packed_float2*>(c + global_idx));
                    }
                    *reinterpret_cast<device packed_float2*>(d + global_idx) = packed_float2(out_value);
                }

                if ((N & 1u) != 0u && global_n == N - 1u) {
                    float out_value = value[0];
                    if constexpr (WITH_ACCUMULATION) {
                        out_value += c[global_idx];
                    }
                    d[global_idx] = out_value;
                }
            }
        }
    }
}

METAL_FUNC constexpr uint align16(uint x) {
    return (x + 15u) & ~15u;
}

template<uint BM, uint BN, uint BK, uint STAGES, uint GROUP_K, bool ACC>
constexpr uint fp16_gemm_smem_bytes() {
    constexpr uint sfb_rows = (BK % BN == 0u) ? 1u : 2u;
    constexpr uint stage_bytes = align16(BM * BK * sizeof(half)) + align16(BN * BK * sizeof(half)) + align16(BM * sizeof(float));
    constexpr uint sfb_bytes = align16(GROUP_K * sfb_rows * sizeof(float));
    constexpr uint sync_bytes = align16((3u * STAGES + 4u) * sizeof(atomic_uint));
    constexpr uint num_math_simdgroups = BM*BN <= 64u*64u ? 4u : 16u;
    constexpr uint mma_scratch_bytes = num_math_simdgroups * 8u * 8u * sizeof(float);
    return stage_bytes * STAGES + sfb_bytes + sync_bytes + align16(mma_scratch_bytes);
}

#define DEFINE_FP16_GEMM(BM, BN, BK, GROUP_K, STAGES, ACC, NAME) \
kernel void NAME( \
    device const half* a [[buffer(0)]], \
    device const half* b [[buffer(1)]], \
    device const float* sa [[buffer(2)]], \
    device const float* sb [[buffer(3)]], \
    device float* d [[buffer(4)]], \
    device const float* c [[buffer(5)]], \
    constant uint& M [[buffer(6)]], \
    constant uint& N [[buffer(7)]], \
    constant uint& K [[buffer(8)]], \
    constant uint& num_m_blocks [[buffer(9)]], \
    constant uint& num_n_blocks [[buffer(10)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    uint3 tg_grid [[threadgroups_per_grid]], \
    uint3 tid3 [[thread_position_in_threadgroup]]) \
{ \
    constexpr uint SMEM_BYTES = fp16_gemm_smem_bytes<BM, BN, BK, STAGES, GROUP_K, ACC>(); \
    threadgroup uchar smem_storage[SMEM_BYTES]; \
    fp16_gemm_impl<BM, BN, BK, STAGES, ACC>( \
        a, b, sa, sb, d, c, M, N, K, \
        num_m_blocks, num_n_blocks, tg, tg_grid, tid3.x, \
        reinterpret_cast<threadgroup uchar*>(smem_storage)); \
}

DEFINE_FP16_GEMM(32, 32, 32, 4, 2, false, fp16_gemm_bm32_bn32_bk32_gk4_s2_a0);
DEFINE_FP16_GEMM(32, 32, 32, 4, 2, true, fp16_gemm_bm32_bn32_bk32_gk4_s2_a1);
DEFINE_FP16_GEMM(32, 32, 64, 4, 2, false, fp16_gemm_bm32_bn32_bk64_gk4_s2_a0);
DEFINE_FP16_GEMM(32, 32, 64, 4, 2, true, fp16_gemm_bm32_bn32_bk64_gk4_s2_a1);
DEFINE_FP16_GEMM(32, 32, 32, 256, 2, false, fp16_gemm_bm32_bn32_bk32_gk256_s2_a0);
DEFINE_FP16_GEMM(32, 32, 32, 256, 2, true, fp16_gemm_bm32_bn32_bk32_gk256_s2_a1);
DEFINE_FP16_GEMM(32, 32, 64, 256, 2, false, fp16_gemm_bm32_bn32_bk64_gk256_s2_a0);
DEFINE_FP16_GEMM(32, 32, 64, 256, 2, true, fp16_gemm_bm32_bn32_bk64_gk256_s2_a1);
DEFINE_FP16_GEMM(64, 48, 64, 8, 2, false, fp16_gemm_bm64_bn48_bk64_gk8_s2_a0);
DEFINE_FP16_GEMM(64, 48, 64, 8, 2, true, fp16_gemm_bm64_bn48_bk64_gk8_s2_a1);
DEFINE_FP16_GEMM(64, 128, 32, 256, 2, false, fp16_gemm_bm64_bn128_bk32_gk256_s2_a0);
DEFINE_FP16_GEMM(64, 128, 32, 256, 2, true, fp16_gemm_bm64_bn128_bk32_gk256_s2_a1);
DEFINE_FP16_GEMM(128, 80, 32, 4, 2, false, fp16_gemm_bm128_bn80_bk32_gk4_s2_a0);
DEFINE_FP16_GEMM(128, 80, 32, 4, 2, true, fp16_gemm_bm128_bn80_bk32_gk4_s2_a1);



