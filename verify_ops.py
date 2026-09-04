import time
import numpy as np
from tensor import Tensor
from runtime import Runtime
from op import GELUOp, SoftmaxOp, CrossEntropyLossOp, GatherOp, ScaledDotProductAttenOp, FusedLinearCrossEntropyOp
from op import FusedMLPOp, FusedQKVProjOp, FusedLinearAddOp, FusedLayerNormFlashAttentionOp
from module import LayerNorm
from optimizer import AdamW
from utils import tensor_to_numpy

import gc

# B, T, C = 4, 128, 768
B, T, C = 2, 128, 256
# B, T, C = 4, 512, 384
V = 1000
NH = 4
HS = C // NH
def get_test_case(name, device, cache):
    if name == "MatMul (Linear)":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, C).astype(np.float32), np.random.randn(C, C).astype(np.float32)]
        x_np, w_np = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        w = Tensor((C, C), np.float32, device=device, requires_grad=True, init_val=w_np)
        return [x, w], lambda: x @ w
    elif name == "MatMul (Attn)":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32), np.random.randn(B, NH, HS, T).astype(np.float32)]
        q_np, k_np = cache[name]
        q = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=q_np)
        k = Tensor((B, NH, HS, T), np.float32, device=device, requires_grad=True, init_val=k_np)
        return [q, k], lambda: q @ k
    elif name == "LayerNorm":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, C).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        ln = LayerNorm(C, np.float32, device=device)
        return [x, ln.gamma, ln.beta], lambda: ln(x)
    elif name == "GELU":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, 4*C).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B*T, 4*C), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: GELUOp()(x)
    elif name == "Softmax":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, T).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, T), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: SoftmaxOp(dim=-1)(x)
    elif name == "CrossEntropy":
        if name not in cache:
            cache[name] = [np.random.randn(B, T, V).astype(np.float32), np.random.randint(0, V, (B, T)).astype(np.int32)]
        logits_np, labels_np = cache[name]
        logits = Tensor((B, T, V), np.float32, device=device, requires_grad=True, init_val=logits_np)
        labels = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=labels_np)
        return [logits], lambda: CrossEntropyLossOp()(logits, labels)
    elif name == "Add":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, C).astype(np.float32), np.random.randn(B*T, C).astype(np.float32)]
        x_np, y_np = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        y = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=y_np)
        return [x, y], lambda: x + y
    elif name == "Mul":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, C).astype(np.float32), np.random.randn(B*T, C).astype(np.float32)]
        x_np, y_np = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        y = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=y_np)
        return [x, y], lambda: x * y
    elif name == "Exp":
        if name not in cache:
            cache[name] = [np.random.randn(B*T, C).astype(np.float32) * 0.1] 
        x_np, = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.exp()
    elif name == "Log":
        if name not in cache:
            cache[name] = [np.random.uniform(0.1, 1.0, (B*T, C)).astype(np.float32)] 
        x_np, = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.log()
    elif name == "Rsqrt":
        if name not in cache:
            cache[name] = [np.random.uniform(0.1, 1.0, (B*T, C)).astype(np.float32)] 
        x_np, = cache[name]
        x = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.rsqrt()
    elif name == "Gather":
        if name not in cache:
            cache[name] = [np.random.randint(0, V, (B, T)).astype(np.int32), np.random.randn(V, C).astype(np.float32)]
        idx_np, w_np = cache[name]
        idx = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=idx_np)
        w = Tensor((V, C), np.float32, device=device, requires_grad=True, init_val=w_np)
        return [idx, w], lambda: GatherOp()(idx, w)
    elif name == "Transpose":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.transpose(1, 2)
    elif name == "Contiguous":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.transpose(1, 2).contiguous()
    elif name == "View":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.view(B, NH*T*HS)
    elif name == "Reshape":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.reshape(B, NH*T*HS)
    elif name == "Permute":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.permute(0, 2, 1, 3)
    elif name == "ScaledDotProductAttention":
        if name not in cache:
            cache[name] = [
                np.random.randn(B, NH, T, HS).astype(np.float32),
                np.random.randn(B, NH, T, HS).astype(np.float32),
                np.random.randn(B, NH, T, HS).astype(np.float32)
            ]
        q_np, k_np, v_np = cache[name]
        q = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=q_np)
        k = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=k_np)
        v = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=v_np)
        return [q, k, v], lambda: ScaledDotProductAttenOp(is_causal=True)(q, k, v)
    elif name == "FusedLinearCrossEntropy":
        if name not in cache:
            cache[name] = [
                np.random.randn(B, T, C).astype(np.float32),
                np.random.randn(V, C).astype(np.float32),
                np.random.randint(0, V, (B, T)).astype(np.int32)
            ]
        x_np, w_np, targets_np = cache[name]
        
        x = Tensor((B, T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        w = Tensor((V, C), np.float32, device=device, requires_grad=True, init_val=w_np)
        targets = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=targets_np)
        
        return [x, w, targets], lambda: FusedLinearCrossEntropyOp()(x, w, targets)
    elif name == "FusedMLPOp":
        C4 = 4 * C
        if name not in cache:
            cache[name] = [
                np.random.randn(B, T, C).astype(np.float32),
                np.random.randn(C, C4).astype(np.float32),
                np.random.randn(C4).astype(np.float32),
                np.random.randn(C4, C).astype(np.float32),
                np.random.randn(C).astype(np.float32)
            ]
        x_np, w1_np, b1_np, w2_np, b2_np = cache[name]
        
        x = Tensor((B, T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        w1 = Tensor((C, C4), np.float32, device=device, requires_grad=True, init_val=w1_np)
        b1 = Tensor((C4,), np.float32, device=device, requires_grad=True, init_val=b1_np)
        w2 = Tensor((C4, C), np.float32, device=device, requires_grad=True, init_val=w2_np)
        b2 = Tensor((C,), np.float32, device=device, requires_grad=True, init_val=b2_np)
        
        from op import FusedMLPOp
        return [x, w1, b1, w2, b2], lambda: FusedMLPOp()(x, w1, b1, w2, b2)
    elif name == "FusedQKVProj":
        n_head = 12
        C_test = 288
        if C_test % n_head != 0:
            C_test = (C_test // n_head + 1) * n_head
            
        if name not in cache:
            cache[name] = [
                np.random.randn(B, T, C_test).astype(np.float32),
                np.random.randn(C_test, C_test).astype(np.float32),
                np.random.randn(C_test).astype(np.float32),
                np.random.randn(C_test, C_test).astype(np.float32),
                np.random.randn(C_test).astype(np.float32),
                np.random.randn(C_test, C_test).astype(np.float32),
                np.random.randn(C_test).astype(np.float32)
            ]
        x_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np = cache[name]
        
        x = Tensor((B, T, C_test), np.float32, device=device, requires_grad=True, init_val=x_np)
        wq = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wq_np)
        bq = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bq_np)
        wk = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wk_np)
        bk = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bk_np)
        wv = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wv_np)
        bv = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bv_np)
        
        from op import FusedQKVProjOp
        def func():
            q, k, v = FusedQKVProjOp(n_head)(x, wq, bq, wk, bk, wv, bv)
            return q + k + v 
            
        return [x, wq, bq, wk, bk, wv, bv], func

    elif name == "AttnOutContiguous":
        if name not in cache:
            cache[name] = [np.random.randn(B, NH, T, HS).astype(np.float32)]
        x_np, = cache[name]
        x = Tensor((B, NH, T, HS), np.float32, device=device, requires_grad=True, init_val=x_np)
        return [x], lambda: x.attn_out_contiguous()

    elif name == "FusedLayerNormMLP":
        C4 = 4 * C
        if name not in cache:
            cache[name] = [
                np.random.randn(B, T, C).astype(np.float32),          # x
                np.random.randn(C).astype(np.float32),                # ln_w
                np.random.randn(C).astype(np.float32),                # ln_b
                np.random.randn(C, C4).astype(np.float32),            # w1
                np.random.randn(C4).astype(np.float32),               # b1
                np.random.randn(C4, C).astype(np.float32),            # w2
                np.random.randn(C).astype(np.float32)                 # b2
            ]
        x_np, ln_w_np, ln_b_np, w1_np, b1_np, w2_np, b2_np = cache[name]
        
        x = Tensor((B, T, C), np.float32, device=device, requires_grad=True, init_val=x_np)
        ln_w = Tensor((C,), np.float32, device=device, requires_grad=True, init_val=ln_w_np)
        ln_b = Tensor((C,), np.float32, device=device, requires_grad=True, init_val=ln_b_np)
        w1 = Tensor((C, C4), np.float32, device=device, requires_grad=True, init_val=w1_np)
        b1 = Tensor((C4,), np.float32, device=device, requires_grad=True, init_val=b1_np)
        w2 = Tensor((C4, C), np.float32, device=device, requires_grad=True, init_val=w2_np)
        b2 = Tensor((C,), np.float32, device=device, requires_grad=True, init_val=b2_np)
        
        from op import FusedLayerNormMLPOp
        return [x, ln_w, ln_b, w1, b1, w2, b2], lambda: FusedLayerNormMLPOp()(x, ln_w, ln_b, w1, b1, w2, b2)

    elif name == "FlashAttention":
        # 增大计算量：B=4, 序列长度 T_test=512, 12 个头，每个头维度 64，总维度 C_test=768
        B_test = 4
        T_test = 512
        NH_test = 12
        C_test = NH_test * 64 
        scale = 1.0 / np.sqrt(C_test)
        if name not in cache:
            cache[name] = [
                np.random.randn(B_test, T_test, C_test).astype(np.float32),          # x
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wq
                np.random.randn(C_test).astype(np.float32),                          # bq
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wk
                np.random.randn(C_test).astype(np.float32),                          # bk
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wv
                np.random.randn(C_test).astype(np.float32)                           # bv
            ]
        x_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np = cache[name]
        
        x = Tensor((B_test, T_test, C_test), np.float32, device=device, requires_grad=True, init_val=x_np)
        wq = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wq_np)
        bq = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bq_np)
        wk = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wk_np)
        bk = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bk_np)
        wv = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wv_np)
        bv = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bv_np)
        
        from op import FlashAttentionOp
        return [x, wq, bq, wk, bk, wv, bv], lambda: FlashAttentionOp(n_head=NH_test, is_causal=True)(x, wq, bq, wk, bk, wv, bv)

    elif name == "FusedLayerNormFlashAttention":
        # 增大计算量：B=4, 序列长度 T_test=512, 12 个头，每个头维度 64，总维度 C_test=768
        B_test = 4
        T_test = 512
        NH_test = 12
        C_test = NH_test * 64 
        scale = 1.0 / np.sqrt(C_test)
        if name not in cache:
            cache[name] = [
                np.random.randn(B_test, T_test, C_test).astype(np.float32),          # x (未归一化)
                np.random.randn(C_test).astype(np.float32),                          # ln_w
                np.random.randn(C_test).astype(np.float32),                          # ln_b
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wq
                np.random.randn(C_test).astype(np.float32),                          # bq
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wk
                np.random.randn(C_test).astype(np.float32),                          # bk
                (np.random.randn(C_test, C_test) * scale).astype(np.float32),                  # wv
                np.random.randn(C_test).astype(np.float32)                           # bv
            ]
        x_np, ln_w_np, ln_b_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np = cache[name]
        
        x = Tensor((B_test, T_test, C_test), np.float32, device=device, requires_grad=True, init_val=x_np)
        ln_w = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=ln_w_np)
        ln_b = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=ln_b_np)
        wq = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wq_np)
        bq = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bq_np)
        wk = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wk_np)
        bk = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bk_np)
        wv = Tensor((C_test, C_test), np.float32, device=device, requires_grad=True, init_val=wv_np)
        bv = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=bv_np)
        
        return [x, ln_w, ln_b, wq, bq, wk, bk, wv, bv], lambda: FusedLayerNormFlashAttentionOp(n_head=NH_test, eps=1e-5, is_causal=True)(x, ln_w, ln_b, wq, bq, wk, bk, wv, bv)

    elif name == "FusedLinearAdd":
        B_test = 4
        T_test = 512
        C_test = 768
        K_test = 768  # Attention 输出的维度
        scale = 1.0 / np.sqrt(K_test)
        
        if name not in cache:
            cache[name] = [
                np.random.randn(B_test, T_test, C_test).astype(np.float32),          # x (残差输入)
                np.random.randn(B_test, T_test, K_test).astype(np.float32),          # act (Attention输出)
                (np.random.randn(K_test, C_test) * scale).astype(np.float32),        # w (投影权重)
                np.random.randn(C_test).astype(np.float32),                          # b (投影偏置)
            ]
        x_np, act_np, w_np, b_np = cache[name]
        
        x = Tensor((B_test, T_test, C_test), np.float32, device=device, requires_grad=True, init_val=x_np)
        act = Tensor((B_test, T_test, K_test), np.float32, device=device, requires_grad=True, init_val=act_np)
        w = Tensor((K_test, C_test), np.float32, device=device, requires_grad=True, init_val=w_np)
        b = Tensor((C_test,), np.float32, device=device, requires_grad=True, init_val=b_np)
        
        return [x, act, w, b], lambda: FusedLinearAddOp()(x, act, w, b)

def get_tensor_data(t):
    if t.device == 'cpu':
        from utils import tensor_to_numpy
        return tensor_to_numpy(t).copy()
    else:
        rt = Runtime.get_default()
        
        dummy = Tensor(t.storage.shape, t.dtype, device=t.device)
        dummy.storage = t.storage
        
        buf = rt.backend.numpy_view(dummy)
        
        flat = buf.reshape(-1)
        if t.offset:
            flat = flat[t.offset:]
            
        if t.is_contiguous:
            return flat[:t.numel].reshape(t.shape).copy()
            
        itemsize = np.dtype(t.dtype).itemsize
        byte_strides = tuple(s * itemsize for s in t.strides)
        return np.lib.stride_tricks.as_strided(flat, shape=t.shape, strides=byte_strides).copy()

def run_and_time(inputs, op_func, iters=1):
    rt = Runtime.get_default()
    out = op_func()
    
    out.backward()
    rt.backend.synchronize()
    
    out_data = get_tensor_data(out)
    grads_data = []
    for t in inputs:
        if t.requires_grad and t.grad is not None:
            grads_data.append(get_tensor_data(t.grad))
        else:
            grads_data.append(None)
    
    gc.collect()

    t0 = time.perf_counter()
    for _ in range(iters):
        rt.forward()
    rt.backend.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    gc.collect()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        rt.backward_graph()
    rt.backend.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000.0 / iters
    
    return out_data, grads_data, fwd_ms, bwd_ms
def test_optimizer(cache, iters=10):
    name = "AdamW"
    if name not in cache:
        cache[name] = [np.random.randn(B*T, C).astype(np.float32), np.random.randn(B*T, C).astype(np.float32)]
    p_np, g_np = cache[name]
    def run_opt(device):
        Runtime._default_runtime = Runtime(device=device)
        rt = Runtime.get_default()
        
        p = Tensor((B*T, C), np.float32, device=device, requires_grad=True, init_val=p_np)
        # rt.allocate(p, with_grad=True)
        p.storage = rt.allocator.alloc(p.shape, p.dtype)
        if device == 'cpu':
            p.storage.data[:] = p_np
        else:
            rt.backend.copy_from_host(p, p_np)

        p.grad = Tensor(p.shape, p.dtype, device=device, requires_grad=False)
        p.grad.storage = rt.allocator.alloc(p.grad.shape, p.grad.dtype)
        rt.backend.fill(p.grad, 0)
        if device == 'cpu':
            p.grad.storage.data[:] = g_np
        else:
            rt.backend.copy_from_host(p.grad, g_np)
            
        opt = AdamW([p], lr=1e-3, weight_decay=0.01)
        
        # Correctness step
        opt.step()
        rt.backend.synchronize()
        p_out = get_tensor_data(p)
        
        # Perf steps
        t0 = time.perf_counter()
        for _ in range(iters):
            opt.step()
        rt.backend.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0 / iters
        
        return p_out, step_ms
    cpu_p_out, cpu_step_ms = run_opt('cpu')
    metal_p_out, metal_step_ms = run_opt('mps')
    
    is_correct = True
    if not np.allclose(cpu_p_out, metal_p_out, atol=1e-3, rtol=1e-3):
        is_correct = False
        max_diff = np.max(np.abs(cpu_p_out - metal_p_out))
        print(f"\n[FAIL] {name} output mismatch! Max diff: {max_diff:.6f}")
        
    speedup = cpu_step_ms / metal_step_ms if metal_step_ms > 0 else 0
    correct_str = "PASS" if is_correct else "FAIL"
    
    print(f"{name:<18s} | {'-':>8s} | {'-':>9s} | {cpu_step_ms:6.2f}ms | {metal_step_ms:7.2f}ms | {'-':>4s} / {speedup:4.1f}x | {correct_str}")
def main():
    print(f"{'Operation':<18s} | {'CPU Fwd':<8s} | {'Metal Fwd':<9s} | {'CPU Bwd':<8s} | {'Metal Bwd':<9s} | {'Speedup (Fwd/Bwd)':<17s} | {'Correctness'}")
    print("-" * 106)
    
    op_names = [
        "MatMul (Linear)", "MatMul (Attn)", "LayerNorm", "GELU", "Softmax", 
        "CrossEntropy", "Add", "Mul", "Exp", "Log", "Rsqrt", "Gather", 
        "View", "Reshape", "Permute", "Transpose", "Contiguous", "ScaledDotProductAttention", 
        "FusedLinearCrossEntropy", 'FusedMLPOp', 'FusedQKVProj', 'AttnOutContiguous',
        'FusedLayerNormMLP', 'FlashAttention', 'FusedLayerNormFlashAttention', 'FusedLinearAdd',
    ]
    cache = {}
    
    for name in op_names:
        if name != 'FusedLayerNormMLP':
            continue
        Runtime._default_runtime = Runtime(device='cpu')
        cpu_inputs, cpu_func = get_test_case(name, 'cpu', cache)
        cpu_out_data, cpu_grads_data, cpu_fwd, cpu_bwd = run_and_time(cpu_inputs, cpu_func)

        del cpu_inputs
        del cpu_func
        Runtime._default_runtime = None
        gc.collect()
        
        Runtime._default_runtime = Runtime(device='mps')
        metal_inputs, metal_func = get_test_case(name, 'mps', cache)
        metal_out_data, metal_grads_data, metal_fwd, metal_bwd = run_and_time(metal_inputs, metal_func)

        del metal_inputs
        del metal_func
        gc.collect()
        
        # Check Correctness
        is_correct = True
        
        if not np.allclose(cpu_out_data, metal_out_data, atol=1e-3, rtol=1e-3):
            is_correct = False
            max_diff = np.max(np.abs(cpu_out_data - metal_out_data))
            print(f"\n[FAIL] {name} Forward output mismatch! Max diff: {max_diff:.6f}")
        
        for i, (c_grad, m_grad) in enumerate(zip(cpu_grads_data, metal_grads_data)):
            if c_grad is not None and m_grad is not None:
                if not np.allclose(c_grad, m_grad, atol=1e-2, rtol=1e-2):
                    is_correct = False
                    diff = np.abs(c_grad - m_grad)
                    max_diff = np.max(diff)
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    print(f"\n[FAIL] {name} Backward grad mismatch on input {i}! Max diff: {max_diff:.6f} at {max_idx}")
                    if name == "FusedLinearAdd" and i == 1:
                        for b in range(c_grad.shape[0]):
                            diff_b = np.abs(c_grad[b] - m_grad[b])
                            print(f"Batch {b} max diff: {np.max(diff_b):.6f}")
        
        correct_str = "PASS" if is_correct else "FAIL"
        
        speedup_fwd = cpu_fwd / metal_fwd if metal_fwd > 0 else 0
        speedup_bwd = cpu_bwd / metal_bwd if metal_bwd > 0 else 0
        
        print(f"{name:<18s} | {cpu_fwd:6.2f}ms | {metal_fwd:7.2f}ms | {cpu_bwd:6.2f}ms | {metal_bwd:7.2f}ms | {speedup_fwd:4.1f}x / {speedup_bwd:4.1f}x | {correct_str}")

    test_optimizer(cache)
if __name__ == "__main__":
    main()