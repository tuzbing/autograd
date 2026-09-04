import numpy as np
from tensor import Tensor
from storage import CPUAllocator
from runtime import Runtime

def test_linear_regression_with_layout_ops():
    Runtime._default_runtime = None
    rt = Runtime.get_default()
    allocator = CPUAllocator()
    np.random.seed(42)
    x_data = np.random.randn(4, 3).astype(np.float32)
    true_w = np.array([[2.0], [-1.0], [3.0]], dtype=np.float32)
    true_b = 1.0
    y_data = x_data @ true_w + true_b
    x = Tensor(shape=(4, 3), dtype=np.float32, requires_grad=False, name="x")
    x.storage = allocator.alloc(x.shape, x.dtype)
    x.storage.data[:] = x_data
    neg_y = Tensor(shape=(4, 1), dtype=np.float32, requires_grad=False, name="neg_y")
    neg_y.storage = allocator.alloc(neg_y.shape, neg_y.dtype)
    neg_y.storage.data[:] = -y_data
    w = Tensor(shape=(3, 1), dtype=np.float32, requires_grad=True, name="w")
    w.storage = allocator.alloc(w.shape, w.dtype)
    w.storage.data[:] = np.random.randn(3, 1).astype(np.float32)
    b = Tensor(shape=(4, 1), dtype=np.float32, requires_grad=True, name="b")
    b.storage = allocator.alloc(b.shape, b.dtype)
    b.storage.data[:] = np.random.randn(4, 1).astype(np.float32)
    w_t = w.transpose(0, 1)
    w_tc = w_t.contiguous()
    w_col = w_tc.transpose(0, 1).contiguous()
    w_mat = w_col.reshape(3, 1)
    w_perm = w_mat.permute(1, 0).contiguous().permute(1, 0).contiguous()
    out1 = x @ w_perm
    out2 = out1 + b
    diff = out2 + neg_y
    diff_flat = diff.reshape(4, 1)
    loss = diff_flat * diff_flat
    lr = 0.01
    epochs = 100
    print("开始训练（含 layout 算子）...")
    for step in range(epochs):
        loss.backward()
        w.storage.data[:] -= lr * w.grad.storage.data
        b.storage.data[:] -= lr * b.grad.storage.data
        if step % 10 == 0 or step == epochs - 1:
            print(f"Step {step:3d} | Loss: {np.mean(loss.storage.data):.6f}")
    print("\n训练完成！")
    print("真实 w:", true_w.flatten())
    print("学习 w:", w.storage.data.flatten())
    print("真实 b:", true_b)
    print("学习 b(均值):", np.mean(b.storage.data))
    w_np = w.storage.data.copy()
    b_np = b.storage.data.copy()
    pred = x_data @ w_np + b_np
    ref_loss = np.mean((pred - y_data) ** 2)
    print("NumPy 参考 loss:", ref_loss)

def test_layout_ops_grad_sanity():
    Runtime._default_runtime = None
    allocator = CPUAllocator()
    x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = Tensor(shape=(2, 3), dtype=np.float32, requires_grad=True)
    x.storage = allocator.alloc(x.shape, x.dtype)
    x.storage.data[:] = x_np
    # permute(1,0) + square sum
    y = x.permute(1, 0).contiguous()
    z = y.reshape(3, 2)
    loss = (z * z).reshape(6,) 
    loss_scalar = loss.reshape(6, 1) 
    print("layout grad sanity: 请在你修好 runtime/contiguous 后取消注释并对比数值梯度")
if __name__ == "__main__":
    test_linear_regression_with_layout_ops()