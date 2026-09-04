import numpy as np
from tensor import Tensor
from op import AddOp, MulOp, MatMulOp, OpNode
from storage import CPUAllocator
from runtime import Runtime
def test_linear_regression():
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

    out1 = x @ w
    out2 = out1 + b
    diff = out2 + neg_y
    loss = diff * diff

    lr = 0.01
    epochs = 100
    print("开始训练...")
    for step in range(epochs):
        loss.backward()
        w.storage.data[:] -= lr * w.grad.storage.data
        b.storage.data[:] -= lr * b.grad.storage.data
        if step % 10 == 0 or step == epochs - 1:
            current_loss = np.mean(loss.storage.data)
            print(f"Step {step:3d} | Loss: {current_loss:.6f}")
    print("\n训练完成！检查拟合结果：")
    print("-" * 30)
    print("真实权重 w:\n", true_w.flatten())
    print("学习到的 w:\n", w.storage.data.flatten())
    print("-" * 30)
    print("真实偏置 b: ", true_b)
    print("学习到的 b (平均): ", np.mean(b.storage.data))
if __name__ == "__main__":
    test_linear_regression()