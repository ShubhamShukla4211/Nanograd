from engine.tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(out_features, in_features) * 0.01, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x):
        out = (self.W @ x) + self.b
        return out

class ReLU:
    def __call__(self, x):
        out = Tensor(np.maximum(0, x.data))
        
        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (x.data > 0) * out.grad if x.grad is not None else (x.data > 0) * out.grad
        out._backward = _backward
        out._prev = {x}
        out.requires_grad = x.requires_grad
        return out
