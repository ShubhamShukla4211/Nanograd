import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad
        out._backward = _backward
        out._prev = {self, other}
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad if self.grad is not None else other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad if other.grad is not None else self.data * out.grad
        out._backward = _backward
        out._prev = {self, other}
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()
