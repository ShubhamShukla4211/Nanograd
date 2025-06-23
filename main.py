from engine.tensor import Tensor
from nn.layers import Linear, ReLU
from optim.sgd import SGD
import numpy as np

def generate_data(n=100):
    X = np.random.randn(2, n)
    Y = (X[0, :] * X[1, :] > 0).astype(np.float32) * 2 - 1
    return Tensor(X), Tensor(Y)

def main():
    X, Y = generate_data()
    
    model = [Linear(2, 4), ReLU(), Linear(4, 1)]
    params = []
    for layer in model:
        if isinstance(layer, Linear):
            params += [layer.W, layer.b]
    
    optimizer = SGD(params, lr=0.1)
    
    for epoch in range(1000):
        out = X
        for layer in model:
            out = layer(out)
        
        loss = ((out - Y)**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

if __name__ == "__main__":
    main()
