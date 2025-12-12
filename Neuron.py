import numpy as np

class Neuron:
    def __init__(self, input_dim, activation=None):
        self.w = np.random.randn(input_dim) * 0.1
        self.b = 0.0
        self.activation = activation

    def activate(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        if self.activation == "tanh":
            return np.tanh(z)
        return z

    def activate_deriv(self, z):
        if self.activation == "relu":
            return (z > 0).astype(float)
        if self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        if self.activation == "tanh":
            return 1 - np.tanh(z)**2
        return 1

    def forward(self, x):
        self.x = np.array(x).flatten()
        self.z = self.x.dot(self.w) + self.b
        self.a = self.activate(self.z)
        return self.a

    def backward(self, grad_output, lr):
        
        x_arr = np.array(self.x).flatten() if hasattr(self, 'x') else None
        
        if x_arr is None:
            x_arr = np.zeros_like(self.w)
        z = x_arr.dot(self.w) + self.b
        grad_z = grad_output * self.activate_deriv(z)
        grad_w = x_arr * grad_z
        grad_b = grad_z
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        return grad_z * self.w
