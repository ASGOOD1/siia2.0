import numpy as np


class Layer:
    

    def __init__(self, input_dim, num_neurons, activation=None):
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.W = np.random.randn(input_dim, num_neurons) * 0.1
        self.b = np.zeros((1, num_neurons))
        self.activation = activation

    def _activate(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        if self.activation == "tanh":
            return np.tanh(z)
        return z

    def _activate_deriv(self, z):
        if self.activation == "relu":
            return (z > 0).astype(float)
        if self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        if self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        return np.ones_like(z)

    def forward(self, X):
        # X: (batch, input_dim)
        self.last_input = X
        self.last_z = X.dot(self.W) + self.b  # (batch, num_neurons)
        self.last_a = self._activate(self.last_z)
        return self.last_a

    def backward(self, grad_outputs, lr):
        # grad_outputs: (batch, num_neurons)
        batch_size = grad_outputs.shape[0]
        grad_z = grad_outputs * self._activate_deriv(self.last_z)
        # Gradient w.r.t. weights: (input_dim, num_neurons)
        grad_W = (self.last_input.T.dot(grad_z)) / batch_size
        grad_b = np.mean(grad_z, axis=0, keepdims=True)
        # Gradient to pass to previous layer: (batch, input_dim)
        grad_input = grad_z.dot(self.W.T)

        # Update parameters
        self.W -= lr * grad_W
        self.b -= lr * grad_b

        return grad_input
