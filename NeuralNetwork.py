import numpy as np
from Layer import Layer


class NeuralNetwork:
    def __init__(self):
        # input_dim increased to 6 after adding engineered features
        self.l1 = Layer(4, 6, activation="tanh")
        self.l2 = Layer(6, 4, activation="tanh")
        self.l3 = Layer(4, 1, activation="sigmoid")

    def forward(self, X):
        h1 = self.l1.forward(X)
        h2 = self.l2.forward(h1)
        y = self.l3.forward(h2)
        return h1, h2, y

    def train(self, X, Y, epochs=1000, lr=0.05, batch_size=64, on_epoch_end=None):
        n_samples = X.shape[0]
        for e in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            loss_total = 0.0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]

                # forward batch
                _, _, y_pred = self.forward(X_batch)
                # y_pred: (batch,1)
                y_flat = y_pred.flatten()
                loss_total += np.sum((y_flat - Y_batch) ** 2)

                # backward batch
                grad_y = (2 * (y_flat - Y_batch)).reshape(-1, 1)
                grad_h2 = self.l3.backward(grad_y, lr)
                grad_h1 = self.l2.backward(grad_h2, lr)
                self.l1.backward(grad_h1, lr)

            if e % 50 == 0:
                print(f"Epoch {e}, loss={loss_total:.4f}")
            # Callback for external monitoring/plotting
            if on_epoch_end is not None:
                try:
                    on_epoch_end(e, loss_total)
                except Exception:
                    pass

    def predict(self, x):
        x = np.array(x).reshape(1, -1)
        _, _, y = self.forward(x)
        return float(y.flatten()[0])
