import numpy as np

class NeuralNetwork:
    def __init__(self, input_units, hidden_units, output_units, task='classification'):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.task = task
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.W1 = np.random.uniform(-0.01, 0.01, (self.input_units, self.hidden_units))
        self.b1 = np.zeros(self.hidden_units)
        self.W2 = np.random.uniform(-0.01, 0.01, (self.hidden_units, self.output_units))
        self.b2 = np.zeros(self.output_units)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2

    def compute_loss(self, y_true, y_pred):
        if self.task == 'classification':
            # Use softmax for predictions
            y_pred = self.softmax(y_pred)
            return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]
        else:
            return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred, learning_rate):
        if self.task == 'classification':
            m = y_pred.shape[0]
            probs = self.softmax(y_pred)
            dz2 = probs - y_true
            dW2 = np.dot(self.relu(np.dot(X, self.W1) + self.b1).T, dz2) / m
            db2 = np.sum(dz2, axis=0) / m

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.relu(np.dot(X, self.W1) + self.b1) > 0)
            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0) / m
        else:
            m = y_pred.shape[0]
            dz2 = (y_pred - y_true) / m
            dW2 = np.dot(self.relu(np.dot(X, self.W1) + self.b1).T, dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.relu(np.dot(X, self.W1) + self.b1) > 0)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0)
        
        # Clip gradients
        max_norm = 1.0
        for grad in [dW1, db1, dW2, db2]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2