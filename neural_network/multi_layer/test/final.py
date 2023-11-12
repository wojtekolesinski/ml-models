import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow.keras.utils import to_categorical

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    x = sigmoid(x)
    return x * (1 - x)


class Net:
    def __init__(self, layers, alpha=0.01):
        self.num_layers = len(layers)
        self.weights = [np.random.randn(m, n)/np.sqrt(n) for n, m in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]
        self.alpha = alpha

    def predict(self, X):
        for W, b in list(zip(self.weights, self.biases)):
            X = sigmoid(np.dot(W, X) + b)
        return X

    def feed_forward(self, X):
        activations = [X]
        for W, b in list(zip(self.weights, self.biases)):
            X = sigmoid(np.dot(W, X) + b)
            activations.append(X)
        return activations, X

    def backpropagate(self, X, y):
        delta_W = [np.zeros(W.shape) for W in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        activations, y_hat = self.feed_forward(X)
        delta = (y_hat - y) * activations[-1] * (1 - activations[-1])

        delta_b[-1] = delta
        delta_W[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * activations[-l] * (1 - activations[-l])
            delta_b[-l] = delta
            delta_W[-l] = np.dot(delta, activations[-l - 1].T)
        return delta_W, delta_b

    def train(self, train_data, epochs, batch_size, test_data=None):
        history = []
        for e in range(epochs):
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i: i + batch_size]

                delta_W = [np.zeros(W.shape) for W in self.weights]
                delta_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in batch:

                    batch_delta_W, batch_delta_b = self.backpropagate(x, y)
                    delta_W = [dW+bdW for dW, bdW in zip(delta_W, batch_delta_W)]
                    delta_b = [db+bdb for db, bdb in zip(delta_b, batch_delta_b)]

                    if batch_size == 1:
                        self.weights = [w - self.alpha / len(batch) * dW for w, dW in zip(self.weights, delta_W)]
                        self.biases = [b - self.alpha / len(batch) * db for b, db in zip(self.biases, delta_b)]
                        delta_W = [np.zeros(W.shape) for W in self.weights]
                        delta_b = [np.zeros(b.shape) for b in self.biases]

                if batch_size != 1:
                    self.weights = [w - self.alpha / len(batch) * dW for w, dW in zip(self.weights, delta_W)]
                    self.biases = [b - self.alpha / len(batch) * db for b, db in zip(self.biases, delta_b)]

            if test_data:
                print(f'Epoch {e}/{epochs}: accuracy: {self.score(test_data)}/{len(test_data)}')
            else:
                print(f'Epoch: {e}/{epochs}')
            history.append(self.predict(x) for x, _ in train_data)
        return history

    def score(self, test_data):
        X = [d[0] for d in test_data]
        y = [d[1] for d in test_data]
        y_hat = np.array([np.argmax(self.predict(x)) for x in X])
        y_true = np.array([np.argmax(y_) for y_ in y])
        return np.sum(y_hat == y_true)


iris = datasets.load_iris()
X = iris.data
y = iris.target
y = to_categorical(y)
list(zip(X, y))
X = [np.array(x).reshape(-1, 1) for x in X]
y = [np.array(x).reshape(-1, 1) for x in y]
data = list(zip(X, y))

m1 = Net([4, 3, 3], 0.05)
m2 = Net([4, 3, 3], 0.05)

# def loss(y_true, y_pred):


h1 = m1.train(data, epochs=500, batch_size=5, test_data=data)
print('\n\n\n')
# print(np.array(y).reshape(150, 3))
print(np.array(h1))
# h2 = m2.train(data, epochs=300, batch_size=1, test_data=data)

