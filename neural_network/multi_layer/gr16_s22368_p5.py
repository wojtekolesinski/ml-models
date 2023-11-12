import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

iris = datasets.load_iris()
X = iris.data
y = iris.target
y = to_categorical(y)

# norm=np.linalg.norm(y)
# y/=norm

print(np.shape(X))
print(np.shape(y))
print(iris.target)


class Sigmoid:
    def acti(self, x):
        return 1 / (1 + np.exp(-x))

    def der(self, x):
        a = self.acti(x)
        return a * (1 - a)


class Neuron:
    def __init__(self, input, acti, eta):
        self.W = np.random.rand(input)
        self.Wb = np.random.rand(1)[0]
        self.acti = acti
        self.eta = eta

    def predict(self, x):
        self.s = np.dot(self.W, x) + self.Wb
        return self.acti.acti(self.s)

    def fit(self, e):
        error = np.dot(e, self.W)
        delta = self.acti.der(self.s) - error
        self.W += self.eta * delta * self.W
        return error


class Layer:
    def __init__(self, input, output, acti, eta):
        self.neurons = []
        for i in range(output):
            self.neurons.append(Neuron(input, acti, eta))

    def predict(self, x):
        return np.array([n.predict(x) for n in self.neurons])

    def fit(self, e):
        errors = np.array([n.fit(e) for n in self.neurons])
        return errors


class NeuronNetwork:
    def __init__(self, layers, acti, eta):
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i - 1], layers[i], acti, eta))

    def predict(self, x):
        res = self.layers[0].predict(x)
        for l in self.layers[1:]:
            res = l.predict(res)
        return res

    def fit(self, X, y, e):
        errors = e
        for i, layer in enumerate(self.layers[:-1:-1]):
            errors = layer.fit(errors)


def batch(NN, epoch=100):
    for i in range(epoch):
        e = 0
        for xe, ye in zip(X, y):
            p = NN.predict(xe)
            e += p - ye
        e /= len(X)
        NN.fit(X, y, e)


def online(NN, epoch=100):
    acc = []
    for i in range(epoch):
        preds = []
        for xe, ye in zip(X, y):
            e = NN.predict(xe)
            preds.append(e)
            e -= ye
            NN.fit(X, y, e)
        y_true = np.array(iris.target)
        y_pred = np.array([np.argmax(y_) for y_ in preds])
        acc.append(np.sum(y_true == y_pred)/len(y_true))
    return acc



# batch(NeuronNetwork([4, 10, 20, 50, 3], Sigmoid(), 0.001), 1)
net = NeuronNetwork([4, 10, 20, 50, 3], Sigmoid(), 0.001)
acc = online(net, 100)


import matplotlib.pyplot as plt

plt.plot(acc, range(len(acc)))
print(acc)
plt.show()
