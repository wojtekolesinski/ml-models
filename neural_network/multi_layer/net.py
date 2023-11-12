from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        a = self(x)
        return a * (1 - a)


class Relu(ActivationFunction):
    def __call__(self, x):
        return np.max(0, x)

    def derivative(self, x):
        Sigmoid()(x)


class Layer:
    def __init__(self, input, output, f: ActivationFunction, eta):
        self.weights = np.random.rand(input, output)
        self.bias = np.random.rand(input)
        self.f = f
        self.alpha = eta

    def predict(self, x):
        self.z = self.weights @ x + self.bias
        self.a = self.f(self.z)
        return self.a

    def fit(self, e, s):
        # Zapisac wagi dla wyliczenia bledu do poprzedniej warstwy
        # Wykonac korekcje wag dla aktualnej warstwy
        return 1.

    def getW(self):
        allW = []
        for n in self.neurons:
            for w in n.W:
                allW.append(w)
        return np.array(allW)


class NeuronNetwork:
    def __init__(self, layers, acti, eta):
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i - 1], layers[i], acti, eta))

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.predict(out)
        return out

    def backpropagate(self, y):
        errors = np.zeros(len(self.layers))
        partials = np.zeros((self.layers - 1, 2))
        last = self.layers[-1]
        errors[-1] = -(y - last.a) * last.f.derivative(last.z)
        for i, layer in reversed(list(enumerate(self.layers[:-1]))):
            errors[i] = layer.weights.T * errors[i + 1] * layer.f.derivative(layer.z)
            partials[i, 0], partials[i, 1] = errors[i + 1] @ layer.a.T, errors[i + 1]
        return errors, partials

    def fit(self, X, y, e, s):
        for x, y_ in zip(X, y):
            self.predict(x)
            errors, partials = self.backpropagate(y_)
        # Idac od warstwy wyjsciowej, pobrac wagi i blad aktualnej warstwy do wyliczenia bledu dla kolejnej, i przeprowadzic trening aktualnej warstwy.
        return 1.
