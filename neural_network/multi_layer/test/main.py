from final import Net
from stolen import Network
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
import numpy as np

iris = datasets.load_iris()
# datasets.
X = iris.data
y = iris.target
y = to_categorical(y)
# print(X)
# model = Network([4, 1, 3])
# print(model.feedforward(X[0]))
# model.SGD(list(zip(X, y)), 5, 10, 0.01)

X = [np.array(x).reshape(-1, 1) for x in X]
y = [np.array(x).reshape(-1, 1) for x in y]
data = list(zip(X, y))
test = [(x, np.argmax(y_)) for x, y_ in data]
m1 = Network([4, 3, 3])
m2 = Net([4, 3, 3])
print(model.feedforward(X[0]))
model.SGD(list(zip(X, y)), 1000, 10, 0.1, test)
