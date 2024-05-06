import numpy as np
# function: y = x
# derivative of function: y = 1
# function: y = x
# derivative of function: y = 1
class Linear:
    def func(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class ReLU:
    def func(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid:
    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.func(x) * (1 - self.func(x))


class Softmax:
    def func(self, x):
        tmp = np.exp(x - np.max(x, axis=0, keepdims=True))
        tmp /= np.sum(tmp, axis=0, keepdims=True)
        return tmp

    def derivative(self, x):
        return 1#np.ones((len(x), 1))


class Tanh:
    def func(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 / (np.cosh(x) * np.cosh(x))


class LeakyReLU:
    def func(self, x):
        return np.where(x > 0, x, 0.1*x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0.1)


# y is the true value, y_hat is the predicted value
class Loss_Func:
    def loss(self, y, y_hat):
        pass

    def derivative(self, y, y_hat):
        return y


class MSE(Loss_Func):
    def loss(self, y, y_hat):
        return np.mean((y - y_hat) * (y - y_hat))

    def derivative(self, y, y_hat):
        return 2 * (y_hat - y)


class Logistic(Loss_Func):
    def loss(self, y, y_hat):
        return np.mean(-y * np.log(y_hat + 1e-6) - (1 - y) * np.log(1 - y_hat + 1e-6))

    def derivative(self, y, y_hat):
        return -y / y_hat + (1 - y) / (1 - y_hat)


# Assume y_hat is LINEAR
class CrossEntropy(Loss_Func):
    def loss(self, y, y_hat):
        return -np.mean(np.log(y_hat[np.arange(len(y[0])), y[0]]+1e-10))

    def derivative(self, y, y_hat):
        tmp = y_hat.copy()
        tmp[y[0], np.arange(len(y[0]))] -= 1
        return tmp
