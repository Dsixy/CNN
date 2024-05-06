import numpy as np
from Func import *


class Layer:
    # 层
    def __init__(self, input_size, output_size):
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, X):
        return X

    def backward(self, dA):
        return dA

    def update(self, alpha):
        pass


class FC(Layer):
    # 全连接层
    def __init__(self, output_size=10, activation="relu", input_size=10):
        super().__init__(input_size, output_size)

        activations = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "softmax": Softmax(),
            "linear": Linear(),
            "tanh": Tanh(),
            "LeakyReLU": LeakyReLU(),
        }
        func = activations[activation]

        self.activation_name = activation
        self.activation = func.func
        self.derivative = func.derivative

        self.X = None
        self.A = None
        self.Z = None
        self.dZ = None
        self.dW = None
        self.optimizer = None
        if self.activation_name == "tanh":
            self.W = np.random.uniform(-np.sqrt(6. / (input_size + output_size)),
                                       np.sqrt(6. / (input_size + output_size)), size=(output_size, input_size))
        elif self.activation_name == 'sigmoid':
            self.W = 4 * np.random.uniform(-np.sqrt(6. / (input_size + output_size)),
                                           np.sqrt(6. / (input_size + output_size)), size=(output_size, input_size))
        else:
            self.W = np.random.randn(output_size, input_size)  # / input_size

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(self.W, X)
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, l2_rate=0.0003):
        self.dZ = dA * self.derivative(self.Z)
        self.dW = np.matmul(self.dZ, self.X.T) / len(self.X[0]) + l2_rate * self.W
        return np.matmul(self.W.T, self.dZ)

    def update(self, alpha):
        self.W -= alpha * self.optimizer.update(self.dW)

    def get_opt(self, optimizer):
        self.optimizer = optimizer()
        self.optimizer.m = np.zeros(self.W.shape)
        self.optimizer.v = np.zeros(self.W.shape)


class InputLayer(Layer):
    def __init__(self, input_size):
        super().__init__(input_size, input_size)

class Conv(Layer):
    def __init__(self, kernel_size, filter_num, padding='same', activation='relu'):
        
        pass

    def forward(self, X):
        pass

    def backward(self, dA):
        pass

    def update(self, alpha):
        pass

    def get_opt(self, optimizer):
        self.optimizer = optimizer()
        self.optimizer.m = np.zeros(self.W.shape)
        self.optimizer.v = np.zeros(self.W.shape)