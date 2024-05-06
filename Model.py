import numpy as np
from Layer import FC
import pickle


class Model:
    def __init__(self, layers: list):
        self.layers = layers
        self.layer_num = len(self.layers)

    def get_opt(self, optimizer=None):
        for layer in self.layers:
            layer.get_opt(optimizer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

    def train(self, X, y, batch_size, epochs, alpha, loss_func, optimizer, X_test=None, y_test=None):
        if y.ndim == 1:
            y = y.reshape((len(y), 1))

        sample_num = X.shape[0]
        left = sample_num % batch_size
        self.get_opt(optimizer)

        if X_test is None:
            X_test = X
            y_test = y

        y_test = y_test.reshape(1, -1)

        for epoch in range(epochs):
            idx = np.random.permutation(sample_num)
            # 使用打乱的索引来重新排序数据集
            X = X[idx]
            y = y[idx]

            X_split = np.split(X[left:], batch_size)
            y_split = np.split(y[left:], batch_size)
            for X_batch, y_batch in zip(X_split, y_split):
                # idx = np.random.randint(sample_num, size=batch_size)
                # X_batch, y_batch = X[idx], y[idx]
                X_batch, y_batch = X_batch.T, y_batch.T

                # Forward propagation
                y_hat = self.forward(X_batch)
                # print(y_hat)

                # Compute loss
                # softmax
                if self.layers[-1].activation_name in ["softmax", "sigmoid", "linear"]:
                    loss = loss_func.derivative(y_batch, y_hat)
                else:
                    raise "Don't support this output layer!"

                # Backward propagation
                self.backward(loss)
                self.update(alpha)

            if epoch % 10 == 0:
                print("epoch:{:d}".format(epoch), "loss:{:.5f}".format(loss_func.loss(y_test, self.predict(X_test))))

        print("accuracy:{:.2f}".format(np.sum(y_test[0] == np.argmax(self.predict(X_test), axis=1)) / len(X_test)))

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape((len(X), 1))
        elif X.ndim == 3:
            X = X.reshape(len(X), -1)
        return self.forward(X.T).T

    def save_model(self):
        weight = []
        activation = []

        for layer in self.layers:
            weight.append(layer.W)
            activation.append(layer.activation_name)

        rec = [weight, activation]

        with open("model.pkl", "wb") as f:
            pickle.dump(rec, f)

    def load_parameter(self, file):
        with open(file, "rb") as f:
            rec = pickle.load(f)

        weight, activation = rec[0], rec[1]
        self.layer_num = len(weight)
        self.layers = [0] * self.layer_num
        for i in range(self.layer_num):
            self.layers[i] = FC(len(weight[i]), activation[i], len(weight[i][0]))
            self.layers[i].W = weight[i]
