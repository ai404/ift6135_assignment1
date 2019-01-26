import numpy as np
import pickle


class NN(object):

    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, mode='train', datapath=None, model_path=None):

        assert len(hidden_dims) == n_hidden, "Hidden dims mismatch!"

        self.hidden_dims = hidden_dims
        self.n_hidden = n_hidden
        self.mode = mode
        self.datapath = datapath
        self.model_path = model_path
        self.epsilon = 1e-6
        self.lr = 1e-1
        self.tr, self.va, self.te = np.load(open(datapath, "rb"))
        self.n_epochs = 1000
        self.batch_size = 1000

    def initialize_weights(self, dims):
        """
        :param dims: the size of input/output layers
        :return: None
        """
        if self.mode == "train":
            self.weights = {}
            all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
            print(all_dims)
            for layer_n in range(1, self.n_hidden + 2):
                self.weights[f"W{layer_n}"] = np.random.rand(all_dims[layer_n - 1], all_dims[layer_n]) / 50
                self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))  # np.random.rand(1, all_dims[layer_n])
        elif self.mode == "test":
            pass
        else:
            raise Exception("Unknown Mode!")

    def forward(self, input):  #
        cache = {"H0": input}
        for layer in range(1, self.n_hidden + 1):
            cache[f"A{layer}"] = cache[f"H{layer-1}"] @ self.weights[f"W{layer}"] + self.weights[f"b{layer}"]
            cache[f"H{layer}"] = self.activation(cache[f"A{layer}"])

        layer = self.n_hidden + 1
        cache[f"A{layer}"] = cache[f"H{layer-1}"] @ self.weights[f"W{layer}"] + self.weights[f"b{layer}"]
        cache[f"H{layer}"] = self.softmax(cache[f"A{layer}"])
        return cache

    def activation(self, input, prime=False):
        if prime:
            return input > 0
        return np.maximum(0, input)

    def loss(self, prediction, labels):  #
        # TODO
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return - np.sum(labels * np.log(prediction)) # / prediction.shape[0]

    def softmax(self, input):  #
        Z = np.exp(input - np.max(input))
        return Z / np.sum(Z, axis=1, keepdims=True)

    def backward(self, cache, labels):  #
        # TODO
        output = cache[f"H{self.n_hidden+1}"]
        grads = {
            f"dA{self.n_hidden+1}": - (labels - output),
        }
        for layer in range(self.n_hidden + 1, 0, -1):
            # print(f"Shape dA=", grads[f"dA{layer}"].shape)
            # print(f"Shape H=", cache[f"H{layer-1}"].shape)

            grads[f"dW{layer}"] = cache[f"H{layer-1}"].T @ grads[f"dA{layer}"]
            grads[f"db{layer}"] = grads[f"dA{layer}"]

            if layer > 1:
                grads[f"dH{layer-1}"] = grads[f"dA{layer}"] @ self.weights[f"W{layer}"].T
                grads[f"dA{layer-1}"] = grads[f"dH{layer-1}"] * self.activation(cache[f"A{layer-1}"], prime=True)
                # print(f"Shape dA=", grads[f"dA{layer-1}"].shape)
        return grads

    def update(self, grads):  #
        # rint(grads.keys())
        for layer in range(1, self.n_hidden + 1):
            # print(grads[f"dW{layer}"].shape,self.weights[f"W{layer}"].shape)
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"] / self.batch_size

    def train(self):
        X_train, y_train = self.tr
        y_onehot = np.eye(np.max(y_train) - np.min(y_train) + 1)[y_train]
        # print(y_train.shape,y_onehot.shape)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(self.n_epochs):
            predictedY = np.zeros_like(y_train)
            trainLoss = 0
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

                trainLoss += self.loss(cache[f"H{self.n_hidden+1}"], minibatchY)
                predictedY[self.batch_size * batch:self.batch_size * (batch + 1)] = np.argmax(
                    cache[f"H{self.n_hidden + 1}"], axis=1)

            X_val, y_val = self.va
            onVal_y = np.eye(np.max(y_train) - np.min(y_train) + 1)[y_val]
            valCache = self.forward(X_val)

            predicted_valY = np.argmax(valCache[f"H{self.n_hidden + 1}"], axis=1)
            valAccuracy = np.mean(y_val == predicted_valY)
            valLoss = self.loss(valCache[f"H{self.n_hidden+1}"], onVal_y)

            trAccuracy = np.mean(y_train == predictedY)

            print(f"Epoch= {epoch}, Loss={trainLoss:10.2f}, Accuracy={trAccuracy:4.2f}, Val.Loss={valLoss:10.2f}, Val.Accuracy= {valAccuracy:4.2f}")
            # break

    def test(self):
        pass


neural_net = NN(datapath="mnist.npy", hidden_dims=(500, 400))
neural_net.train()
