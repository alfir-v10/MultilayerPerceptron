import pandas as pd
import numpy as np


class MLP:
    def __init__(self, X=None, Y=None, X_val=None, Y_val=None, L=1, N_l=128):
        self.batch_size = None
        self.lr = None
        if X is not None:
            self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        if Y is not None:
            self.Y = np.squeeze(np.eye(2)[Y.astype(np.int64).reshape(-1)])
        if X_val is not None:
            self.X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1)
        if Y_val is not None:
            self.Y_val = np.squeeze(np.eye(2)[Y_val.astype(np.int64).reshape(-1)])
        self.L = L
        self.N_l = N_l
        if X is not None:
            self.n_samples = self.X.shape[0]
        if Y is not None:
            self.Y_shape = self.Y.shape[1]
        if X is not None:
            self.layer_sizes = np.array([self.X.shape[1]] + [N_l] * L + [self.Y.shape[1]])
        self.weights = None
        self.__h = None
        self.__out = None
        if X is not None:
            self.__init_weights()
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.metrics = [self.train_loss, self.train_acc, self.val_loss, self.val_acc]

    def __sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def __softmax(self, x):
        exponent = np.exp(x)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def __loss(self, y_pred, y):
        return ((-np.log(y_pred)) * y).sum(axis=1).mean()

    def __accuracy(self, y_pred, y):
        return np.all(y_pred == y, axis=1).mean()

    def __sigmoid_prime(self, h):
        return h * (1 - h)

    def __to_categorical(self, x):
        categorical = np.zeros((x.shape[0], self.Y_shape))
        categorical[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
        return categorical

    def __init_weights(self):
        self.weights = []
        for i in range(self.layer_sizes.shape[0] - 1):
            self.weights.append(np.random.uniform(-1, 1, size=[self.layer_sizes[i], self.layer_sizes[i + 1]]))
        self.weights = np.asarray(self.weights, dtype=object)

    def __init_layers(self, batch_size):
        self.__h = [np.empty((batch_size, layer)) for layer in self.layer_sizes]

    def __feed_forward(self, batch):
        h_l = batch
        self.__h[0] = h_l
        for i, weights in enumerate(self.weights):
            h_l = self.__sigmoid(h_l.dot(weights))
            self.__h[i + 1] = h_l
        self.__out = self.__softmax(self.__h[-1])

    def __back_prop(self, batch_y):
        delta_t = (self.__out - batch_y) * self.__sigmoid_prime(self.__h[-1])
        for i in range(1, len(self.weights) + 1):
            self.weights[-i] -= self.lr * (self.__h[-i - 1].T.dot(delta_t)) / self.batch_size
            delta_t = self.__sigmoid_prime(self.__h[-i - 1]) * (delta_t.dot(self.weights[-i].T))

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.__init_layers(X.shape[0])
        self.__feed_forward(X)
        return self.__to_categorical(self.__out)

    def evaluate(self, X, Y):
        prediction = self.predict(X)
        return self.__accuracy(prediction, Y)

    def save_model(self, file='model.pkl'):
        params = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'L': self.L,
            'N_l': self.N_l,
            'n_samples': self.n_samples,
            'layer_sizes': self.layer_sizes,
            '__h': self.__h,
            'weights': self.weights,
            '__out': self.__out,
            'Y_shape': self.Y_shape
        }
        pd.to_pickle(obj=params, filepath_or_buffer=file)

    def load_model(self, file='model.pkl'):
        params = pd.read_pickle(filepath_or_buffer=file)
        self.batch_size = params['batch_size']
        self.lr = params['lr']
        self.L = params['L']
        self.N_l = params['N_l']
        self.n_samples = params['n_samples']
        self.layer_sizes = params['layer_sizes']
        self.__h = params['__h']
        self.weights = params['weights']
        self.__out = params['__out']
        self.Y_shape = params['Y_shape']

    def train(self, batch_size=8, epochs=25, lr=1.0, early_stopping=2):
        self.lr = lr
        self.batch_size = batch_size
        flag_stop = 0
        for epoch in range(epochs):
            self.__init_layers(self.batch_size)
            shuffle = np.random.permutation(self.n_samples)
            train_loss = 0
            train_acc = 0
            X_batches = np.array_split(self.X[shuffle], self.n_samples / self.batch_size)
            Y_batches = np.array_split(self.Y[shuffle], self.n_samples / self.batch_size)
            for batch_x, batch_y in zip(X_batches, Y_batches):
                self.__feed_forward(batch_x)
                train_loss += self.__loss(self.__out, batch_y)
                train_acc += self.__accuracy(self.__to_categorical(self.__out), batch_y)
                self.__back_prop(batch_y)

            train_loss = (train_loss / len(X_batches))
            train_acc = (train_acc / len(X_batches))
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            self.__init_layers(self.X_val.shape[0])
            self.__feed_forward(self.X_val)
            val_loss = self.__loss(self.__out, self.Y_val)
            val_acc = self.__accuracy(self.__to_categorical(self.__out), self.Y_val)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            print(
                f"epoch {epoch + 1}/{epochs} - loss: {round(train_loss, 4)} - acc: {round(train_acc, 4)} "
                f"- val_loss: {round(val_loss, 4)} - val_acc: {round(val_acc, 4)}")

            if early_stopping:

                if len(self.val_loss) > 2 and val_loss > self.val_loss[-2]:
                    flag_stop += 1
                if flag_stop > early_stopping:
                    print('Early stopping activated')
                    break
