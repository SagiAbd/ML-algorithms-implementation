import random
import numpy as np


class MyLineReg():

    def __init__(self, learning_rate, sgd_sample=None, random_state=42, n_iter=100, weights=None, metric=None, reg=None,
                 l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        X.insert(loc=0, column="w0", value=1)
        feature_num = len(X.columns)
        self.weights = np.ones(feature_num)

        for i in range(self.n_iter):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                X_sampled = X.to_numpy()[sample_rows_idx]
                y_sampled = y.to_numpy()[sample_rows_idx]
            if isinstance(self.sgd_sample, float):
                sample_size = int(X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sampled = X.to_numpy()[sample_rows_idx]
                y_sampled = y.to_numpy()[sample_rows_idx]
            else:
                X_sampled = X.to_numpy()
                y_sampled = y.to_numpy()

            y_predicted = X.to_numpy() @ self.weights
            if self.reg == 'l1':
                mse = np.mean(np.square(np.subtract(y_predicted, y.to_numpy())).sum()) + self.l1_coef * np.abs(
                    self.weights).sum()
                gradient = 2 / len(y_sampled) * (
                            X_sampled @ self.weights - y_sampled) @ X_sampled + self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                mse = np.mean(np.square(np.subtract(y_predicted, y.to_numpy())).sum()) + self.l2_coef * np.square(
                    self.weights).sum()
                gradient = 2 / len(y_sampled) * (
                            X_sampled @ self.weights - y_sampled) @ X_sampled + 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                mse = np.mean(np.square(np.subtract(y_predicted, y.to_numpy())).sum()) + self.l1_coef * np.abs(
                    self.weights).sum() + self.l2_coef * np.square(self.weights).sum()
                gradient = 2 / len(y_sampled) * (
                            X_sampled @ self.weights - y_sampled) @ X_sampled + self.l1_coef * np.sign(
                    self.weights) + 2 * self.l2_coef * self.weights
            else:
                mse = np.mean(np.square(np.subtract(y_predicted, y.to_numpy())).sum())
                gradient = 2 / len(y_sampled) * (X_sampled @ self.weights - y_sampled) @ X_sampled

            if callable(self.learning_rate):
                self.weights = self.weights - self.learning_rate(i + 1) * gradient
            else:
                self.weights = self.weights - self.learning_rate * gradient

            if verbose:
                if i % verbose == 0:
                    print(f"{i} | loss: {mse} | {self.metric}: {self.calculate_metric(y, y_predicted)}")

        self.best_score = self.calculate_metric(y, X.to_numpy() @ self.weights)

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X.insert(loc=0, column="w0", value=1)
        return X.to_numpy() @ self.weights

    def calculate_metric(self, y, y_pred):
        if self.metric == 'mae':
            return np.mean(np.abs(y - y_pred))
        elif self.metric == 'mse':
            return np.mean((y - y_pred) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2))
        elif self.metric == 'mape':
            return np.mean(np.abs((y - y_pred) / y)) * 100
        elif self.metric == 'r2':
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total)
        else:
            return None

    def get_best_score(self):
        return self.best_score
