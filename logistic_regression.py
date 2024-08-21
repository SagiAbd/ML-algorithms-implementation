import random

import numpy as np
import pandas as pd


class MyLogReg():
    def __init__(self, weights=None, sgd_sample=None, random_state=42, n_iter=10, learning_rate=0.1, metric=None,
                 reg=None, l1_coef=0, l2_coef=0):
        """
        Initialize the logistic regression model with optional parameters for regularization,
        learning rate, stochastic gradient descent sampling, and other settings.
        """
        self.weights = weights  # Model weights
        self.n_iter = n_iter  # Number of iterations for training
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.metric = metric  # Evaluation metric (accuracy, precision, recall, etc.)
        self.best_score = None  # Best score during training
        self.reg = reg  # Regularization type ('l1', 'l2', 'elasticnet')
        self.l1_coef = l1_coef  # Coefficient for L1 regularization
        self.l2_coef = l2_coef  # Coefficient for L2 regularization
        self.random_state = random_state  # Random seed for reproducibility
        self.sgd_sample = sgd_sample  # Sample size for SGD (int for fixed size, float for fraction of data)

    def __str__(self):
        """Return a string representation of the model's main parameters."""
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        """
        Train the logistic regression model using the given data X and labels y.
        Supports stochastic gradient descent (SGD) if sgd_sample is specified.
        """
        random.seed(self.random_state)

        y = y.to_numpy()  # Convert labels to numpy array
        self.weights = np.ones(X.shape[1] + 1)  # Initialize weights with ones
        X = np.insert(X.to_numpy(), 0, 1, axis=1)  # Add bias term (intercept) to the feature matrix

        for i in range(self.n_iter + 1):
            # Sample a subset of data for SGD if sgd_sample is specified
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                X_sampled = X[sample_rows_idx]
                y_sampled = y[sample_rows_idx]
            elif isinstance(self.sgd_sample, float):
                sample_size = int(X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sampled = X[sample_rows_idx]
                y_sampled = y[sample_rows_idx]
            elif self.sgd_sample is None:
                X_sampled = X
                y_sampled = y

            # Compute predicted probabilities and loss
            y_pred_sampled = self._sigmoid(X_sampled @ self.weights)
            LogLoss = self._loss_function(y_sampled, y_pred_sampled)
            gradient = self._gradient(X_sampled, y_sampled, y_pred_sampled)

            if i < self.n_iter:  # Avoid updating weights after the last iteration
                if callable(self.learning_rate):
                    lr = self.learning_rate(i + 1)
                else:
                    lr = self.learning_rate

                # Update weights using the computed gradient
                self.weights -= lr * gradient

            # Print progress if verbose is enabled
            if verbose and i % verbose == 0:
                print(f"{i} | loss: {LogLoss}  | {self.metric}: {self._calculate_metric(y_sampled, y_pred_sampled)}")

        # Compute the final metric score on the full data after the last iteration
        self.best_score = self._calculate_metric(y, self._sigmoid(X @ self.weights))

    def _calculate_metric(self, y, y_pred):
        """
        Calculate the evaluation metric specified during initialization.
        """
        if self.metric is None:
            return None
        elif self.metric == 'accuracy':
            return self._accuracy(y, y_pred)
        elif self.metric == 'precision':
            return self._precision(y, y_pred)
        elif self.metric == 'recall':
            return self._recall(y, y_pred)
        elif self.metric == 'f1':
            return self._f1(y, y_pred)
        elif self.metric == 'roc_auc':
            return self._roc_auc(y, y_pred)

    def _roc_auc(self, y, y_pred):
        """
        Calculate the ROC AUC score manually, using the ranks of positive and negative samples.
        """
        df = pd.DataFrame({'prob': y_pred, 'class': y})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)

        roc_sum = 0
        P = np.sum(df['class'] == 1)  # Number of positive samples
        N = np.sum(df['class'] == 0)  # Number of negative samples

        for index, row in df.iterrows():
            if row['class'] == 0:
                count_higher = df[df['prob'] > row['prob']]['class'].sum()
                count_same = df[(df['prob'] == row['prob']) & (df['class'] == 1)].shape[0]
                roc_sum += count_higher + count_same / 2

        roc_auc = roc_sum / (P * N)
        return roc_auc

    def _f1(self, y, y_pred):
        """Calculate the F1 score."""
        f1 = 2 * (self._precision(y, y_pred) * self._recall(y, y_pred)) / (
                self._precision(y, y_pred) + self._recall(y, y_pred))
        return f1

    def _recall(self, y, y_pred):
        """Calculate the recall score."""
        tp, tn, fp, fn = self._error_matrix(y, y_pred)
        recall = tp / (tp + fn)
        return recall

    def _precision(self, y, y_pred):
        """Calculate the precision score."""
        tp, tn, fp, fn = self._error_matrix(y, y_pred)
        precision = tp / (tp + fp)
        return precision

    def _accuracy(self, y, y_pred):
        """Calculate the accuracy score."""
        tp, tn, fp, fn = self._error_matrix(y, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    def _convert_to_binary(self, y_pred):
        """Convert predicted probabilities to binary outcomes based on a threshold of 0.5."""
        return (y_pred > 0.5).astype(int)

    def _error_matrix(self, y, y_pred):
        """
        Calculate the error matrix (True Positive, True Negative, False Positive, False Negative).
        """
        y_pred_bin = self._convert_to_binary(y_pred)
        tp = sum((y_pred_bin == 1) & (y == 1))
        tn = sum((y_pred_bin == 0) & (y == 0))
        fp = sum((y_pred_bin == 1) & (y == 0))
        fn = sum((y_pred_bin == 0) & (y == 1))

        return tp, tn, fp, fn

    def _sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def _loss_function(self, y, y_pred):
        """
        Calculate the logistic loss function, with optional regularization (L1, L2, or ElasticNet).
        """
        eps = 1e-15  # Small value to avoid log(0)
        loss = - 1 / len(y) * (y @ np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)).sum()
        if self.reg == 'l1':
            loss += self.l1_coef * np.abs(self.weights).sum()
        elif self.reg == 'l2':
            loss += self.l2_coef * np.power(self.weights, 2).sum()
        elif self.reg == 'elasticnet':
            loss += self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * np.power(self.weights, 2).sum()

        return loss

    def _gradient(self, X, y, y_pred):
        """
        Calculate the gradient of the loss function with respect to the weights, including regularization.
        """
        grad = 1 / len(y) * (y_pred - y) @ X
        if self.reg == 'l1':
            grad += self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            grad += 2 * self.l2_coef * self.weights
        elif self.reg == 'elasticnet':
            grad += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

        return grad

    def get_coef(self):
        """Return the coefficients (weights) of the trained model, excluding the intercept term."""
        return self.weights[1:]

    def predict_proba(self, X):
        """Predict probabilities for the input data X."""
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        return self._sigmoid(X @ self.weights).mean()

    def predict(self, X):
        """Convert probabilities to binary classes based on threshold > 0.5."""
        X = np.insert(X.to_numpy(), 0, 1, axis=1)
        prob = self._sigmoid(X @ self.weights)
        return (prob > 0.5).astype(int)

    def get_best_score(self):
        """Return the best metric score obtained during training."""
        return self.best_score
