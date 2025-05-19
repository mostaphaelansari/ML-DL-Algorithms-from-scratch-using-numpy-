"""Logistic Regression Model.

 This module implements a simple logistic regression model using gradient descent.
"""

# Import necessary libraries

import numpy as np


class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_predicted):
        """
        Binary cross-entropy loss function
        """
        epsilon = 1e-15  # avoid log(0)
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return loss

    def fit(self, X, y):
        """
        Train the logistic regression model
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optionally print loss
            if i % 100 == 0:
                loss = self.compute_loss(y, y_predicted)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """
        Predict class probabilities for input features
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels using a threshold of 0.5
        """
        y_probs = self.predict_proba(X)
        return (y_probs >= 0.5).astype(int)
