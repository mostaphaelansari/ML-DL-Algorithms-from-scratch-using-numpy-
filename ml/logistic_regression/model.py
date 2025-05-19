import numpy as np

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """
        Initialize Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        """
        self.learning_rate =learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the log-likelihood cost function
        """
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Compute the gradient of the cost function with respect to weights and bias
        """
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the model to the training data
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Gradient computation
            dw, db = self._compute_gradient(X, y, y_predicted)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Cost tracking
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for given input features
        """
        y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.where(y_predicted >= 0.5, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for given input features
        """
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the accuracy of the model
        """
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)
