import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

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
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_predicted):
        """
        Binary cross-entropy loss
        """
        epsilon = 1e-15  # avoid log(0)
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return loss

    def fit(self, X, y):
        """
        Fit the model to the training data
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: print loss every 100 iterations
            if i % 100 == 0:
                loss = self.compute_loss(y, y_predicted)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """
        Predict probability estimates for the input data
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels for input data using a 0.5 threshold
        """
        y_probs = self.predict_proba(X)
        return np.array([1 if i > 0.5 else 0 for i in y_probs])


# --- Example usage with synthetic data ---

# Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                           n_informative=5, random_state=42)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy on test set: {accuracy:.4f}")
