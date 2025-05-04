import numpy as np

class LinearRegression:
    """
    Linear Regression implementation from scratch
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        """

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias  = None
        self.cost_history = []

    def fit(self,X :np.ndarray,y :np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data features
        y : numpy.ndarray
            Target values
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Initialize parameters
        
        n_samples ,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass (predictions)
            y_predicted = self._predict(X)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost for monitoring
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            
        return self
    
    def predict(self, X):
        """
        Predict using the linear model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        return self._predict(X)
    
    def _predict(self, X):
        """
        Make predictions with current weights and bias
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        """
        Compute Mean Squared Error cost
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True target values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        float
            Mean Squared Error cost
        """
        n_samples = len(y_true)
        cost = (1/n_samples) * np.sum((y_pred - y_true)**2)
        return cost
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test features
        y : numpy.ndarray
            True target values
            
        Returns:
        --------
        float
            R^2 score
        """
        y_pred = self.predict(X)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y))**2)
        
        # Residual sum of squares
        ss_res = np.sum((y - y_pred)**2)
        
        # R^2 score
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
