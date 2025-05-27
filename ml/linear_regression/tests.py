"""Test a linear regression model with early stopping and data validation."""

import argparse
import logging
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Add the path to import your custom LinearRegression
sys.path.append(os.path.join(os.path.dirname(__file__), 'linear_regression'))

try:
    from model import LinearRegression
except ImportError:
    print("Could not import LinearRegression from model.py")
    print("Make sure the model.py file is in the linear_regression/ directory")
    sys.exit(1)

class TestLinearRegressionLogic(unittest.TestCase):
    """Unit tests for LinearRegression class"""
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = LinearRegression(learning_rate=0.01, n_iterations=1000)

        # Create simple test data
        np.random.seed(42)
        self.X_simple = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        self.y_simple = np.array([0., 0., 0., 1., 1., 1.])  # floats for regression

        # Create more complex test data
        self.X_complex, self.y_complex = make_regression(
            n_samples=200, n_features=4, n_informative=3, noise=0.1, random_state=42
        )

    
