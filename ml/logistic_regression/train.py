"""Train a logistic regression model with early stopping and data validation.
This script includes functions for loading data, training the model,
plotting cost history, and saving the model.
It also includes error handling and logging for better debugging."""

import argparse
import logging
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression.model import LogisticRegression


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)n

# Function to load data

def validate_parameters(args):
    """
    Validate input parameters
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if args.n_iterations <= 0:
        raise ValueError("Number of iterations must be positive")
    if args.n_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if args.n_features <= 0:
        raise ValueError("Number of features must be positive")
    if args.noise < 0:
        raise ValueError("Noise must be non-negative")

def load_data(n_samples, n_features, noise):
    """
    Load synthetic data for training and validation
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise : float
        Standard deviation of Gaussian noise added to the output
    
    Returns:
    --------
    X_train, X_val, y_train, y_val : tuple of np.ndarray
        Training and validation data
    """
    logger.info("Loading data...")
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val

def 
