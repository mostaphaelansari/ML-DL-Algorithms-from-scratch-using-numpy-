"""Train a linear regression model with early stopping and data validation.
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

from linear_regression.model import LinearRegression


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

def load_data(use_sklearn=True, n_samples=1000, n_features=1, noise=20, test_size=0.2, random_state=42, 
            data_path=None, output_dir='output'):
    """
    Load or generate regression data
    
    Parameters:
    -----------
    use_sklearn : bool
        Whether to generate synthetic data using sklearn
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features to generate
    noise : float
        Noise to add to the data
    test_size : float
        Test split ratio
    random_state : int
        Random seed
    data_path : str, optional
        Path to custom dataset
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    try:
        if use_sklearn:
            # Generate synthetic data
            logger.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
            X, y = make_regression(n_samples=n_samples, 
                                n_features=n_features, 
                                noise=noise, 
                                random_state=random_state)
            
            # Keep y as a flat array (sklearn convention)
            y = y.ravel()
        else:
            # Load custom dataset if provided
            if data_path:
                logger.info(f"Loading data from {data_path}")
                try:
                    data = np.loadtxt(data_path, delimiter=',')
                    X = data[:, :-1]
                    y = data[:, -1].ravel()  # Keep as flat array
                except Exception as e:
                    logger.error(f"Error loading data from {data_path}: {e}")
                    logger.warning("Falling back to synthetic data")
                    X, y = make_regression(n_samples=n_samples, 
                                        n_features=n_features, 
                                        noise=noise, 
                                        random_state=random_state)
                    y = y.ravel()
            else:
                # Fallback to synthetic data
                logger.info("No custom data path provided, using synthetic data")
                X, y = make_regression(n_samples=n_samples, 
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_state)
                y = y.ravel()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save the scaler for later use
        output_path = Path(output_dir)
        scaler_path = output_path / 'scaler.joblib'
        
        try:
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Unexpected error in load_data: {e}")
        raise

def plot_cost_history(cost_history, save_path='cost_history.png'):
    """
    Plot the cost history
    
    Parameters:
    -----------
    cost_history : list
        Cost values for each iteration
    save_path : str
        Path to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(cost_history)), cost_history)
        plt.title('Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.tight_layout()
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error creating cost history plot: {e}")
    finally:
        plt.close()  # Ensure figure is closed to free memory


def train_model(X_train, y_train, learning_rate=0.01, n_iterations=1000, tol=1e-4, patience=5):
    """
    Train the linear regression model with early stopping
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Maximum number of iterations for gradient descent
    tol : float
        Tolerance for early stopping
    patience : int
        Number of iterations to wait for improvement before early stopping
        
    Returns:
    --------
    model : LinearRegression
        Trained model
    """
    start_time = time.time()
    
    try:
        # Initialize and train the model
        model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit(X_train, y_train)
        
        # Early stopping logic
        best_cost = float('inf')
        patience_counter = 0
        
        for i in range(n_iterations):
            y_pred = model.predict(X_train)
            current_cost = model._compute_cost(y_train, y_pred)
            model.cost_history.append(current_cost)
            
            if current_cost < best_cost - tol:
                best_cost = current_cost
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at iteration {i+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model, save_path='linear_regression_model.joblib'):
    """
    Save the trained model using joblib
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    save_path : str
        Path to save the model
    """
    try:
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        joblib.dump(model, save_path)
        logger.info(f"Model saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a linear regression model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--n_iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=1, help='Number of features')
    parser.add_argument('--noise', type=float, default=20, help='Noise level')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--data_path', type=str, default=None, help='Path to custom dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--early_stopping_tol', type=float, default=1e-4, help='Tolerance for early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()
    
    try:
        # Validate parameters
        validate_parameters(args)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise
        
        logger.info(f"Starting model training with max {args.n_iterations} iterations")
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            random_state=args.random_state,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        # Train the model with early stopping
        model = train_model(
            X_train, 
            y_train, 
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            tol=args.early_stopping_tol,
            patience=args.early_stopping_patience
        )
        
        if model is None:
            logger.error("Model training failed")
            return
        
        # Plot cost history
        plot_cost_history(model.cost_history, save_path=output_dir / 'cost_history.png')
        
        # Evaluate on training data
        train_score = model.score(X_train, y_train)
        logger.info(f"Training R² Score: {train_score:.4f}")
        
        # Evaluate on test data
        test_score = model.score(X_test, y_test)
        logger.info(f"Test R² Score: {test_score:.4f}")
        
        # Save the model
        model_path = output_dir / 'linear_regression_model.joblib'
        try:
            save_model(model, save_path=model_path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        # Save test data for later evaluation
        try:
            np.save(output_dir / 'X_test.npy', X_test)
            np.save(output_dir / 'y_test.npy', y_test)
            logger.info(f"Test data saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving test data: {e}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()
