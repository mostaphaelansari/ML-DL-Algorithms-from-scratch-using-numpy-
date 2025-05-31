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
    if not 0 < args.test_size < 1:
        raise ValueError("Test size must be between 0 and 1")


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
        (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        if use_sklearn or data_path is None:
            # Generate synthetic data
            logger.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
            X, y = make_regression(n_samples=n_samples, 
                                n_features=n_features, 
                                noise=noise, 
                                random_state=random_state)
            y = y.ravel()
        else:
            # Load custom dataset
            logger.info(f"Loading data from {data_path}")
            try:
                data = np.loadtxt(data_path, delimiter=',')
                X = data[:, :-1]
                y = data[:, -1].ravel()
                logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
            except Exception as e:
                logger.error(f"Error loading data from {data_path}: {e}")
                logger.warning("Falling back to synthetic data")
                X, y = make_regression(n_samples=n_samples, 
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_state)
                y = y.ravel()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
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
        
        return X_train, X_test, y_train, y_test, scaler
        
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
        plt.plot(range(len(cost_history)), cost_history, linewidth=2)
        plt.title('Training Cost History', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cost history plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error creating cost history plot: {e}")
    finally:
        plt.close()  # Ensure figure is closed to free memory


def train_model_with_early_stopping(X_train, y_train, X_val, y_val, learning_rate=0.01, 
                                  n_iterations=1000, tol=1e-6, patience=10):
    """
    Train the linear regression model with proper early stopping using validation set
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation targets
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
        # Create a custom LinearRegression class that allows step-by-step training
        class LinearRegressionWithEarlyStopping(LinearRegression):
            def fit_with_early_stopping(self, X_train, y_train, X_val, y_val, tol, patience):
                # Initialize parameters
                n_samples, n_features = X_train.shape
                self.weights = np.zeros(n_features)
                self.bias = 0
                self.cost_history = []
                
                best_val_cost = float('inf')
                patience_counter = 0
                best_weights = None
                best_bias = None
                
                for i in range(self.n_iterations):
                    # Forward pass
                    y_pred_train = self.predict(X_train)
                    
                    # Compute gradients
                    dw = (1/n_samples) * np.dot(X_train.T, (y_pred_train - y_train))
                    db = (1/n_samples) * np.sum(y_pred_train - y_train)
                    
                    # Update parameters
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                    
                    # Compute validation cost
                    y_pred_val = self.predict(X_val)
                    val_cost = self._compute_cost(y_val, y_pred_val)
                    self.cost_history.append(val_cost)
                    
                    # Early stopping check
                    if val_cost < best_val_cost - tol:
                        best_val_cost = val_cost
                        best_weights = self.weights.copy()
                        best_bias = self.bias
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at iteration {i+1}")
                        # Restore best parameters
                        if best_weights is not None:
                            self.weights = best_weights
                            self.bias = best_bias
                        break
                
                self.is_fitted = True
                return self
        
        # Initialize and train the model
        model = LinearRegressionWithEarlyStopping(learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit_with_early_stopping(X_train, y_train, X_val, y_val, tol, patience)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final validation cost: {model.cost_history[-1]:.6f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def train_model_standard(X_train, y_train, learning_rate=0.01, n_iterations=1000):
    """
    Train the linear regression model without early stopping
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of iterations for gradient descent
        
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
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training cost: {model.cost_history[-1]:.6f}")
        
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
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--data_path', type=str, default=None, help='Path to custom dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--early_stopping_tol', type=float, default=1e-6, help='Tolerance for early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size (from training set)')
    
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
        logger.info(f"Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
        
        # Load data
        X_train, X_test, y_train, y_test, scaler = load_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            test_size=args.test_size,
            random_state=args.random_state,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        # Train the model
        if args.early_stopping:
            # Split training data into train/validation for early stopping
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=args.val_size, random_state=args.random_state
            )
            
            model = train_model_with_early_stopping(
                X_train_fit, y_train_fit, X_val, y_val,
                learning_rate=args.learning_rate,
                n_iterations=args.n_iterations,
                tol=args.early_stopping_tol,
                patience=args.early_stopping_patience
            )
        else:
            model = train_model_standard(
                X_train, y_train,
                learning_rate=args.learning_rate,
                n_iterations=args.n_iterations
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
        
        # Check for overfitting
        if train_score - test_score > 0.1:
            logger.warning(f"Potential overfitting detected! Training R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
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
        
        # Print model parameters
        params = model.get_params()
        logger.info(f"Model weights: {params['weights']}")
        logger.info(f"Model bias: {params['bias']}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == '__main__':
    main()
