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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression.model import LogisticRegression


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
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=2,
        flip_y=noise,
        weights=[.25, .75],
        random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    return X_train, X_val, y_train, y_val, scaler


def train_model(args):
    """
    Train the logistic regression model with early stopping
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing training parameters
    
    Returns:
    --------
    model : LogisticRegression
        Best trained logistic regression model
    """
    validate_parameters(args)
    X_train, X_val, y_train, y_val, scaler = load_data(args.n_samples, args.n_features, args.noise)

    # Initialize model
    model = LogisticRegression(learning_rate=args.learning_rate, num_iterations=1)
    
    # Initialize parameters for early stopping
    best_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    val_loss_history = []
    train_loss_history = []
    
    logger.info("Starting training with early stopping...")
    
    # Training loop with early stopping
    for epoch in range(args.n_iterations):
        # Train for one iteration
        model.num_iterations = 1
        model.fit(X_train, y_train)
        
        # Calculate losses
        train_pred = model.predict_proba(X_train)
        val_pred = model.predict_proba(X_val)
        
        train_loss = model.compute_loss(y_train, train_pred)
        val_loss = model.compute_loss(y_val, val_pred)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # Log progress every 50 epochs
        if epoch % 50 == 0 or epoch < 10:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = LogisticRegression(learning_rate=args.learning_rate)
            best_model.weights = model.weights.copy()
            best_model.bias = model.bias
            best_model.cost_history = model.cost_history.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_loss:.4f}")
            break
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info("Best model and scaler saved to models/")
    
    # Plot training curves
    plot_training_curves(train_loss_history, val_loss_history)
    
    # Calculate final metrics
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    
    train_accuracy = np.mean(train_pred == y_train)
    val_accuracy = np.mean(val_pred == y_val)
    
    logger.info(f"Final Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"Final Validation Accuracy: {val_accuracy:.4f}")
    logger.info("Training complete.")
    
    return best_model


def plot_training_curves(train_loss_history, val_loss_history):
    """
    Plot training and validation loss curves
    
    Parameters:
    -----------
    train_loss_history : list
        History of training loss values
    val_loss_history : list
        History of validation loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Loss curves saved to plots/loss_curves.png")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train logistic regression with early stopping")
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for gradient descent (default: 0.01)')
    parser.add_argument('--n_iterations', type=int, default=1000,
                       help='Maximum number of training iterations (default: 1000)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--n_features', type=int, default=10,
                       help='Number of features (default: 10)')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Noise level for data generation (default: 0.01)')
    return parser.parse_args()


def main():
    """Main function to run the training pipeline"""
    try:
        args = parse_args()
        logger.info("Starting logistic regression training pipeline...")
        logger.info(f"Parameters: lr={args.learning_rate}, iterations={args.n_iterations}, "
                   f"samples={args.n_samples}, features={args.n_features}, noise={args.noise}")
        
        start_time = time.time()
        model = train_model(args)
        end_time = time.time()
        
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
