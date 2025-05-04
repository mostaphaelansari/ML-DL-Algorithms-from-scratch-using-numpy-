import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import os
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our custom linear regression model
from model import LinearRegression

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
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Cost history plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error creating cost history plot: {e}")


def train_model(X_train, y_train, learning_rate=0.01, n_iterations=1000):
    """
    Train the linear regression model
    
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
    
    # Initialize and train the model
    model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

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
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Starting model training with {args.n_iterations} iterations")
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            random_state=args.random_state,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        # Train the model
        model = train_model(
            X_train, 
            y_train, 
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations
        )
        
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
        save_model(model, save_path=model_path)
        
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
