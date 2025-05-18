import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
import os
import time
import logging
from pathlib import Path

#local models

from linear_regression.model import LinearRegression
from utils.metrics import mean_squared_error , mean_absolute_error ,r2_score


#Set up logging

logging.basicConfig(
    level= logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import our model from model.py 

def load_model(model_path = 'output/linear_regression_model.joblib'):
    """
    Load a trained model using joblib
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    model : LinearRegression
        Loaded model
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found : {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model : {e}")
        raise

def load_test_data(X_test_path: str = 'output/X_test.npy', y_test_path: str = 'output/y_test.npy') -> tuple:
    """
    Load test data with error handling
    
    Parameters:
    -----------
    X_test_path : str
        Path to X test data
    y_test_path : str
        Path to y test data
        
    Returns:
    --------
    tuple
        (X_test, y_test)
    """
    try:
        logger.info(f"Loading test data from {X_test_path} and {y_test_path}")
        X_test = np.load(X_test_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)
        
        # Ensure y is flat (following sklearn convention)
        if y_test.ndim > 1:
            y_test = y_test.ravel()
            
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def evaluate_model(model, X_test :np.ndarray, y_test :np.ndarray) -> dict : 
    """
    Evaluate the model on test data
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test targets
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    start_time = time.time()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate model size
    model_size = 0
    if hasattr(model, 'weights'):
        model_size += model.weights.size * model.weights.itemsize
    if hasattr(model, 'bias'):
        model_size += 8  # Assuming bias is a float64
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'inference_time': inference_time,
        'model_size': model_size
    }
def plot_predictions(X_test :np.ndarray , y_test : np.ndarray ,y_pred :np.ndarray ,save_path='output/predictions.png'):
    # If we have more than 1 feature, only use the first one for plotting
    if X_test.shape[1] > 1:
        X_plot = X_test[:, 0]
    else:
        X_plot = X_test.flatten()
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(X_plot, y_test, color='blue', label='Actual')
    plt.scatter(X_plot, y_pred, color='red', alpha=0.5, label='Predicted')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_test, y_pred, save_path='output/residuals.png'):
    """
    Plot residuals to check for patterns
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted values
    save_path : str
        Path to save the plot
    """
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(y_pred, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def main():
    #parse arguments 
    parser = argparse.ArgumentParser(description='Test a trained linear regression model')
    parser.add_argument('--model_path', type=str, default='output/linear_regression_model.joblib', 
                        help='Path to the saved model')
    parser.add_argument('--X_test_path', type=str, default='output/X_test.npy',
                        help='Path to X test data')
    parser.add_argument('--y_test_path', type=str, default='output/y_test.npy',
                        help='Path to y test data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    args = parser.parse_args()

    try :
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Starting model evaluation")
        
        # Load model and test data
        model = load_model(args.model_path)
        X_test, y_test = load_test_data(args.X_test_path, args.y_test_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Print metrics
        logger.info("\nModel Evaluation Metrics:")
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        logger.info(f"Inference Time: {metrics['inference_time']:.6f} seconds")
        logger.info(f"Model Size: {metrics['model_size'] / 1024.0:.2f} KB")
        
        # If we have weights, print them
        if hasattr(model, 'weights'):
            logger.info("\nModel Parameters:")
            logger.info(f"Weights: {model.weights}")
            logger.info(f"Bias: {model.bias}")
        
        # Create visualizations
        plot_predictions(X_test, y_test, y_pred, save_path=output_dir / 'predictions.png')
        plot_residuals(y_test, y_pred, save_path=output_dir / 'residuals.png')
        
        logger.info("\nVisualization plots saved to output directory")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

if __name__ == '__main__':
    main()
