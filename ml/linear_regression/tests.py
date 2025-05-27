"""
Comprehensive test suite for LinearRegression model.

This file contains unit tests, integration tests, and performance evaluation
for the LinearRegression implementation.
"""
import sys
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

    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.n_iterations, 1000)
        self.assertIsNone(self.model.weights)
        self.assertIsNone(self.model.bias)
        self.assertEqual(len(self.model.cost_history), 0)

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters"""
        model = LinearRegression(learning_rate=0.001, n_iterations=500)
        self.assertEqual(model.learning_rate, 0.001)
        self.assertEqual(model.n_iterations, 500)

    def test_fit_simple_data(self):
        """Test fitting on simple data"""
        self.model.fit(self.X_simple, self.y_simple)
        
        # Check that weights and bias are initialized
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertEqual(len(self.model.weights), self.X_simple.shape[1])
        
        # Check that cost history is populated
        self.assertGreater(len(self.model.cost_history), 0)
        self.assertEqual(len(self.model.cost_history), self.model.n_iterations)

    def test_fit_complex_data(self):
        """Test fitting on complex data"""
        self.model.fit(self.X_complex, self.y_complex)
        
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertEqual(len(self.model.weights), self.X_complex.shape[1])

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting"""
        with self.assertRaises((ValueError, AttributeError, TypeError)):
            self.model.predict(self.X_simple)

    def test_predict_after_fit(self):
        """Test prediction after fitting"""
        self.model.fit(self.X_simple, self.y_simple)
        predictions = self.model.predict(self.X_simple)
        
        self.assertEqual(len(predictions), len(self.y_simple))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_single_sample(self):
        """Test prediction on single sample"""
        self.model.fit(self.X_simple, self.y_simple)
        single_sample = self.X_simple[0:1]  # Keep 2D shape
        prediction = self.model.predict(single_sample)
        
        self.assertEqual(len(prediction), 1)

    def test_cost_decreasing(self):
        """Test that cost generally decreases during training"""
        self.model.fit(self.X_complex, self.y_complex)
        
        # Cost should generally decrease (allow some fluctuation)
        initial_cost = np.mean(self.model.cost_history[:10])
        final_cost = np.mean(self.model.cost_history[-10:])
        self.assertLess(final_cost, initial_cost)

    def test_input_validation(self):
        """Test input validation"""
        # Test mismatched dimensions
        X_wrong = np.array([[1, 2], [3, 4]])
        y_wrong = np.array([1, 2, 3])  # Different length
        
        with self.assertRaises((ValueError, AssertionError)):
            self.model.fit(X_wrong, y_wrong)

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        
        with self.assertRaises((ValueError, AssertionError, ZeroDivisionError)):
            self.model.fit(X_empty, y_empty)

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        np.random.seed(42)
        X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
        
        model1 = LinearRegression(learning_rate=0.01, n_iterations=500)
        model2 = LinearRegression(learning_rate=0.01, n_iterations=500)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        np.testing.assert_array_almost_equal(model1.weights, model2.weights, decimal=10)
        self.assertAlmostEqual(model1.bias, model2.bias, places=10)


class TestLinearRegressionIntegration(unittest.TestCase):
    """Integration tests comparing with sklearn"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.X_test, self.y_test = make_regression(
            n_samples=100, n_features=3, n_informative=2, noise=10, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_test, self.y_test, test_size=0.3, random_state=42
        )

    def test_comparison_with_sklearn(self):
        """Compare results with sklearn LinearRegression"""
        # Train custom model
        custom_model = LinearRegression(learning_rate=0.01, n_iterations=5000)
        custom_model.fit(self.X_train, self.y_train)
        custom_predictions = custom_model.predict(self.X_val)
        
        # Train sklearn model
        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(self.X_train, self.y_train)
        sklearn_predictions = sklearn_model.predict(self.X_val)
        
        # Compare R² scores (should be reasonably close)
        custom_r2 = r2_score(self.y_val, custom_predictions)
        sklearn_r2 = r2_score(self.y_val, sklearn_predictions)
        
        # Allow some tolerance due to different optimization methods
        self.assertGreater(custom_r2, 0.8)  # Should achieve reasonable fit
        self.assertLess(abs(custom_r2 - sklearn_r2), 0.2)  # Should be reasonably close

    def test_performance_on_normalized_data(self):
        """Test performance on normalized data"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        model = LinearRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X_scaled, self.y_train)
        predictions = model.predict(X_val_scaled)
        
        r2 = r2_score(self.y_val, predictions)
        self.assertGreater(r2, 0.8)  # Should perform well on normalized data

    def test_different_learning_rates(self):
        """Test model with different learning rates"""
        learning_rates = [0.001, 0.01, 0.1]
        scores = []
        
        for lr in learning_rates:
            model = LinearRegression(learning_rate=lr, n_iterations=2000)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_val)
            score = r2_score(self.y_val, predictions)
            scores.append(score)
        
        # At least one learning rate should achieve decent performance
        self.assertGreater(max(scores), 0.7)

    def test_convergence_with_different_iterations(self):
        """Test convergence with different iteration counts"""
        iterations = [100, 500, 1000, 2000]
        final_costs = []
        
        for n_iter in iterations:
            model = LinearRegression(learning_rate=0.01, n_iterations=n_iter)
            model.fit(self.X_train, self.y_train)
            final_costs.append(model.cost_history[-1])
        
        # Cost should generally decrease with more iterations
        # (though it might plateau)
        self.assertLessEqual(final_costs[-1], final_costs[0])


class TestLinearRegressionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_perfect_linear_relationship(self):
        """Test on data with perfect linear relationship"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2*x
        
        model = LinearRegression(learning_rate=0.01, n_iterations=2000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Should achieve very high R²
        r2 = r2_score(y, predictions)
        self.assertGreater(r2, 0.99)

    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.randn(50, 1)
        y = 3 * X.ravel() + np.random.randn(50) * 0.1
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        self.assertEqual(len(model.weights), 1)
        self.assertGreater(r2_score(y, predictions), 0.8)

    def test_many_features(self):
        """Test with many features"""
        X, y = make_regression(n_samples=100, n_features=10, n_informative=5, 
                              noise=0.1, random_state=42)
        
        model = LinearRegression(learning_rate=0.01, n_iterations=2000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        self.assertEqual(len(model.weights), 10)
        self.assertGreater(r2_score(y, predictions), 0.8)

    def test_constant_target(self):
        """Test with constant target values"""
        X = np.random.randn(50, 2)
        y = np.ones(50) * 5  # Constant target
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Predictions should be close to the constant value
        np.testing.assert_allclose(predictions, 5, atol=0.5)

    def test_large_learning_rate_stability(self):
        """Test that large learning rates don't cause instability"""
        X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
        
        # Normalize data to prevent instability
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()
        
        model = LinearRegression(learning_rate=1.0, n_iterations=500)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        # Check that weights and cost are finite
        self.assertTrue(np.all(np.isfinite(model.weights)))
        self.assertTrue(np.isfinite(model.bias))
        self.assertTrue(np.all(np.isfinite(model.cost_history)))


class TestLinearRegressionPerformance(unittest.TestCase):
    """Performance and benchmarking tests"""

    def test_training_time_reasonable(self):
        """Test that training completes in reasonable time"""
        import time
        
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        
        training_time = end_time - start_time
        self.assertLess(training_time, 10)  # Should complete within 10 seconds

    def test_memory_efficiency(self):
        """Test memory usage with large datasets"""
        # Create a moderately large dataset
        X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
        
        model = LinearRegression(learning_rate=0.01, n_iterations=500)
        
        # This should not raise memory errors
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            self.assertEqual(len(predictions), len(y))
        except MemoryError:
            self.fail("Model should handle moderately large datasets without memory errors")

    def test_scalability(self):
        """Test model performance on datasets of different sizes"""
        sizes = [100, 500, 1000]
        training_times = []
        
        import time
        
        for size in sizes:
            X, y = make_regression(n_samples=size, n_features=5, noise=0.1, random_state=42)
            model = LinearRegression(learning_rate=0.01, n_iterations=500)
            
            start_time = time.time()
            model.fit(X, y)
            end_time = time.time()
            
            training_times.append(end_time - start_time)
        
        # Training time should scale reasonably (not exponentially)
        # Allow for some variation in timing
        time_ratio = training_times[-1] / training_times[0]
        size_ratio = sizes[-1] / sizes[0]
        
        # Training time should not grow much faster than dataset size
        self.assertLess(time_ratio, size_ratio * 2)


def create_performance_report():
    """Create a comprehensive performance report"""
    print("=" * 60)
    print("LINEAR REGRESSION PERFORMANCE REPORT")
    print("=" * 60)
    
    # Test 1: Comparison with sklearn
    print("\n1. Comparison with scikit-learn:")
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Custom model
    custom_model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    custom_model.fit(X_train, y_train)
    custom_pred = custom_model.predict(X_test)
    custom_r2 = r2_score(y_test, custom_pred)
    custom_mse = mean_squared_error(y_test, custom_pred)
    
    # Sklearn model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    print(f"   Custom Model  - R²: {custom_r2:.4f}, MSE: {custom_mse:.2f}")
    print(f"   Sklearn Model - R²: {sklearn_r2:.4f}, MSE: {sklearn_mse:.2f}")
    print(f"   Difference    - R²: {abs(custom_r2 - sklearn_r2):.4f}, MSE: {abs(custom_mse - sklearn_mse):.2f}")
    
    # Test 2: Learning curve analysis
    print("\n2. Learning Curve Analysis:")
    iterations = [100, 500, 1000, 2000, 5000]
    for n_iter in iterations:
        model = LinearRegression(learning_rate=0.01, n_iterations=n_iter)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"   {n_iter:4d} iterations - R²: {r2:.4f}")
    
    # Test 3: Learning rate sensitivity
    print("\n3. Learning Rate Sensitivity:")
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, n_iterations=1000)
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            print(f"   LR: {lr:5.3f} - R²: {r2:.4f}")
        except:
            print(f"   LR: {lr:5.3f} - Failed (likely unstable)")
    
    print("\n" + "=" * 60)


def plot_training_progress(model, title="Training Progress"):
    """Plot the training progress"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.cost_history)
    plt.title(f'{title} - Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history[50:])  # Skip first 50 iterations for better view
    plt.title(f'{title} - Cost Function (After 50 iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Run all tests
    print("Running comprehensive Linear Regression test suite...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLinearRegressionLogic,
        TestLinearRegressionIntegration,
        TestLinearRegressionEdgeCases,
        TestLinearRegressionPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Generate performance report if all tests pass
    if not result.failures and not result.errors:
        print(f"\nAll tests passed! Generating performance report...\n")
        try:
            create_performance_report()
        except Exception as e:
            print(f"Could not generate performance report: {e}")
    else:
        print(f"\nSome tests failed. Fix issues before running performance report.")
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}")
