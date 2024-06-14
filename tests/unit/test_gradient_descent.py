import numpy as np
import pytest
from src.gradient_descent import batch_simple_linear, batch_multi_linear

def test_batch_simple_linear_output():
    x = np.array([4, 8, 12, 16])
    w_true = 0.50
    b_true = 0.05
    y = w_true * x + b_true
    w, b = batch_simple_linear(x, y, learning_rate=0.001)
    print(f"Slope of the fitted line = {w}, Actual Slope = {w_true}")
    print(f"Intercept of the fitted line = {b}, Actual Intercept = {b_true}")
    np.testing.assert_almost_equal(w_true, w, decimal=2)
    np.testing.assert_almost_equal(b_true, b, decimal=2)

def test_batch_multi_linear_output():
    x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
    w_true = np.array([[0.50],
                       [0.50]])
    b_true = 0.05
    y = x @ w_true + b_true  # Using dot product
    w, b = batch_multi_linear(x, y)
    print(f"Weights of the fitted plane = {w}, Actual Weights = {w_true}")
    print(f"Bias term = {b}, Actual Bias = {b_true}")
    np.testing.assert_almost_equal(w_true, w, decimal=2)
    np.testing.assert_almost_equal(b_true, b, decimal=2)

if __name__ == '__main__':
    pytest.main()