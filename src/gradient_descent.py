"""
Gradient descent is an optimization algorithm which can be used to reach local minima for any function by using partial derivatives and iterative updates.
Let's define some notations:
*N* - Number of training samples
*B* - Batch Size for Stochastic Gradient Descent (equal to N in case of batch gradient descent)
*D* - Number of features per training sample
*P* - Number of parameters or weights in the model function
*I* - Number of iterations

**Space Complexity**:
Space for storing parameter values: O(P)
Space for storing feature data: O(BD)
Space for storing partial derivatives: O(P)
Total Space Complexity = O(P+BD)
**Time Complexity**:
Each iteration will update values of P, by going through B training samples, i.e. BP
Total Time Complexity = O(IBP)
"""
import argparse
import sys
import numpy as np

def batch_simple_linear(x, y, max_iterations=1000, learning_rate=0.001, stopping_threshold=1e-6):
    """
    Performs batch gradient descent using a simple linear regression model, where
    y = wx + b,
    and we use mean squared error as the cost function
    J = 1/m * (Y_pred - Y)**2
    The function returns the slope and intercept for the fitted line that minimizes loss

    Args:
    x: Vector of predictor variable values.
    y: Vector of target variable values.
    max_iterations: Maximum number of iterations.
    stopping_threshold: Threshold to consider the change in W and b as insignificant between subsequent iterations.

    Returns:
    w: slope of the fitted line
    b: intercept of the fitted line
    """
    # Initialize w and b
    w, b = 0, 0
    print(f"Started with w = {w} and b = {b}")

    # Number of training samples
    m = len(x)

    for i in range(max_iterations):
        # Predict y using current values of w and b
        y_pred = w * x + b

        # Compute cost as mean squared error between predicted and actual y-values.
        cost = np.mean((y_pred - y)**2)

        # Compute gradients
        dw = 2 * np.mean((y_pred - y) * x)
        db = 2 * np.mean((y_pred - y))

        # Update params
        w -= learning_rate * dw
        b -= learning_rate * db

        # Check if the updates are insignificant
        if max(abs(learning_rate * dw), abs(learning_rate * db)) < stopping_threshold:
            break
    
    print(f"Ran {i+1} iterations")
    return w, b

def batch_multi_linear(x, y, max_iterations=1000, learning_rate=0.001, stopping_threshold=1e-6):
    """
    Performs batch gradient descent using a multiple linear regression model, where
    y = WX + b, where W is a vector of shape (D,1) and X is an array of shape (m, D)
    and we use mean squared error as the cost function
    J = 1/m * (y_pred - y)**2
    The function returns the weight vector and bias for the fitted multi-dimensional plane that minimizes loss

    Args:
    x: Multi-dimensional array of predictor variable values with each row representing one training sample.
    y: Vector of target variable values.
    max_iterations: Maximum number of iterations.
    stopping_threshold: Threshold to consider the change in W and b as insignificant between subsequent iterations.

    Returns:
    W: Vector of parameter weights
    b: Bias term
    """
    # Initialize w and b
    w, b = np.zeros((x.shape[1],1)), 0 # we should have as many weights as features
    print(f"Started with w = {w} and b = {b}")

    # Number of training samples
    m = len(x)

    for i in range(max_iterations):
        # Predict y using current values of w and b
        # Use dot product as both w and x are arrays
        y_pred = x @ w + b # Note the order as x is shape (m, D) and w is shape (D, 1)
        # Now shape of y_pred is (m,1)

        # Compute cost as mean squared error between predicted and actual y-values.
        cost = np.mean((y_pred - y)**2)

        # Compute gradients
        dw = (2/m) * (x.T @ (y_pred - y)) # Use dot product to make dw of shape (D, 1)
        db = 2 * np.mean((y_pred - y))

        # Update params
        w -= learning_rate * dw # This automatically updates all weights in the vector
        b -= learning_rate * db

        # Check if the updates are insignificant
        if max(max(abs(learning_rate * dw)), abs(learning_rate * db)) < stopping_threshold:
            break
    
    print(f"Ran {i+1} iterations")
    return w, b

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--variant', type=str, default='batch_simple_linear',
                             help=
                             """
                             Variant of gradient descent to use: 
                             batch_simple_linear = Batch mode, single variable, linear regression
                             batch_multi_linear = Batch mode, multiple input variables, linear regression
                             """)
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help=
                            """
                            Specify the learning rate or the step size for gradient update
                            - Too big learning rate can cause convergence issues
                            - Too small learning rate can cause long convergence times
                            """)
        parser.add_argument('--max_iterations', type=int, default=1000,
                            help=
                            """
                            Maximum number of iterations to run even if the gradient updates are significant
                            """)
        parser.add_argument('--stopping_threshold', type=float, default=1e-6,
                            help=
                            """
                            Define the minimum delta which is considered a significant update in gradient between subsequent iterations
                            """)
        args = parser.parse_args()
    
        if args.variant == 'batch_simple_linear':
            x = np.array([4, 8, 12, 16])
            w_true = 0.50
            b_true = 0.05
            y = w_true * x + b_true
            w, b = batch_simple_linear(x, y, learning_rate=args.learning_rate, max_iterations=args.max_iterations, stopping_threshold=args.stopping_threshold)
            print(f"Slope of the fitted line = {w}, Actual Slope = {w_true}")
            print(f"Intercept of the fitted line = {b}, Actual Intercept = {b_true}")
        elif args.variant == 'batch_multi_linear':
            x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
            w_true = np.array([[0.50],
                            [0.50]])
            b_true = 0.05
            y = x @ w_true + b_true # Using dot product
            w, b = batch_multi_linear(x, y, learning_rate=args.learning_rate, max_iterations=args.max_iterations, stopping_threshold=args.stopping_threshold)
            print(f"Weights of the fitted plane = {w}, Actual Weights = {w_true}")
            print(f"Bias term = {b}, Actual Bias = {b_true}")
        else:
            print("Invalid argument for --variant:", args.variant)
            sys.exit(1)