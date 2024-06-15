"""
Linear regression is a prediction algorithm for estimating continuous variable which works by fitting a linear model to the training data.
"""

import numpy as np
from .feature_scaler import meanNormScaler,zScoreScaler
from .gradient_descent import batch_multi_linear

class linearRegression:
    def __init__(self):
        # Initialize weights and bias
        self.w = None
        self.b = None
        self.scaler = None
    
    def fit(self, x: np.ndarray, y: np.ndarray, scaler:str = 'zscore') -> None:
        """
        Fit model to the training data

        Args:
        x: Array of predictor variable values.
        y: Vector of target variable values.
        scaler: Type of scaler to use. Default is 'zscore'. Alternative is 'mean_norm'
        """
        self.scaler = meanNormScaler() if scaler == 'mean_norm' else zScoreScaler()
        # Feature scaling
        x_scaled = self.scaler.fit_transform(x)
        # Use gradient descent to fit w and b
        self.w, self.b = batch_multi_linear(x_scaled, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable values for the given predictor variable values.

        Args:
        x: Array of predictor variable values

        Returns:
        y: Vector of target variable values
        """
        assert self.w is not None and self.b is not None and self.scaler is not None, "Model needs to be trained first"
        # Scale the input before feeding to model
        x_scaled = self.scaler.transform(x)
        # Predict using linear regression equation
        y = x_scaled @ self.w + self.b
        return y