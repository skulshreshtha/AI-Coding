"""
Logistic Regression is a binary classification algorithm for predicting the class of a categorical variable which works by fitting a decision boundary to segregate the positive and negative classes.
"""

import numpy as np
from .feature_scaler import meanNormScaler,zScoreScaler
from .gradient_descent import batch_multi_logistic
from .utils.sigmoid import sigmoid

class logisticRegression:
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
        self.w, self.b = batch_multi_logistic(x_scaled, y)

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
        y_prob = sigmoid(x_scaled @ self.w + self.b)
        y = np.where(y_prob>0.5, 1, 0)
        return y