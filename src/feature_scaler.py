"""
Feature Scaling is a technique used to bring features with differing ranges to a comparable range.
This makes it easier to compare feature importance, understand model weights, and converge gradient descent faster.
There are several techniques for feature scaling which are implemented below:
- Mean Normalization
- Z-score Normalization
For feature scaling, we will need to implement classes as the same scaler object that is used for training data shall be used for scaling the test/prediction data
"""

import numpy as np

class featureScaler:
    def fit(self):
        pass
    def transform(self):
        pass
    def fit_transform(self):
        pass

class meanNormScaler(featureScaler):
    
    def __init__(self) -> None:
        super().__init__()
        self.mu = None
        self.spread = None
    
    def fit(self, x: np.ndarray) -> None:
        # Set mu to the mean for each feature
        self.mu = np.mean(x, axis=0)
        # Set spread to the difference of min and max for each feature
        self.spread = np.max(x, axis=0) - np.min(x, axis=0)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.spread is not None, "Scaler needs to be fit first before running transform. Use fit->transform or fit_transform"
        return (x-self.mu)/self.spread
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
    
class zScoreScaler(featureScaler):

    def __init__(self) -> None:
        super().__init__()
        self.mu = None
        self.stdev = None
    
    def fit(self, x: np.ndarray) -> None:
        # Set mu to the mean for each feature
        self.mu = np.mean(x, axis=0)
        # Set std to the standard deviation for each feature
        self.stdev = np.std(x, axis=0)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.stdev is not None, "Scaler needs to be fit first before running transform. Use fit->transform or fit_transform"
        return (x-self.mu)/self.stdev
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)