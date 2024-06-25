import numpy as np
import pytest
from src.logistic_regression import logisticRegression

def test_logistic_regression_fit():
    x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
    y = np.array([[0],
                [0],
                [1],
                [1]])
    
    # Fit the linear regression model
    lr = logisticRegression()
    lr.fit(x, y)
    assert lr.w is not None and lr.b is not None

def test_logistic_regression_predict():
    x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
    y = np.array([[0],
                [0],
                [1],
                [1]])
    x_test = np.array([[1, 1],
                  [20, 20]])
    # Manually assign w and b to the linear regression model
    lr = logisticRegression()
    lr.fit(x,y)
    y_pred = lr.predict(x_test)
    np.testing.assert_array_equal(y_pred, np.array([[0],[1]]))

if __name__ == "__main__":
    pytest.main()