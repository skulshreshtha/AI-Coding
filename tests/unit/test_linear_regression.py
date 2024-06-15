import numpy as np
import pytest
from src.linear_regression import linearRegression

def test_linear_regression(mocker):
    x = x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
    y = np.array([[ 4.05],
                [ 8.05],
                [12.05],
                [16.05]])
    mock_batch_multi_linear = mocker.patch('src.linear_regression.batch_multi_linear', return_value=(np.array([[0.50],
                   [0.50]]), 0.05))

    # Fit the linear regression model
    lr = linearRegression()
    lr.fit(x, y)
    mock_batch_multi_linear.assert_called_once()

if __name__ == "__main__":
    pytest.main()