import numpy as np
import pytest
from unittest.mock import MagicMock
from src.logistic_regression import logisticRegression
from src.feature_scaler import meanNormScaler

@pytest.fixture(scope='module')
def mocked_feature_scaler():
    return MagicMock(spec=meanNormScaler)

def test_logistic_regression_fit(mocker):
    x = np.array([[4, 4],
                  [8, 8],
                  [12, 12],
                  [16, 16]])
    y = np.array([[0],
                [0],
                [1],
                [1]])
    
    mock_batch_multi_logistic = mocker.patch('src.logistic_regression.batch_multi_logistic', return_value=(np.array([[0.50],
                   [0.50]]), 0.05))

    # Fit the linear regression model
    lr = logisticRegression()
    lr.fit(x, y)
    assert lr.w is not None and lr.b is not None
    mock_batch_multi_logistic.assert_called_once()

def test_logistic_regression_predict(mocker, mocked_feature_scaler):
    x = np.array([[1, 1],
                  [20, 20]])
    # Manually assign w and b to the linear regression model
    lr = logisticRegression()
    lr.w = np.array([[0.50],
                   [0.50]])
    lr.b = 0.05
    lr.scaler = mocked_feature_scaler
    mocked_feature_scaler.transform.return_value = np.array([[-0.75      , -0.75      ],
                                                            [ 0.83333333,  0.83333333]])
    mock_sigmoid = mocker.patch('src.logistic_regression.sigmoid', return_value=np.array([[0.2],
                                                                            [0.9]]))
    y = lr.predict(x)
    np.testing.assert_array_equal(y, np.array([[0],[1]]))
    mocked_feature_scaler.transform.assert_called_once_with(x)
    mock_sigmoid.assert_called_once()

if __name__ == "__main__":
    pytest.main()