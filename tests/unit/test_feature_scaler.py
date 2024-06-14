import numpy as np
import pytest
from src.feature_scaler import meanNormScaler, zScoreScaler

def test_mean_normalization():
        scaler = meanNormScaler()
        x = np.array([[ 4,  4],
                    [ 8,  8],
                    [12, 12],
                    [16, 16]])
        scaler.fit(x)
        np.testing.assert_allclose(scaler.mu, np.array([10., 10.])) # type: ignore
        np.testing.assert_allclose(scaler.spread, np.array([12, 12])) # type: ignore
        x_scaled = scaler.transform(x)
        np.testing.assert_allclose(x_scaled, np.array([[-0.5, -0.5],
                                                        [-0.16666667, -0.16666667],
                                                        [ 0.16666667,  0.16666667],
                                                        [ 0.5, 0.5]])) # type: ignore
    
def test_z_score_normalization():
    scaler = zScoreScaler()
    x = np.array([[ 4,  4],
                [ 8,  8],
                [12, 12],
                [16, 16]])
    scaler.fit(x)
    np.testing.assert_allclose(scaler.mu, np.array([10., 10.])) # type: ignore
    np.testing.assert_allclose(scaler.stdev, np.array([4.47213595, 4.47213595])) # type: ignore
    x_scaled = scaler.transform(x)
    np.testing.assert_allclose(x_scaled, np.array([[-1.34164079, -1.34164079],
                                                    [-0.4472136 , -0.4472136 ],
                                                    [ 0.4472136 ,  0.4472136 ],
                                                    [ 1.34164079,  1.34164079]])) # type: ignore

if __name__ == "__main__":
        pytest.main()