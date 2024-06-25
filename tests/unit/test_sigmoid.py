import pytest
from src.utils.sigmoid import sigmoid

def test_sigmoid():
    assert sigmoid(0) == 0.5


if __name__ == "__main__":
    pytest.main()