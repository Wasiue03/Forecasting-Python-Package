import pytest
import pandas as pd
from Forecasting.decompose import decompose_series

@pytest.fixture
def time_series_data():
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def test_decompose_series(time_series_data):
    decomposition = decompose_series(time_series_data, model='add', period=4)
    assert decomposition.trend is not None
    assert decomposition.seasonal is not None
    assert decomposition.resid is not None
