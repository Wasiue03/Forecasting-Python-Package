# tests/test_forecast.py
import pytest
import pandas as pd
from Forecasting.forecast import forecast_arima, forecast_sarima, forecast_exponential_smoothing

@pytest.fixture
def time_series_data():
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def test_forecast_arima_no_optimize(time_series_data):
    forecast = forecast_arima(time_series_data, order=(1, 1, 1), steps=3, optimize=False)
    assert len(forecast) == 3

def test_forecast_arima_with_optimize(time_series_data):
    forecast = forecast_arima(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3

def test_forecast_sarima_no_optimize(time_series_data):
    forecast = forecast_sarima(time_series_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), steps=3, optimize=False)
    assert len(forecast) == 3

def test_forecast_sarima_with_optimize(time_series_data):
    forecast = forecast_sarima(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3

def test_forecast_exponential_smoothing_no_optimize(time_series_data):
    forecast = forecast_exponential_smoothing(time_series_data, seasonal='add', seasonal_periods=12, steps=3, optimize=False)
    assert len(forecast) == 3

def test_forecast_exponential_smoothing_with_optimize(time_series_data):
    forecast = forecast_exponential_smoothing(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3
