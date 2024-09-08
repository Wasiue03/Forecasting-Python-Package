import pytest
import pandas as pd
from Forecasting.forecast import forecast_arima, forecast_sarima, forecast_exponential_smoothing

@pytest.fixture
def time_series_data():
    # Provide some sample time series data, potentially with more observations
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

def test_forecast_arima_no_optimize(time_series_data):
    # Test ARIMA forecasting without optimization
    forecast = forecast_arima(time_series_data, steps=3, optimize=False)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'

def test_forecast_arima_with_optimize(time_series_data):
    # Test ARIMA forecasting with optimization
    forecast = forecast_arima(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'

def test_forecast_sarima_no_optimize(time_series_data):
    # Test SARIMA forecasting without optimization, avoiding conflicting MA lags
    forecast = forecast_sarima(time_series_data, steps=3, optimize=False)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'

def test_forecast_sarima_with_optimize(time_series_data):
    # Test SARIMA forecasting with optimization, avoiding conflicting MA lags
    forecast = forecast_sarima(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'

def test_forecast_exponential_smoothing_no_optimize(time_series_data):
    # Test Exponential Smoothing forecasting without optimization
    forecast = forecast_exponential_smoothing(time_series_data, steps=3, optimize=False)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'

def test_forecast_exponential_smoothing_with_optimize(time_series_data):
    # Test Exponential Smoothing forecasting with optimization
    forecast = forecast_exponential_smoothing(time_series_data, steps=3, optimize=True)
    assert len(forecast) == 3
    assert forecast.name == 'Forecast'
