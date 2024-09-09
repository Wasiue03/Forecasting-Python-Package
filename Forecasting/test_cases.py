import pytest
import pandas as pd
import os
from Forecasting.forecast import main_forecasting

# Sample data
test_csv = 'Forecasting/data/metrics.csv'
test_column = 'value'
data_directory = 'Forecasting/data'  # Ensure this directory exists

@pytest.fixture(scope='module', autouse=True)
def setup_module():
    # Create necessary directories if they do not exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    # Create a sample CSV file for testing
    data = {
        'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'value': [i + (i % 10) for i in range(100)]  # Sample time series data
    }
    df = pd.DataFrame(data)
    df.to_csv(test_csv, index=False)

def test_load_dataset():
    # Test if the dataset loads correctly
    series = main_forecasting(test_csv, test_column, model_type='arima', steps=1)
    assert not series.empty, "The series should not be empty."

def test_forecast_arima_no_optimize():
    # Test ARIMA forecasting without optimization
    series = main_forecasting(test_csv, test_column, model_type='arima', steps=5, optimize=False)
    assert len(series) == 5, "The forecast length should match the requested steps."

def test_forecast_arima_optimize():
    # Test ARIMA forecasting with optimization
    series = main_forecasting(test_csv, test_column, model_type='arima', steps=5, optimize=True)
    assert len(series) == 5, "The forecast length should match the requested steps."

def test_forecast_sarima_no_optimize():
    # Test SARIMA forecasting without optimization
    series = main_forecasting(test_csv, test_column, model_type='sarima', steps=5, optimize=False)
    assert len(series) == 5, "The forecast length should match the requested steps."

def test_forecast_sarima_optimize():
    # Test SARIMA forecasting with optimization
    series = main_forecasting(test_csv, test_column, model_type='sarima', steps=5, optimize=True)
    assert len(series) == 5, "The forecast length should match the requested steps."

def test_forecast_exponential_smoothing_no_optimize():
    # Test Exponential Smoothing forecasting without optimization
    series = main_forecasting(test_csv, test_column, model_type='exponential_smoothing', steps=5, optimize=False)
    assert len(series) == 5, "The forecast length should match the requested steps."

def test_forecast_exponential_smoothing_optimize():
    # Test Exponential Smoothing forecasting with optimization
    series = main_forecasting(test_csv, test_column, model_type='exponential_smoothing', steps=5, optimize=True)
    assert len(series) == 5, "The forecast length should match the requested steps."

# @pytest.fixture(scope='module', autouse=True)
# def teardown_module():
#     # Clean up the test CSV file
#     if os.path.exists(test_csv):
#         os.remove(test_csv)
#     # Clean up any created directories
#     if os.path.exists(data_directory):
#         os.rmdir(data_directory)
