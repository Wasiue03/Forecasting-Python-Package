import unittest
import os
from Forecasting.forecast import main_forecasting, load_dataset, forecast_arima, plot_forecast, forecast_sarima
import pandas as pd
class TestForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_file_path = 'Forecasting/data/metrics.csv'
        cls.column_name = 'value'

        if not os.path.exists(cls.temp_file_path):
            raise FileNotFoundError(f"Dataset file '{cls.temp_file_path}' not found.")

    def test_forecast_arima(self):
        forecast_series = main_forecasting(
            file_path=self.temp_file_path,
            column_name=self.column_name,
            model_type='arima',
            steps=10,
            optimize=True,
            plot=True,  # Set to True to display plot
            plot_path=None  # None to always show plot
        )
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")

    def test_forecast_sarima(self):
        forecast_series = main_forecasting(
            file_path=self.temp_file_path,
            column_name=self.column_name,
            model_type='sarima',
            steps=10,
            optimize=True,
            plot=True,  # Set to True to display plot
            plot_path=None  # None to always show plot
        )
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")

    def test_forecast_exponential_smoothing(self):
        forecast_series = main_forecasting(
            file_path=self.temp_file_path,
            column_name=self.column_name,
            model_type='exponential_smoothing',
            steps=10,
            optimize=True,
            plot=True,  # Set to True to display plot
            plot_path=None  # None to always show plot
        )
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")
    
    def test_load_dataset_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset('non_existent_file.csv', self.column_name)

    def test_load_dataset_column_not_found(self):
        with self.assertRaises(ValueError):
            load_dataset(self.temp_file_path, 'non_existent_column')

    def test_forecast_arima_no_optimization(self):
        forecast_series = main_forecasting(
            file_path=self.temp_file_path,
            column_name=self.column_name,
            model_type='arima',
            steps=10,
            optimize=False,
            plot=False
        )
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")
        self.assertEqual(len(forecast_series), 10, "Forecast series length should be equal to steps.")

    def test_forecast_arima_invalid_parameters(self):
        # Modify ARIMA parameters to test invalid cases (e.g., too short series)
        short_series = pd.Series(range(5))
        forecast_series = forecast_arima(short_series, steps=2, optimize=False)
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")


    def test_forecast_sarima_no_optimization(self):
        forecast_series = main_forecasting(
            file_path=self.temp_file_path,
            column_name=self.column_name,
            model_type='sarima',
            steps=10,
            optimize=False,
            plot=False
        )
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")
        self.assertEqual(len(forecast_series), 10, "Forecast series length should be equal to steps.")

    def test_forecast_sarima_invalid_parameters(self):
        # Modify SARIMA parameters to test invalid cases (e.g., too short series)
        short_series = pd.Series(range(5))
        forecast_series = forecast_sarima(short_series, steps=2, optimize=False)
        self.assertIsNotNone(forecast_series, "Forecast series should not be None.")



    
if __name__ == '__main__':
    unittest.main()
