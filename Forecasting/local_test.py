from Forecasting.forecast import main_forecasting

# Test the forecasting function
forecast_series = main_forecasting(
    file_path='Forecasting/data/metrics.csv',
    column_name='value',
    model_type='arima',
    steps=10,
    optimize=True,
    plot=True,
    plot_path='arima_forecast.png'
)

print(forecast_series)
