import pandas as pd
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(file_path, column_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")
        
        series = df[column_name]
        return series
    except Exception as e:
        raise ValueError(f"Error loading the dataset: {e}")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_forecast(series, forecast_series, title='Forecast', forecast_label='Forecast', save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Actual Data')
    plt.plot(forecast_series.index, forecast_series, label=forecast_label, color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)  # Save the plot to a file
        plt.close()  # Close the figure to free up memory
    if not save_path or not os.path.exists(save_path):
        plt.show()  # Show the plot on screen if not saving or if save path is invalid


# ARIMA forecasting function with optimization using Optuna
def optimize_arima(series, steps=1):
    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)
        order = (p, d, q)
        train = series[:-steps]
        test = series[-steps:]
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return mean_squared_error(test, forecast)
        except Exception as e:
            return float('inf')  
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    best_order = (best_params['p'], best_params['d'], best_params['q'])
    return ARIMA(series, order=best_order).fit()

def forecast_arima(series, steps=1, optimize=False, plot=False, plot_path=None):
    if optimize:
        model_fit = optimize_arima(series, steps)
    else:
        model_fit = ARIMA(series, order=(1, 1, 1)).fit()
    
    forecast = model_fit.forecast(steps=steps)
    forecast_series = pd.Series(forecast, name='Forecast')
    
    if plot:
        plot_forecast(series, forecast_series, title='ARIMA Forecast', save_path=plot_path)

    return forecast_series

# SARIMA forecasting function with optimization
def optimize_sarima(series, steps=1):
    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)
        P = trial.suggest_int('P', 0, 5)
        D = trial.suggest_int('D', 0, 2)
        Q = trial.suggest_int('Q', 0, 5)
        s = trial.suggest_int('s', 4, 12)

        if q == Q:
            trial.set_user_attr("invalid", True)
            return float('inf') 
        
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        train = series[:-steps]
        test = series[-steps:]
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=steps)
            return mean_squared_error(test, forecast)
        except Exception as e:
            trial.set_user_attr("exception", str(e))
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    valid_trials = [trial for trial in study.trials if not trial.user_attrs.get("invalid", False)]
    if not valid_trials:
        raise ValueError("All trials were invalid due to overlapping MA lags.")
    
    best_trial = min(valid_trials, key=lambda t: t.value)
    best_params = best_trial.params
    best_order = (best_params['p'], best_params['d'], best_params['q'])
    best_seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
    
    return SARIMAX(series, order=best_order, seasonal_order=best_seasonal_order).fit()

def forecast_sarima(series, steps=1, optimize=False, plot=False, plot_path=None):
    if optimize:
        model_fit = optimize_sarima(series, steps)
    else:
        try:
            model_fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        except np.linalg.LinAlgError:
            print("SARIMA fitting error: Schur decomposition solver failed.")
            return pd.Series([None] * steps, name='Forecast')
    
    try:
        forecast = model_fit.forecast(steps=steps)
        forecast_series = pd.Series(forecast, name='Forecast')
        
        if plot:
            plot_forecast(series, forecast_series, title='SARIMA Forecast', save_path=plot_path)

        return forecast_series
    except IndexError as e:
        print(f"IndexError encountered during forecasting: {e}")
        return pd.Series([None] * steps, name='Forecast')

# Exponential Smoothing forecasting function with optimization
def optimize_exponential_smoothing(series, steps=1):
    def objective(trial):
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul'])
        seasonal_periods = trial.suggest_int('seasonal_periods', 2, 12)
        train = series[:-steps]
        test = series[-steps:]
        try:
            model = ExponentialSmoothing(train, seasonal=seasonal, seasonal_periods=seasonal_periods)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return mean_squared_error(test, forecast)
        except ValueError:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    return ExponentialSmoothing(series, seasonal=best_params['seasonal'], 
                                seasonal_periods=best_params['seasonal_periods']).fit()

def forecast_exponential_smoothing(series, steps=1, optimize=False, plot=False, plot_path=None):
    if optimize:
        model_fit = optimize_exponential_smoothing(series, steps)
    else:
        if len(series) < 24:
            print("Insufficient data for seasonal Exponential Smoothing.")
            return pd.Series([None] * steps, name='Forecast')
        model_fit = ExponentialSmoothing(series, seasonal='add', seasonal_periods=12).fit()
    
    forecast = model_fit.forecast(steps=steps)
    forecast_series = pd.Series(forecast, name='Forecast')
    
    if plot:
        plot_forecast(series, forecast_series, title='Exponential Smoothing Forecast', save_path=plot_path)

    return forecast_series

# Main function for forecasting
def main_forecasting(file_path, column_name, model_type='arima', steps=1, optimize=False, plot=False, plot_path=None):
    series = load_dataset(file_path, column_name)

    if model_type == 'arima':
        return forecast_arima(series, steps=steps, optimize=optimize, plot=plot, plot_path=plot_path)
    elif model_type == 'sarima':
        return forecast_sarima(series, steps=steps, optimize=optimize, plot=plot, plot_path=plot_path)
    elif model_type == 'exponential_smoothing':
        return forecast_exponential_smoothing(series, steps=steps, optimize=optimize, plot=plot, plot_path=plot_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'arima', 'sarima', or 'exponential_smoothing'.")
