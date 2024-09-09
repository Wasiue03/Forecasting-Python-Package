import pandas as pd
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Function to load and process a dataset
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

def forecast_arima(series, steps=1, optimize=False):
    if optimize:
        model_fit = optimize_arima(series, steps)
    else:
        model_fit = ARIMA(series, order=(1, 1, 1)).fit()
    forecast = model_fit.forecast(steps=steps)
    return pd.Series(forecast, name='Forecast')

# SARIMA optimization
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

def forecast_sarima(series, steps=1, optimize=False):
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
        if np.isscalar(forecast) or forecast.ndim == 0:
            forecast = [forecast] * steps
        return pd.Series(forecast, name='Forecast')
    except IndexError as e:
        print(f"IndexError encountered during forecasting: {e}")
        return pd.Series([None] * steps, name='Forecast')

# Exponential Smoothing optimization
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

def forecast_exponential_smoothing(series, steps=1, optimize=False):
    if optimize:
        model_fit = optimize_exponential_smoothing(series, steps)
    else:
        if len(series) < 24:
            print("Insufficient data for seasonal Exponential Smoothing.")
            return pd.Series([None] * steps, name='Forecast')
        model_fit = ExponentialSmoothing(series, seasonal='add', seasonal_periods=12).fit()
    forecast = model_fit.forecast(steps=steps)
    return pd.Series(forecast, name='Forecast')

# Main function for forecasting
def main_forecasting(file_path, column_name, model_type='arima', steps=1, optimize=False):
    series = load_dataset(file_path, column_name)

    if model_type == 'arima':
        return forecast_arima(series, steps=steps, optimize=optimize)
    elif model_type == 'sarima':
        return forecast_sarima(series, steps=steps, optimize=optimize)
    elif model_type == 'exponential_smoothing':
        return forecast_exponential_smoothing(series, steps=steps, optimize=optimize)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'arima', 'sarima', or 'exponential_smoothing'.")
