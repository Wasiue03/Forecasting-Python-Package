# forecast.py

import pandas as pd
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

def optimize_arima(series, steps=1):
    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)
        order = (p, d, q)
        train = series[:-steps]
        test = series[-steps:]
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return mean_squared_error(test, forecast)
    
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

def optimize_sarima(series, steps=1):
    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)
        P = trial.suggest_int('P', 0, 5)
        D = trial.suggest_int('D', 0, 2)
        Q = trial.suggest_int('Q', 0, 5)
        s = trial.suggest_int('s', 4, 12)
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        train = series[:-steps]
        test = series[-steps:]
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
        return mean_squared_error(test, forecast)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    best_order = (best_params['p'], best_params['d'], best_params['q'])
    best_seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
    return SARIMAX(series, order=best_order, seasonal_order=best_seasonal_order).fit()

def forecast_sarima(series, steps=1, optimize=False):
    if optimize:
        model_fit = optimize_sarima(series, steps)
    else:
        model_fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return pd.Series(forecast, name='Forecast')

def optimize_exponential_smoothing(series, steps=1):
    def objective(trial):
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul'])
        seasonal_periods = trial.suggest_int('seasonal_periods', 2, 12)
        train = series[:-steps]
        test = series[-steps:]
        model = ExponentialSmoothing(train, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return mean_squared_error(test, forecast)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    return ExponentialSmoothing(series, seasonal=best_params['seasonal'], 
                                seasonal_periods=best_params['seasonal_periods']).fit()

def forecast_exponential_smoothing(series, steps=1, optimize=False):
    if optimize:
        model_fit = optimize_exponential_smoothing(series, steps)
    else:
        model_fit = ExponentialSmoothing(series, seasonal='add', seasonal_periods=12).fit()
    forecast = model_fit.forecast(steps=steps)
    return pd.Series(forecast, name='Forecast')
