import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_series(series, model='add', period=None):
    """
    Decompose a time series into trend, seasonality, and residuals.
    
    Parameters:
    - series: pd.Series, the time series data.
    - model: str, the type of decomposition ('add' or 'mul').
    - period: int, the number of periods in a season (for seasonal component).
    
    Returns:
    - decomposition: DecomposeResult object from statsmodels.
    """
    decomposition = seasonal_decompose(series, model=model, period=period)
    return decomposition
