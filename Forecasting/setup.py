# setup.py
from setuptools import setup, find_packages

setup(
    name='time_series_library',
    version='0.2',  # Incremented version
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'scikit-learn',
        'matplotlib',
        'optuna',  # Added Optuna
    ],
    description='A library for time series forecasting and decomposition with automated hyperparameter optimization.',
)
