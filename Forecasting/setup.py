from setuptools import setup, find_packages
import os

# Read the contents of your README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='forecastify',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'statsmodels',
        'optuna',
        'matplotlib',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'forecasting=forecasting.forecast:main_forecasting',
        ],
    },
    test_suite='tests',
    tests_require=['unittest'],
    python_requires='>=3.7',
    author='Abdul Wasiue',
    author_email='abdulwasiueunk@gmail.com',
    description='A package for time series forecasting using ARIMA, SARIMA, and Exponential Smoothing',
    url='https://github.com/Wasiue03/Forecasting-Python-Package',
    long_description=read_readme(),
    long_description_content_type='text/markdown',  # Change to 'text/x-rst' if you use reStructuredText
)
