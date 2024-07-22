import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate some sample data
dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
sales = np.random.randint(100, 200, size=(36,))
data = pd.DataFrame({'date': dates, 'sales': sales})
data.set_index('date', inplace=True)

# Load models
with open('models/arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

with open('models/sarimax_model.pkl', 'rb') as f:
    sarimax_model = pickle.load(f)

# Forecast using ARIMA
forecast_periods = 12
arima_forecast = arima_model.get_forecast(steps=forecast_periods)
arima_forecast_values = arima_forecast.predicted_mean

# Forecast using SARIMAX
sarimax_forecast = sarimax_model.get_forecast(steps=forecast_periods)
sarimax_forecast_values = sarimax_forecast.predicted_mean

# Print forecast results
print("ARIMA Forecast:")
print(arima_forecast_values)

print("\nSARIMAX Forecast:")
print(sarimax_forecast_values)
