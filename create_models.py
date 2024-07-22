import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate some sample data
dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
sales = np.random.randint(100, 200, size=(36,))
data = pd.DataFrame({'date': dates, 'sales': sales})
data.set_index('date', inplace=True)

# Print the sample data to verify
print("Sample Data:")
print(data.head())

# ARIMA model
arima_model = ARIMA(data['sales'], order=(1, 1, 1)).fit()
print("\nARIMA Model Summary:")
print(arima_model.summary())

# Save ARIMA model
with open('models/arima_model.pkl', 'wb') as f:
    pickle.dump(arima_model, f)

# SARIMAX model
sarimax_model = SARIMAX(data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
print("\nSARIMAX Model Summary:")
print(sarimax_model.summary())

# Save SARIMAX model
with open('models/sarimax_model.pkl', 'wb') as f:
    pickle.dump(sarimax_model, f)

print("\nModels have been created and saved successfully.")
