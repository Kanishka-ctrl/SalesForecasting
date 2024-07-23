# arima_sarimax_models.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit


def fit_sarimax(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit


def forecast(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast


if __name__ == "__main__":
    data = pd.read_csv('data/data.csv')
    data.set_index('Date', inplace=True)

    # Example orders, these should be tuned
    arima_order = (5, 1, 0)
    sarimax_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    arima_model = fit_arima(data['Sales'], arima_order)
    sarimax_model = fit_sarimax(data['Sales'], sarimax_order, seasonal_order)

    arima_forecast = forecast(arima_model, steps=12)
    sarimax_forecast = forecast(sarimax_model, steps=12)

    print("ARIMA Forecast:")
    print(arima_forecast)

    print("SARIMAX Forecast:")
    print(sarimax_forecast)
