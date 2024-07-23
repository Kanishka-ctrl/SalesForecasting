import streamlit as st
import pandas as pd
import numpy as np
from load_data import load_data
from arima_sarimax_models import fit_arima, fit_sarimax, forecast
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime


# Function to calculate accuracy metrics
def calculate_accuracy(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae


# Load the data
data_path = 'data/data.csv'
data = load_data(data_path)

# Combine 'YEAR' and 'MONTH' into a 'Date' column
data['Date'] = pd.to_datetime(data['YEAR'].astype(str) + '-' + data['MONTH'].astype(str) + '-01')
data.set_index('Date', inplace=True)

# Sidebar inputs
st.sidebar.title("Model Parameters")
item_selected = st.sidebar.selectbox("Select Item", data['ITEM DESCRIPTION'].unique())
launch_date = st.sidebar.date_input("Select Launch Date", datetime.today())

uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type="csv")

steps = st.sidebar.slider('Forecast Steps', min_value=1, max_value=36, value=12)

# Filter data for the selected item
item_data = data[data['ITEM DESCRIPTION'] == item_selected]

# Model parameters
arima_order = (5, 1, 0)
sarimax_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Fit models
arima_model = fit_arima(item_data['RETAIL SALES'], arima_order)
sarimax_model = fit_sarimax(item_data['RETAIL SALES'], sarimax_order, seasonal_order)

# Forecast
arima_forecast = forecast(arima_model, steps)
sarimax_forecast = forecast(sarimax_model, steps)

# Display results
st.title("Sales Forecasting")
st.write("Data for Item: ", item_selected)
st.write(item_data.tail())

# Plot ARIMA vs Historical Sales
if arima_forecast is not None:
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=item_data.index, y=item_data['RETAIL SALES'], mode='lines', name='Historical Sales', line=dict(color='blue')))
    fig_arima.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    fig_arima.update_layout(title='Historical Sales vs ARIMA Forecast',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            legend_title='Legend',
                            template='plotly_dark')
    st.plotly_chart(fig_arima)

# Plot SARIMAX vs Historical Sales
if sarimax_forecast is not None:
    fig_sarimax = go.Figure()
    fig_sarimax.add_trace(go.Scatter(x=item_data.index, y=item_data['RETAIL SALES'], mode='lines', name='Historical Sales', line=dict(color='blue')))
    fig_sarimax.add_trace(go.Scatter(x=sarimax_forecast.index, y=sarimax_forecast, mode='lines', name='SARIMAX Forecast', line=dict(color='green')))
    fig_sarimax.update_layout(title='Historical Sales vs SARIMAX Forecast',
                              xaxis_title='Date',
                              yaxis_title='Sales',
                              legend_title='Legend',
                              template='plotly_dark')
    st.plotly_chart(fig_sarimax)

# Plot ARIMA vs SARIMAX
if arima_forecast is not None and sarimax_forecast is not None:
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    fig_comparison.add_trace(go.Scatter(x=sarimax_forecast.index, y=sarimax_forecast, mode='lines', name='SARIMAX Forecast', line=dict(color='green')))
    fig_comparison.update_layout(title='ARIMA vs SARIMAX Forecast Comparison',
                                xaxis_title='Date',
                                yaxis_title='Sales',
                                legend_title='Legend',
                                template='plotly_dark')
    st.plotly_chart(fig_comparison)

# Future Forecasting
if arima_forecast is not None or sarimax_forecast is not None:
    forecast_dates = [launch_date + pd.DateOffset(months=i) for i in range(1, steps + 1)]
    future_dates = pd.date_range(start=forecast_dates[0], periods=steps, freq='M')

    # Create a DataFrame for future forecasts
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'ARIMA Forecast': arima_forecast if arima_forecast is not None else np.nan,
        'SARIMAX Forecast': sarimax_forecast if sarimax_forecast is not None else np.nan
    })

    st.write("Future Sales Forecast")
    st.table(forecast_df)

    fig_future = go.Figure()
    if arima_forecast is not None:
        fig_future.add_trace(go.Scatter(x=future_dates, y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    if sarimax_forecast is not None:
        fig_future.add_trace(go.Scatter(x=future_dates, y=sarimax_forecast, mode='lines', name='SARIMAX Forecast', line=dict(color='green')))
    fig_future.update_layout(title='Future Sales Forecast',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            legend_title='Legend',
                            template='plotly_dark')
    st.plotly_chart(fig_future)

# Accuracy calculation if test data is provided
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    test_data['Date'] = pd.to_datetime(test_data['YEAR'].astype(str) + '-' + test_data['MONTH'].astype(str) + '-01')
    test_data.set_index('Date', inplace=True)

    # Filter test data for the selected item
    test_item_data = test_data[test_data['ITEM DESCRIPTION'] == item_selected]

    # Ensure test data aligns with forecast steps
    if len(test_item_data) >= steps:
        test_item_data = test_item_data[:steps]

        # Calculate accuracy
        arima_mse, arima_mae = calculate_accuracy(test_item_data['RETAIL SALES'], arima_forecast[:steps])
        sarimax_mse, sarimax_mae = calculate_accuracy(test_item_data['RETAIL SALES'], sarimax_forecast[:steps])

        # Display accuracy tables
        accuracy_data = {
            'Model': ['ARIMA', 'SARIMAX'],
            'MSE': [arima_mse, sarimax_mse],
            'MAE': [arima_mae, sarimax_mae]
        }
        accuracy_df = pd.DataFrame(accuracy_data)
        st.write("Model Accuracy Metrics")
        st.table(accuracy_df)

# Distribution plots
st.write("Sales Distribution")
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=item_data['RETAIL SALES'], nbinsx=20, name='Sales Distribution', marker_color='rgba(255, 100, 102, 0.7)'))
fig_dist.update_layout(title='Sales Distribution',
                       xaxis_title='Sales',
                       yaxis_title='Frequency',
                       template='plotly_dark')
st.plotly_chart(fig_dist)
