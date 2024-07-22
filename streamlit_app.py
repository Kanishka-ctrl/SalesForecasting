import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# Load models
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found. Please ensure it is in the 'models' directory.")
        return None


arima_model = load_model('models/arima_model.pkl')
sarimax_model = load_model('models/sarimax_model.pkl')

# Page title and description
st.title('Sales Forecasting Application')
st.markdown("""
This application allows you to forecast sales for a selected product using ARIMA and SARIMAX models.
You can upload your historical sales data and see forecasts for future sales.
""")

# Sidebar inputs
st.sidebar.header('User Input Features')
uploaded_file = st.sidebar.file_uploader("Upload Historical Sales Data", type=["csv"])


def preprocess_data(data):
    # Combine YEAR and MONTH into a single date column
    data['date'] = pd.to_datetime(data[['YEAR', 'MONTH']].assign(DAY=1))
    data.set_index('date', inplace=True)
    return data


def moving_average(series, window_size):
    return series.rolling(window=window_size).mean()


# Load data (uploaded by user or sample data)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully. Here is a preview:")
    st.write(data.head())
else:
    sample_data_path = 'data/sample_data.csv'
    if os.path.exists(sample_data_path):
        data = pd.read_csv(sample_data_path)
        st.write("No file uploaded. Using sample data:")
        st.write(data.head())
    else:
        st.write("Please upload historical sales data to proceed.")
        data = None

# Filter data by selected product
if data is not None and 'ITEM DESCRIPTION' in data.columns:
    product_list = data['ITEM DESCRIPTION'].unique()
    selected_product = st.sidebar.selectbox('Select Product', product_list)
    filtered_data = data[data['ITEM DESCRIPTION'] == selected_product]

    if filtered_data.empty:
        st.write(f"No data available for the selected product: {selected_product}")
    else:
        st.write(f"Showing data for product: {selected_product}")
        st.write(filtered_data.head())

        # Preprocess data
        filtered_data = preprocess_data(filtered_data)

        # Assuming we are forecasting based on 'RETAIL SALES'
        sales_data = filtered_data['RETAIL SALES']


        # ADF test for stationarity
        def adf_test(series):
            try:
                result = adfuller(series, autolag='AIC')
                return result[1]  # p-value
            except ValueError as e:
                st.error(f"ADF Test Error: {e}")
                return None


        # Check stationarity and difference if needed
        p_value = adf_test(sales_data)
        d = 0
        while p_value is not None and p_value > 0.05 and len(sales_data) >= 10:
            sales_data = sales_data.diff().dropna()
            p_value = adf_test(sales_data)
            d += 1

        # Use fixed ARIMA parameters
        p, d, q = 1, d, 1

        # Train the ARIMA model with chosen parameters
        try:
            trained_arima_model = ARIMA(sales_data, order=(p, d, q)).fit()
        except Exception as e:
            st.error(f"Error fitting ARIMA model: {e}")
            trained_arima_model = None

        # Define the forecast period
        forecast_periods = 12
        last_date = sales_data.index[-1]
        forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]


        # Generate forecasts
        def generate_forecast(model, steps):
            try:
                forecast = model.get_forecast(steps=steps)
                forecast_values = forecast.predicted_mean
                forecast_values.index = forecast_dates
                return forecast_values
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return None


        arima_forecast = generate_forecast(trained_arima_model,
                                           forecast_periods) if trained_arima_model is not None else None

        # SARIMAX Model parameters
        if len(sales_data) >= 10:
            # Use optimized SARIMAX parameters
            def optimize_sarimax(data):
                best_aic = float('inf')
                best_order = None
                best_seasonal_order = None
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            for P in range(0, 2):
                                for D in range(0, 2):
                                    for Q in range(0, 2):
                                        for s in [12]:
                                            try:
                                                model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(
                                                    disp=False)
                                                aic = model.aic
                                                if aic < best_aic:
                                                    best_aic = aic
                                                    best_order = (p, d, q)
                                                    best_seasonal_order = (P, D, Q, s)
                                            except Exception as e:
                                                continue
                return best_order, best_seasonal_order


            best_order, best_seasonal_order = optimize_sarimax(sales_data)
        else:
            # Use default SARIMAX parameters for small datasets
            best_order = (1, 0, 1)
            best_seasonal_order = (1, 0, 1, 12)

        # Train the SARIMAX model with chosen parameters
        try:
            trained_sarimax_model = SARIMAX(sales_data, order=best_order, seasonal_order=best_seasonal_order).fit(
                disp=False)
        except Exception as e:
            st.error(f"Error fitting SARIMAX model: {e}")
            trained_sarimax_model = None

        # Generate SARIMAX forecasts
        sarimax_forecast = generate_forecast(trained_sarimax_model,
                                             forecast_periods) if trained_sarimax_model is not None else None

        # Enhanced visualizations with Plotly
        if sales_data is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data, mode='lines', name='Historical Sales',
                                     line=dict(color='blue')))

            if arima_forecast is not None:
                fig.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast, mode='lines', name='ARIMA Forecast',
                                         line=dict(color='orange')))

            if sarimax_forecast is not None:
                fig.add_trace(
                    go.Scatter(x=sarimax_forecast.index, y=sarimax_forecast, mode='lines', name='SARIMAX Forecast',
                               line=dict(color='green')))

            fig.update_layout(title=f'{selected_product} - Historical and Forecasted Sales',
                              xaxis_title='Date',
                              yaxis_title='Sales',
                              legend_title='Legend',
                              template='plotly_dark')
            st.plotly_chart(fig)

        # Plot yearly predictions
        if arima_forecast is not None or sarimax_forecast is not None:
            yearly_forecasts = pd.DataFrame()
            if arima_forecast is not None:
                yearly_forecasts['ARIMA'] = arima_forecast.resample('Y').sum()
            if sarimax_forecast is not None:
                yearly_forecasts['SARIMAX'] = sarimax_forecast.resample('Y').sum()

            fig_yearly = go.Figure()
            if 'ARIMA' in yearly_forecasts:
                fig_yearly.add_trace(
                    go.Bar(x=yearly_forecasts.index, y=yearly_forecasts['ARIMA'], name='ARIMA Forecast'))
            if 'SARIMAX' in yearly_forecasts:
                fig_yearly.add_trace(
                    go.Bar(x=yearly_forecasts.index, y=yearly_forecasts['SARIMAX'], name='SARIMAX Forecast'))

            fig_yearly.update_layout(title='Yearly Sales Forecasts',
                                     xaxis_title='Year',
                                     yaxis_title='Total Sales',
                                     legend_title='Legend',
                                     template='plotly_dark')
            st.plotly_chart(fig_yearly)

        # Written summary
        st.markdown("### Forecast Summary")
        if arima_forecast is not None:
            arima_avg = arima_forecast.mean()
            st.write(f"**ARIMA Model Forecast Average:** {arima_avg:.2f} units")
        else:
            st.write("ARIMA Forecast is not available.")

        if sarimax_forecast is not None:
            sarimax_avg = sarimax_forecast.mean()
            st.write(f"**SARIMAX Model Forecast Average:** {sarimax_avg:.2f} units")
        else:
            st.write("SARIMAX Forecast is not available.")

        st.write(f"Forecasting period: {forecast_dates[0].strftime('%B %Y')} to {forecast_dates[-1].strftime('%B %Y')}")

        # Download forecast data
        if arima_forecast is not None:
            arima_forecast_csv = arima_forecast.to_csv().encode('utf-8')
            st.download_button(label="Download ARIMA Forecast as CSV",
                               data=arima_forecast_csv,
                               file_name='arima_forecast.csv',
                               mime='text/csv')
        if sarimax_forecast is not None:
            sarimax_forecast_csv = sarimax_forecast.to_csv().encode('utf-8')
            st.download_button(label="Download SARIMAX Forecast as CSV",
                               data=sarimax_forecast_csv,
                               file_name='sarimax_forecast.csv',
                               mime='text/csv')

else:
    st.write("Please upload historical sales data or ensure sample data is available.")

st.write("Upload the historical sales data to get predictions.")
