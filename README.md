
# MarketCast — Stock Price Forecasting Dashboard

MarketCast is an interactive stock price forecasting application built with Python and Streamlit.

## Features

- Download historical stock market data using Yahoo Finance.
- Explore historical stock prices.
- Generate next-day forecasts using ARIMA(1,1,1).
- Compare ARIMA predictions against a naive forecasting benchmark.
- Evaluate model performance using Mean Absolute Error (MAE).
- Visualize actual versus predicted stock prices.

## Technologies

- Python
- Streamlit
- Pandas
- yfinance
- Statsmodels
- Scikit-learn
- Plotly

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Launch the application:

    streamlit run app.py

## Disclaimer

This application is an educational forecasting project and does not provide financial advice.
