
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error


def get_stock_data(ticker="AAPL", period="5y"):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No stock data found. Check the ticker symbol.")

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    close.name = "Close"

    return close


def evaluate_forecasts(close, test_size=60, order=(1, 1, 1)):
    if len(close) <= test_size:
        raise ValueError("Not enough observations for the selected test period.")

    train = close.iloc[:-test_size]
    test = close.iloc[-test_size:]

    history = train.copy()
    arima_predictions = []

    for date in test.index:
        model = ARIMA(
            history.reset_index(drop=True),
            order=order
        )

        fitted_model = model.fit()
        prediction = fitted_model.forecast(steps=1).iloc[0]

        arima_predictions.append(prediction)

        # Add the actual observation only after making the prediction
        history = pd.concat([history, test.loc[[date]]])

    arima_predictions = pd.Series(
        arima_predictions,
        index=test.index,
        name="ARIMA"
    )

    naive_predictions = close.shift(1).loc[test.index]
    naive_predictions.name = "Naive"

    arima_mae = mean_absolute_error(test, arima_predictions)
    naive_mae = mean_absolute_error(test, naive_predictions)

    return {
        "train": train,
        "test": test,
        "arima_predictions": arima_predictions,
        "naive_predictions": naive_predictions,
        "arima_mae": arima_mae,
        "naive_mae": naive_mae
    }


def forecast_next_day(close, order=(1, 1, 1)):
    model = ARIMA(
        close.reset_index(drop=True),
        order=order
    )

    fitted_model = model.fit()

    prediction = fitted_model.forecast(steps=1).iloc[0]

    return float(prediction)
