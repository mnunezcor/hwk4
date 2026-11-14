
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from forecasting import (
    get_stock_data,
    evaluate_forecasts,
    forecast_next_day
)


st.set_page_config(
    page_title="MarketCast | Stock Forecasting",
    page_icon="📈",
    layout="wide"
)


st.title("📈 MarketCast")
st.subheader("Stock Price Forecasting Dashboard")

st.write(
    "Explore historical stock prices, compare ARIMA predictions "
    "against a naive benchmark, and generate a next-day forecast."
)

st.sidebar.header("Forecast Settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="AAPL"
).strip().upper()

test_size = st.sidebar.slider(
    "Test period (trading days)",
    min_value=30,
    max_value=120,
    value=60
)

st.sidebar.caption("Forecasting model: ARIMA(1,1,1)")
st.sidebar.caption("Benchmark: Tomorrow's price = today's price")


@st.cache_data(ttl=3600)
def run_analysis(ticker, test_size):
    prices = get_stock_data(ticker=ticker, period="5y")

    results = evaluate_forecasts(
        prices,
        test_size=test_size
    )

    next_prediction = forecast_next_day(prices)

    return prices, results, next_prediction


if st.button("Generate Forecast", type="primary"):

    if not ticker:
        st.error("Please enter a stock ticker.")

    else:
        try:
            with st.spinner("Downloading stock data and fitting forecasting models..."):

                prices, results, next_prediction = run_analysis(
                    ticker,
                    test_size
                )

            st.success(f"Analysis completed for {ticker}.")

            latest_price = prices.iloc[-1]

            st.markdown("## Market Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latest Closing Price",
                f"${latest_price:,.2f}"
            )

            col2.metric(
                "Next Trading Day Forecast",
                f"${next_prediction:,.2f}",
                f"{next_prediction - latest_price:+,.2f}"
            )

            col3.metric(
                "Historical Observations",
                f"{len(prices):,}"
            )

            st.divider()

            st.markdown("## Model Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "ARIMA MAE",
                f"${results['arima_mae']:,.2f}"
            )

            col2.metric(
                "Naive MAE",
                f"${results['naive_mae']:,.2f}"
            )

            better_model = (
                "ARIMA"
                if results["arima_mae"] < results["naive_mae"]
                else "Naive"
            )

            col3.metric(
                "Better Performing Model",
                better_model
            )

            st.caption(
                "Lower Mean Absolute Error (MAE) indicates better "
                "average prediction accuracy."
            )

            st.divider()

            st.markdown("## Historical Stock Prices")

            historical_fig = go.Figure()

            historical_fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    mode="lines",
                    name="Closing Price"
                )
            )

            historical_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Adjusted Closing Price ($)",
                hovermode="x unified",
                height=450
            )

            st.plotly_chart(
                historical_fig,
                use_container_width=True
            )

            st.markdown("## Actual vs. Predicted Prices")

            comparison_fig = go.Figure()

            comparison_fig.add_trace(
                go.Scatter(
                    x=results["test"].index,
                    y=results["test"].values,
                    mode="lines",
                    name="Actual Price"
                )
            )

            comparison_fig.add_trace(
                go.Scatter(
                    x=results["arima_predictions"].index,
                    y=results["arima_predictions"].values,
                    mode="lines",
                    name="ARIMA Forecast"
                )
            )

            comparison_fig.add_trace(
                go.Scatter(
                    x=results["naive_predictions"].index,
                    y=results["naive_predictions"].values,
                    mode="lines",
                    name="Naive Forecast"
                )
            )

            comparison_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Stock Price ($)",
                hovermode="x unified",
                height=450
            )

            st.plotly_chart(
                comparison_fig,
                use_container_width=True
            )

            st.markdown("## Forecast Comparison")

            comparison = pd.DataFrame({
                "Actual Price": results["test"],
                "ARIMA Prediction": results["arima_predictions"],
                "Naive Prediction": results["naive_predictions"]
            })

            st.dataframe(
                comparison.style.format("${:,.2f}"),
                use_container_width=True
            )

            st.markdown("## Model Interpretation")

            if results["arima_mae"] < results["naive_mae"]:
                st.write(
                    "ARIMA achieved a lower MAE than the naive benchmark "
                    "during this test period."
                )
            else:
                st.write(
                    "The naive benchmark achieved a lower MAE than ARIMA. "
                    "In this period, using the previous day's closing price "
                    "was at least as effective as the more complex model."
                )

            st.info(
                "This application is for educational purposes only. "
                "Stock forecasts are uncertain and should not be treated "
                "as financial or investment advice."
            )

        except Exception as e:
            st.error(f"Unable to generate forecast: {e}")


else:
    st.info(
        "Enter a stock ticker and click **Generate Forecast** to begin."
    )
