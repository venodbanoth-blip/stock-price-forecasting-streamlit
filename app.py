import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Reliance Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Reliance Industries Stock Price Forecasting")
st.caption("Machine Learning + ARIMA | Future Forecast + Historical Backtesting")


# ============================================================
# FILES
# ============================================================

MODEL_FILE = "reliance_linear_regression.pkl"
SCALER_FILE = "reliance_scaler.pkl"
FEATURE_FILE = "reliance_features.pkl"
DATA_FILE = "Company stock prices (1).xlsx"


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def build_technical_features(data):

    df = data.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date")
    df = df.drop_duplicates("Date")
    df = df.reset_index(drop=True)

    # -------------------------------
    # LAG FEATURES
    # -------------------------------

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)
    df["Close_Lag5"] = df["Close"].shift(5)

    # -------------------------------
    # MOVING AVERAGES
    # -------------------------------

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # -------------------------------
    # EMA
    # -------------------------------

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # -------------------------------
    # RETURNS
    # -------------------------------

    df["Daily_Return"] = df["Close"].pct_change()

    # -------------------------------
    # VOLATILITY
    # -------------------------------

    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

    # -------------------------------
    # RSI
    # -------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI_14"] = 100 - (100 / (1 + rs))

    # -------------------------------
    # MACD
    # -------------------------------

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # -------------------------------
    # BOLLINGER BANDS
    # -------------------------------

    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()

    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

    # -------------------------------
    # PRICE RANGE
    # -------------------------------

    df["High_Low_Range"] = df["High"] - df["Low"]
    df["Open_Close_Range"] = df["Open"] - df["Close"]

    # -------------------------------
    # VOLUME
    # -------------------------------

    df["Volume_MA_10"] = df["Volume"].rolling(10).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    # -------------------------------
    # DATE FEATURES
    # -------------------------------

    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

    # -------------------------------
    # EXTRA FEATURES
    # -------------------------------

    df["Daily_Range_Pct"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["Open_Close_Return"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    df["Volume_Ratio"] = (
        df["Volume"] / df["Volume_MA_10"]
    )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    features = joblib.load(FEATURE_FILE)

    return model, scaler, features


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:

        df = pd.read_excel(
            uploaded_file,
            sheet_name="in"
        )

    else:

        df = pd.read_excel(
            DATA_FILE,
            sheet_name="in"
        )

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for col in required_columns[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Date", "Close"]
    )

    df = df.sort_values("Date")
    df = df.drop_duplicates("Date")
    df = df.reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(
            "At least 60 historical rows are required."
        )

    return df


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_close(
    feature_df,
    model,
    scaler,
    features
):

    X = feature_df[features].copy()

    if X.isnull().any().any():

        missing_cols = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "NaN values found in model features: "
            + str(missing_cols)
        )

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)

    return float(prediction[0])


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    history,
    model,
    scaler,
    features,
    n_days
):

    working = history.copy()

    working["Date"] = pd.to_datetime(
        working["Date"]
    )

    working = working.sort_values("Date")
    working = working.reset_index(drop=True)

    future_dates = pd.bdate_range(
        start=working["Date"].max() + pd.Timedelta(days=1),
        periods=n_days
    )

    predictions = []

    for future_date in future_dates:

        previous_close = float(
            working["Close"].iloc[-1]
        )

        # Use recent average volume
        recent_volume = working["Volume"].tail(10)

        if recent_volume.notna().any():

            future_volume = float(
                recent_volume.mean()
            )

        else:

            future_volume = float(
                working["Volume"].iloc[-1]
            )

        # Create temporary future row
        new_row = {
            "Date": future_date,
            "Open": previous_close,
            "High": previous_close,
            "Low": previous_close,
            "Close": previous_close,
            "Adj Close": previous_close,
            "Volume": future_volume
        }

        temp = pd.concat(
            [
                working,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        # Recalculate features
        temp_features = build_technical_features(temp)

        # Last row
        last_features = temp_features.tail(1)

        # Predict
        prediction = predict_close(
            last_features,
            model,
            scaler,
            features
        )

        # Prevent negative stock prices
        prediction = max(
            prediction,
            0.01
        )

        # Update future OHLC
        temp.loc[
            temp.index[-1],
            "Close"
        ] = prediction

        temp.loc[
            temp.index[-1],
            "Adj Close"
        ] = prediction

        temp.loc[
            temp.index[-1],
            "High"
        ] = max(
            previous_close,
            prediction
        )

        temp.loc[
            temp.index[-1],
            "Low"
        ] = min(
            previous_close,
            prediction
        )

        working = temp

        predictions.append(
            {
                "Date": future_date,
                "Predicted_Close_LR": prediction
            }
        )

    return pd.DataFrame(predictions)


# ============================================================
# BACKTEST
# ============================================================

def backtest_model(
    history,
    model,
    scaler,
    features,
    test_days
):

    if len(history) <= test_days + 60:

        raise ValueError(
            f"Not enough data for {test_days}-day backtest."
        )

    train_data = history.iloc[:-test_days].copy()

    actual_data = history.iloc[-test_days:].copy()

    # Forecast exactly the same number of days
    predictions = recursive_forecast(
        train_data,
        model,
        scaler,
        features,
        test_days
    )

    result = actual_data[
        [
            "Date",
            "Close"
        ]
    ].copy()

    result = result.rename(
        columns={
            "Close": "Actual_Close"
        }
    )

    result = result.reset_index(drop=True)

    predictions = predictions.reset_index(drop=True)

    result["Predicted_Close"] = (
        predictions["Predicted_Close_LR"]
    )

    # Error
    result["Error"] = (
        result["Predicted_Close"]
        - result["Actual_Close"]
    )

    result["Absolute_Error"] = (
        result["Error"].abs()
    )

    result["Percentage_Error"] = (
        result["Absolute_Error"]
        / result["Actual_Close"]
        * 100
    )

    mae = result["Absolute_Error"].mean()

    rmse = np.sqrt(
        np.mean(
            result["Error"] ** 2
        )
    )

    mape = result["Percentage_Error"].mean()

    accuracy = max(
        0,
        100 - mape
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Accuracy": accuracy
    }

    return result, metrics


# ============================================================
# ARIMA
# ============================================================

def arima_forecast(
    history,
    n_days
):

    df = history.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values("Date")

    close_series = (
        df["Close"]
        .astype(float)
        .reset_index(drop=True)
    )

    if len(close_series) < 30:

        raise ValueError(
            "At least 30 observations are required for ARIMA."
        )

    model = ARIMA(
        close_series,
        order=(5, 1, 0)
    )

    fitted = model.fit()

    forecast_result = fitted.get_forecast(
        steps=n_days
    )

    forecast = forecast_result.predicted_mean

    confidence = forecast_result.conf_int(
        alpha=0.05
    )

    future_dates = pd.bdate_range(
        start=df["Date"].max()
        + pd.Timedelta(days=1),
        periods=n_days
    )

    result = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Close_ARIMA": forecast.values,
            "ARIMA_Lower_95": confidence.iloc[:, 0].values,
            "ARIMA_Upper_95": confidence.iloc[:, 1].values
        }
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Forecast Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file",
    type=["xlsx"]
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=60,
    value=30
)

show_arima = st.sidebar.checkbox(
    "Show ARIMA Forecast",
    value=True
)

st.sidebar.markdown("---")

backtest_days = st.sidebar.selectbox(
    "Historical Backtest",
    [30, 60]
)

run_forecast = st.sidebar.button(
    "🚀 Run Forecast",
    use_container_width=True
)

run_backtest = st.sidebar.button(
    "🔍 Run Backtest",
    use_container_width=True
)


# ============================================================
# CHECK FILES
# ============================================================

missing_files = []

for file in [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE
]:

    if not os.path.exists(file):

        missing_files.append(file)


if missing_files:

    st.error(
        "Missing model files: "
        + ", ".join(missing_files)
    )

    st.stop()


if uploaded_file is None and not os.path.exists(DATA_FILE):

    st.error(
        f"Excel file not found: {DATA_FILE}"
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    model, scaler, features = load_artifacts()

    history = load_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Error loading project files: {e}"
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.subheader("📊 Historical Data")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        len(history)
    )

with col2:

    st.metric(
        "Start Date",
        history["Date"].min().strftime("%d-%m-%Y")
    )

with col3:

    st.metric(
        "End Date",
        history["Date"].max().strftime("%d-%m-%Y")
    )

with col4:

    latest_close = history["Close"].iloc[-1]

    st.metric(
        "Latest Close",
        f"₹{latest_close:,.2f}"
    )


# ============================================================
# HISTORICAL CHART
# ============================================================

st.subheader("📈 Historical Reliance Closing Price")

fig, ax = plt.subplots(
    figsize=(14, 5)
)

recent_history = history.tail(250)

ax.plot(
    recent_history["Date"],
    recent_history["Close"],
    label="Actual Close"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price (₹)")
ax.set_title("Reliance Industries Historical Closing Price")

ax.legend()

ax.grid(alpha=0.3)

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close()


# ============================================================
# BACKTEST
# ============================================================

if run_backtest:

    st.subheader(
        f"🔍 Last {backtest_days}-Trading-Day Backtest"
    )

    with st.spinner(
        f"Testing model on last {backtest_days} trading days..."
    ):

        try:

            backtest_result, metrics = backtest_model(
                history,
                model,
                scaler,
                features,
                backtest_days
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "MAE",
                    f"₹{metrics['MAE']:,.2f}"
                )

            with c2:

                st.metric(
                    "RMSE",
                    f"₹{metrics['RMSE']:,.2f}"
                )

            with c3:

                st.metric(
                    "MAPE",
                    f"{metrics['MAPE']:.2f}%"
                )

            with c4:

                st.metric(
                    "Approx. Accuracy",
                    f"{metrics['Accuracy']:.2f}%"
                )

            # -------------------------------
            # BACKTEST GRAPH
            # -------------------------------

            fig2, ax2 = plt.subplots(
                figsize=(14, 6)
            )

            ax2.plot(
                backtest_result["Date"],
                backtest_result["Actual_Close"],
                label="Actual Price",
                linewidth=2
            )

            ax2.plot(
                backtest_result["Date"],
                backtest_result["Predicted_Close"],
                label="Predicted Price",
                linestyle="--",
                linewidth=2
            )

            ax2.set_title(
                f"Actual vs Predicted - Last {backtest_days} Trading Days"
            )

            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price (₹)")

            ax2.legend()

            ax2.grid(alpha=0.3)

            plt.xticks(rotation=45)

            st.pyplot(fig2)

            plt.close()

            # -------------------------------
            # TABLE
            # -------------------------------

            st.dataframe(
                backtest_result.style.format(
                    {
                        "Actual_Close": "₹{:.2f}",
                        "Predicted_Close": "₹{:.2f}",
                        "Error": "₹{:.2f}",
                        "Absolute_Error": "₹{:.2f}",
                        "Percentage_Error": "{:.2f}%"
                    }
                ),
                use_container_width=True
            )

            # -------------------------------
            # DOWNLOAD
            # -------------------------------

            csv = backtest_result.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Backtest CSV",
                csv,
                file_name=f"reliance_backtest_{backtest_days}_days.csv",
                mime="text/csv"
            )

            if metrics["MAPE"] > 20:

                st.warning(
                    "⚠️ The model has high error on this historical period. "
                    "The 30/60-day future forecast should therefore be treated "
                    "as low-confidence."
                )

            elif metrics["MAPE"] > 10:

                st.warning(
                    "⚠️ The model has moderate forecasting error."
                )

            else:

                st.success(
                    "✅ The model shows relatively low historical forecasting error."
                )

        except Exception as e:

            st.error(
                f"Backtest error: {e}"
            )


# ============================================================
# FUTURE FORECAST
# ============================================================

if run_forecast:

    st.subheader(
        f"🔮 Next {horizon} Business-Day Forecast"
    )

    with st.spinner(
        "Generating forecast..."
    ):

        try:

            lr_forecast = recursive_forecast(
                history,
                model,
                scaler,
                features,
                horizon
            )

            if show_arima:

                arima_result = arima_forecast(
                    history,
                    horizon
                )

                forecast_result = pd.merge(
                    lr_forecast,
                    arima_result,
                    on="Date",
                    how="left"
                )

            else:

                forecast_result = lr_forecast.copy()


            # ====================================================
            # FORECAST METRICS
            # ====================================================

            latest_price = float(
                history["Close"].iloc[-1]
            )

            next_day_price = float(
                lr_forecast["Predicted_Close_LR"].iloc[0]
            )

            final_price = float(
                lr_forecast["Predicted_Close_LR"].iloc[-1]
            )

            expected_change = (
                (final_price - latest_price)
                / latest_price
                * 100
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Current Price",
                    f"₹{latest_price:,.2f}"
                )

            with c2:

                st.metric(
                    "Next Trading Day",
                    f"₹{next_day_price:,.2f}"
                )

            with c3:

                st.metric(
                    f"Day {horizon}",
                    f"₹{final_price:,.2f}"
                )

            with c4:

                st.metric(
                    "Expected Change",
                    f"{expected_change:+.2f}%"
                )


            # ====================================================
            # FORECAST CHART
            # ====================================================

            st.subheader("📈 Forecast Chart")

            fig3, ax3 = plt.subplots(
                figsize=(15, 7)
            )

            chart_history = history.tail(120)

            ax3.plot(
                chart_history["Date"],
                chart_history["Close"],
                label="Historical Close",
                linewidth=2
            )

            ax3.plot(
                forecast_result["Date"],
                forecast_result["Predicted_Close_LR"],
                label="Linear Regression",
                linewidth=2,
                linestyle="--"
            )

            if show_arima:

                ax3.plot(
                    forecast_result["Date"],
                    forecast_result["Predicted_Close_ARIMA"],
                    label="ARIMA",
                    linewidth=2,
                    linestyle=":"
                )

                ax3.fill_between(
                    forecast_result["Date"],
                    forecast_result["ARIMA_Lower_95"],
                    forecast_result["ARIMA_Upper_95"],
                    alpha=0.15,
                    label="ARIMA 95% Confidence Interval"
                )

            ax3.axvline(
                history["Date"].max(),
                linestyle="--",
                linewidth=1.5,
                label="Forecast Start"
            )

            ax3.set_title(
                f"Reliance Industries - {horizon} Business-Day Forecast"
            )

            ax3.set_xlabel("Date")
            ax3.set_ylabel("Price (₹)")

            ax3.legend()

            ax3.grid(alpha=0.3)

            plt.xticks(rotation=45)

            st.pyplot(fig3)

            plt.close()


            # ====================================================
            # FORECAST TABLE
            # ====================================================

            st.subheader("📋 Forecast Values")

            display_forecast = forecast_result.copy()

            st.dataframe(
                display_forecast.style.format(
                    {
                        "Predicted_Close_LR": "₹{:.2f}",
                        "Predicted_Close_ARIMA": "₹{:.2f}",
                        "ARIMA_Lower_95": "₹{:.2f}",
                        "ARIMA_Upper_95": "₹{:.2f}"
                    }
                ),
                use_container_width=True
            )


            # ====================================================
            # DOWNLOAD
            # ====================================================

            csv = forecast_result.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Forecast CSV",
                csv,
                file_name=f"reliance_forecast_{horizon}_days.csv",
                mime="text/csv"
            )


            # ====================================================
            # INTERPRETATION
            # ====================================================

            st.subheader("🧠 Forecast Interpretation")

            if expected_change > 5:

                st.success(
                    f"📈 The Linear Regression model forecasts an "
                    f"approximately {expected_change:.2f}% increase "
                    f"over the next {horizon} business days."
                )

            elif expected_change < -5:

                st.error(
                    f"📉 The Linear Regression model forecasts an "
                    f"approximately {abs(expected_change):.2f}% decrease "
                    f"over the next {horizon} business days."
                )

            else:

                st.info(
                    f"➡️ The Linear Regression model forecasts a relatively "
                    f"stable movement over the next {horizon} business days."
                )


            # ====================================================
            # IMPORTANT WARNING
            # ====================================================

            st.warning(
                "⚠️ Stock-market forecasts are uncertain. "
                "The model uses historical price patterns and technical "
                "indicators and cannot predict unexpected market events."
            )


        except Exception as e:

            st.error(
                f"Forecast error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Reliance Industries Stock Price Forecasting | "
    "Machine Learning + Time Series Analysis"
)
