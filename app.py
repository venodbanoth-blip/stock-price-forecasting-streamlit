import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Reliance Stock Forecast",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Reliance Industries - Stock Price Forecast")

st.write(
    "This application predicts Reliance Industries stock prices "
    "using Linear Regression and ARIMA."
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "reliance_linear_regression.pkl"
SCALER_FILE = "reliance_scaler.pkl"
FEATURE_FILE = "reliance_features.pkl"
DATA_FILE = "Company stock prices.xlsx"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_technical_features(data):

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Lag Features
    # --------------------------------------------------------

    data["Close_Lag1"] = data["Close"].shift(1)
    data["Close_Lag2"] = data["Close"].shift(2)
    data["Close_Lag3"] = data["Close"].shift(3)
    data["Close_Lag5"] = data["Close"].shift(5)

    # --------------------------------------------------------
    # Moving Averages
    # --------------------------------------------------------

    data["MA_5"] = (
        data["Close"]
        .rolling(5)
        .mean()
    )

    data["MA_10"] = (
        data["Close"]
        .rolling(10)
        .mean()
    )

    data["MA_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA_50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    data["EMA_20"] = (
        data["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # Daily Return
    # --------------------------------------------------------

    data["Daily_Return"] = (
        data["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    data["Volatility_10"] = (
        data["Daily_Return"]
        .rolling(10)
        .std()
    )

    data["Volatility_20"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss

    data["RSI_14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["BB_Mid"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["BB_Std"] = (
        data["Close"]
        .rolling(20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Mid"] +
        (2 * data["BB_Std"])
    )

    data["BB_Lower"] = (
        data["BB_Mid"] -
        (2 * data["BB_Std"])
    )

    # --------------------------------------------------------
    # Price Ranges
    # --------------------------------------------------------

    data["High_Low_Range"] = (
        data["High"] -
        data["Low"]
    )

    data["Open_Close_Range"] = (
        data["Close"] -
        data["Open"]
    )

    # --------------------------------------------------------
    # Volume Features
    # --------------------------------------------------------

    data["Volume_MA_10"] = (
        data["Volume"]
        .rolling(10)
        .mean()
    )

    data["Volume_Change"] = (
        data["Volume"]
        .pct_change()
    )

    # --------------------------------------------------------
    # Calendar Features
    # --------------------------------------------------------

    data["Day_of_Week"] = (
        data["Date"]
        .dt.dayofweek
    )

    data["Month"] = (
        data["Date"]
        .dt.month
    )

    data["Quarter"] = (
        data["Date"]
        .dt.quarter
    )

    # --------------------------------------------------------
    # Additional Features
    # --------------------------------------------------------

    data["Daily_Range_Pct"] = (
        (
            (data["High"] - data["Low"])
            / data["Close"]
        )
        * 100
    )

    data["Open_Close_Return"] = (
        (
            (data["Close"] - data["Open"])
            / data["Open"]
        )
        * 100
    )

    data["Volume_Ratio"] = (
        data["Volume"]
        / data["Volume_MA_10"]
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    features = joblib.load(
        FEATURE_FILE
    )

    return model, scaler, features


# ============================================================
# LOAD EXCEL DATA
# ============================================================

@st.cache_data
def load_data(file):

    if file is not None:

        df = pd.read_excel(
            file,
            sheet_name="in"
        )

    else:

        df = pd.read_excel(
            DATA_FILE,
            sheet_name="in"
        )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Sort by Date
    # --------------------------------------------------------

    df = (
        df[
            required_columns
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LINEAR REGRESSION RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    history,
    model,
    scaler,
    features,
    n_days
):

    working = history.copy()

    future_dates = pd.bdate_range(
        start=(
            working["Date"].max()
            + pd.Timedelta(days=1)
        ),
        periods=n_days
    )

    records = []

    # --------------------------------------------------------
    # Forecast one day at a time
    # --------------------------------------------------------

    for future_date in future_dates:

        previous_close = (
            working["Close"]
            .iloc[-1]
        )

        previous_volume_average = (
            working["Volume"]
            .tail(10)
            .mean()
        )

        # ----------------------------------------------------
        # Create temporary future row
        # ----------------------------------------------------

        new_row = pd.DataFrame(
            {
                "Date": [
                    future_date
                ],

                "Open": [
                    previous_close
                ],

                "High": [
                    previous_close
                ],

                "Low": [
                    previous_close
                ],

                "Close": [
                    previous_close
                ],

                "Adj Close": [
                    previous_close
                ],

                "Volume": [
                    previous_volume_average
                ]
            }
        )

        working = pd.concat(
            [
                working,
                new_row
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # Build technical indicators
        # ----------------------------------------------------

        feature_data = (
            build_technical_features(
                working
            )
        )

        # ----------------------------------------------------
        # Select required features
        # ----------------------------------------------------

        latest_features = (
            feature_data[
                features
            ]
            .iloc[[-1]]
        )

        # ----------------------------------------------------
        # Check missing values
        # ----------------------------------------------------

        if latest_features.isnull().any().any():

            raise ValueError(
                "Technical features contain "
                "missing values."
            )

        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        scaled_features = (
            scaler.transform(
                latest_features
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = (
            model.predict(
                scaled_features
            )[0]
        )

        prediction = float(
            prediction
        )

        # ----------------------------------------------------
        # Update future row
        # ----------------------------------------------------

        index = working.index[-1]

        working.loc[
            index,
            "Close"
        ] = prediction

        working.loc[
            index,
            "Adj Close"
        ] = prediction

        working.loc[
            index,
            "High"
        ] = max(
            previous_close,
            prediction
        )

        working.loc[
            index,
            "Low"
        ] = min(
            previous_close,
            prediction
        )

        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        records.append(
            {
                "Date": future_date,
                "Predicted_Close_LR": prediction
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# ARIMA FORECAST
# ============================================================

def arima_forecast(
    history,
    n_days
):

    series = (
        history
        .set_index("Date")["Close"]
    )

    # --------------------------------------------------------
    # ARIMA Model
    # --------------------------------------------------------

    model = ARIMA(
        series,
        order=(5, 1, 0)
    )

    fitted_model = model.fit()

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    forecast = (
        fitted_model
        .get_forecast(
            steps=n_days
        )
    )

    predicted_values = (
        forecast
        .predicted_mean
    )

    confidence_interval = (
        forecast
        .conf_int(
            alpha=0.05
        )
    )

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    future_dates = pd.bdate_range(
        start=(
            history["Date"].max()
            + pd.Timedelta(days=1)
        ),
        periods=n_days
    )

    result = pd.DataFrame(
        {
            "Date": future_dates,

            "Predicted_Close_ARIMA":
                predicted_values.values,

            "ARIMA_Lower_95":
                confidence_interval.iloc[:, 0].values,

            "ARIMA_Upper_95":
                confidence_interval.iloc[:, 1].values
        }
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload Stock Price Excel File",
        type=["xlsx"]
    )

    horizon = st.slider(
        "Forecast Horizon",
        min_value=1,
        max_value=60,
        value=30
    )

    show_arima = st.checkbox(
        "Show ARIMA Forecast",
        value=True
    )

    run_forecast = st.button(
        "🚀 Run Forecast",
        type="primary"
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

missing_files = []

if not os.path.exists(
    MODEL_FILE
):
    missing_files.append(
        MODEL_FILE
    )

if not os.path.exists(
    SCALER_FILE
):
    missing_files.append(
        SCALER_FILE
    )

if not os.path.exists(
    FEATURE_FILE
):
    missing_files.append(
        FEATURE_FILE
    )


if missing_files:

    st.error(
        "The following model files are missing:"
    )

    for file_name in missing_files:

        st.write(
            f"❌ `{file_name}`"
        )

    st.info(
        "Place the required .pkl files in the "
        "same folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

try:

    model, scaler, features = (
        load_artifacts()
    )

except Exception as error:

    st.error(
        "Unable to load the model files."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# LOAD STOCK DATA
# ============================================================

try:

    history = load_data(
        uploaded_file
    )

except FileNotFoundError:

    st.warning(
        "Company stock prices.xlsx was not found."
    )

    st.info(
        "Please upload your Excel file "
        "using the sidebar."
    )

    st.stop()

except Exception as error:

    st.error(
        "Unable to read the Excel file."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.success(
    f"Successfully loaded {len(history):,} rows."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Start Date",
    str(
        history["Date"]
        .min()
        .date()
    )
)

col2.metric(
    "End Date",
    str(
        history["Date"]
        .max()
        .date()
    )
)

col3.metric(
    "Latest Close",
    f"₹{history['Close'].iloc[-1]:,.2f}"
)


# ============================================================
# HISTORICAL PRICE CHART
# ============================================================

st.subheader(
    "📊 Historical Reliance Stock Price"
)

fig_history, ax_history = (
    plt.subplots(
        figsize=(12, 5)
    )
)

recent_history = history.tail(
    250
)

ax_history.plot(
    recent_history["Date"],
    recent_history["Close"],
    label="Historical Close"
)

ax_history.set_xlabel(
    "Date"
)

ax_history.set_ylabel(
    "Close Price (₹)"
)

ax_history.set_title(
    "Reliance Industries Historical Close Price"
)

ax_history.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(
    fig_history
)

plt.close(
    fig_history
)


# ============================================================
# RUN FORECAST
# ============================================================

if run_forecast:

    try:

        with st.spinner(
            "Generating stock price forecast..."
        ):

            # ------------------------------------------------
            # Linear Regression Forecast
            # ------------------------------------------------

            forecast_lr = (
                recursive_forecast(
                    history,
                    model,
                    scaler,
                    features,
                    horizon
                )
            )

            # ------------------------------------------------
            # ARIMA Forecast
            # ------------------------------------------------

            if show_arima:

                forecast_arima = (
                    arima_forecast(
                        history,
                        horizon
                    )
                )

            else:

                forecast_arima = None

        # ----------------------------------------------------
        # Latest and predicted prices
        # ----------------------------------------------------

        latest_close = (
            history["Close"]
            .iloc[-1]
        )

        next_day_prediction = (
            forecast_lr[
                "Predicted_Close_LR"
            ]
            .iloc[0]
        )

        final_prediction = (
            forecast_lr[
                "Predicted_Close_LR"
            ]
            .iloc[-1]
        )

        next_day_change = (
            next_day_prediction
            - latest_close
        )

        final_change = (
            final_prediction
            - latest_close
        )

        # ====================================================
        # METRICS
        # ====================================================

        st.subheader(
            "📌 Prediction Summary"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Latest Close",
            f"₹{latest_close:,.2f}"
        )

        c2.metric(
            "Next-Day Prediction",
            f"₹{next_day_prediction:,.2f}",
            f"₹{next_day_change:+,.2f}"
        )

        c3.metric(
            f"Day {horizon} Prediction",
            f"₹{final_prediction:,.2f}",
            f"₹{final_change:+,.2f}"
        )

        percentage_change = (
            (
                final_prediction
                - latest_close
            )
            / latest_close
        ) * 100

        c4.metric(
            "Expected Change",
            f"{percentage_change:+.2f}%"
        )

        # ====================================================
        # FORECAST CHART
        # ====================================================

        st.subheader(
            f"📈 {horizon}-Day Forecast"
        )

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        recent = history.tail(
            120
        )

        # Historical price

        ax.plot(
            recent["Date"],
            recent["Close"],
            label="Historical Close"
        )

        # Linear Regression

        ax.plot(
            forecast_lr["Date"],
            forecast_lr[
                "Predicted_Close_LR"
            ],
            label="Linear Regression Forecast",
            marker="o",
            markersize=3
        )

        # ARIMA

        if forecast_arima is not None:

            ax.plot(
                forecast_arima["Date"],
                forecast_arima[
                    "Predicted_Close_ARIMA"
                ],
                label="ARIMA Forecast",
                marker="o",
                markersize=3
            )

            ax.fill_between(
                forecast_arima["Date"],
                forecast_arima[
                    "ARIMA_Lower_95"
                ],
                forecast_arima[
                    "ARIMA_Upper_95"
                ],
                alpha=0.15,
                label="ARIMA 95% Confidence Interval"
            )

        # Forecast start line

        ax.axvline(
            history["Date"].max(),
            linestyle="--",
            linewidth=1,
            label="Forecast Start"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Close Price (₹)"
        )

        ax.set_title(
            f"Reliance Industries - "
            f"{horizon}-Day Stock Forecast"
        )

        ax.legend()

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        # ====================================================
        # FORECAST TABLE
        # ====================================================

        st.subheader(
            "📋 Forecast Table"
        )

        if forecast_arima is not None:

            forecast_table = (
                forecast_lr.merge(
                    forecast_arima,
                    on="Date"
                )
            )

        else:

            forecast_table = (
                forecast_lr.copy()
            )

        st.dataframe(
            forecast_table,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD CSV
        # ====================================================

        csv_data = (
            forecast_table
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv_data,
            file_name=(
                f"reliance_"
                f"{horizon}_day_forecast.csv"
            ),
            mime="text/csv"
        )

        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.subheader(
            "📌 Forecast Interpretation"
        )

        if final_prediction > latest_close:

            st.success(
                f"Linear Regression predicts that "
                f"the price may increase from "
                f"₹{latest_close:,.2f} to "
                f"₹{final_prediction:,.2f} "
                f"over the selected forecast horizon."
            )

        elif final_prediction < latest_close:

            st.warning(
                f"Linear Regression predicts that "
                f"the price may decrease from "
                f"₹{latest_close:,.2f} to "
                f"₹{final_prediction:,.2f} "
                f"over the selected forecast horizon."
            )

        else:

            st.info(
                "The predicted price is approximately "
                "equal to the latest closing price."
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.caption(
            "⚠️ This application provides a statistical "
            "forecast based on historical price behavior. "
            "It does not consider news, company announcements, "
            "earnings, market sentiment, or macroeconomic events. "
            "Predictions become less reliable as the forecast "
            "horizon increases. This is not financial advice."
        )

    except Exception as error:

        st.error(
            "❌ An error occurred while generating "
            "the forecast."
        )

        st.exception(
            error
        )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👈 Select the forecast horizon and click "
        "**🚀 Run Forecast** to generate predictions."
    )