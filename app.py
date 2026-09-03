import os
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
st.caption(
"ARIMA Time Series Forecasting | Historical Backtesting | Future Forecast"
)

# ============================================================

# FILE

# ============================================================

DATA_FILE = "Company stock prices (1).xlsx"

# ============================================================

# LOAD DATA

# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

```
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
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )

# --------------------------------------------------------
# DATE
# --------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# --------------------------------------------------------
# NUMERIC COLUMNS
# --------------------------------------------------------

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# --------------------------------------------------------
# CLEAN
# --------------------------------------------------------

df = df.dropna(
    subset=[
        "Date",
        "Close"
    ]
)

df = df.sort_values(
    "Date"
)

df = df.drop_duplicates(
    subset=["Date"]
)

df = df.reset_index(
    drop=True
)

if len(df) < 60:

    raise ValueError(
        "At least 60 historical rows are required."
    )

return df
```

# ============================================================

# PREPARE TIME SERIES

# ============================================================

def prepare_close_series(history):

```
df = history.copy()

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Date",
        "Close"
    ]
)

df = df.sort_values(
    "Date"
)

df = df.drop_duplicates(
    subset=["Date"]
)

# IMPORTANT:
# Set Date as DatetimeIndex.
# This fixes:
# ValueError: No supported index is available.

df = df.set_index(
    "Date"
)

close_series = df["Close"].astype(float)

return close_series
```

# ============================================================

# ARIMA MODEL

# ============================================================

def fit_arima_model(close_series):

```
if len(close_series) < 30:

    raise ValueError(
        "At least 30 observations are required for ARIMA."
    )

model = ARIMA(
    close_series,
    order=(5, 1, 0)
)

fitted_model = model.fit()

return fitted_model
```

# ============================================================

# ARIMA FUTURE FORECAST

# ============================================================

def arima_forecast(
history,
n_days
):

```
close_series = prepare_close_series(
    history
)

fitted_model = fit_arima_model(
    close_series
)

forecast_result = fitted_model.get_forecast(
    steps=n_days
)

forecast = forecast_result.predicted_mean

confidence = forecast_result.conf_int(
    alpha=0.05
)

# --------------------------------------------------------
# FUTURE BUSINESS DATES
# --------------------------------------------------------

future_dates = pd.bdate_range(
    start=close_series.index[-1]
    + pd.Timedelta(days=1),
    periods=n_days
)

result = pd.DataFrame({

    "Date": future_dates,

    "Predicted_Close_ARIMA":
        forecast.to_numpy(),

    "ARIMA_Lower_95":
        confidence.iloc[:, 0].to_numpy(),

    "ARIMA_Upper_95":
        confidence.iloc[:, 1].to_numpy()

})

return result
```

# ============================================================

# ARIMA BACKTEST

# ============================================================

def arima_backtest(
history,
test_days
):

```
if len(history) <= test_days + 30:

    raise ValueError(
        f"Not enough data for a {test_days}-day backtest."
    )

# --------------------------------------------------------
# TRAIN DATA
# --------------------------------------------------------

train_data = history.iloc[
    :-test_days
].copy()

# --------------------------------------------------------
# ACTUAL TEST DATA
# --------------------------------------------------------

actual_data = history.iloc[
    -test_days:
].copy()

actual_data["Date"] = pd.to_datetime(
    actual_data["Date"]
)

actual_data["Close"] = pd.to_numeric(
    actual_data["Close"],
    errors="coerce"
)

actual_data = actual_data.dropna(
    subset=[
        "Date",
        "Close"
    ]
)

# --------------------------------------------------------
# TRAIN ARIMA
# --------------------------------------------------------

train_series = prepare_close_series(
    train_data
)

fitted_model = fit_arima_model(
    train_series
)

# --------------------------------------------------------
# FORECAST
# --------------------------------------------------------

forecast_result = fitted_model.get_forecast(
    steps=test_days
)

predictions = forecast_result.predicted_mean

# --------------------------------------------------------
# CONFIDENCE INTERVAL
# --------------------------------------------------------

confidence = forecast_result.conf_int(
    alpha=0.05
)

# --------------------------------------------------------
# RESULT
# --------------------------------------------------------

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

result = result.reset_index(
    drop=True
)

result["Predicted_Close"] = (
    predictions.to_numpy()
)

result["ARIMA_Lower_95"] = (
    confidence.iloc[:, 0].to_numpy()
)

result["ARIMA_Upper_95"] = (
    confidence.iloc[:, 1].to_numpy()
)

# --------------------------------------------------------
# ERROR
# --------------------------------------------------------

result["Error"] = (
    result["Predicted_Close"]
    - result["Actual_Close"]
)

result["Absolute_Error"] = (
    result["Error"].abs()
)

# Avoid division by zero
result["Percentage_Error"] = np.where(
    result["Actual_Close"] != 0,

    (
        result["Absolute_Error"]
        / result["Actual_Close"]
        * 100
    ),

    np.nan
)

# --------------------------------------------------------
# METRICS
# --------------------------------------------------------

mae = (
    result["Absolute_Error"]
    .mean()
)

rmse = np.sqrt(
    np.mean(
        result["Error"] ** 2
    )
)

mape = (
    result["Percentage_Error"]
    .mean()
)

metrics = {

    "MAE": mae,

    "RMSE": rmse,

    "MAPE": mape

}

return result, metrics
```

# ============================================================

# SIDEBAR

# ============================================================

st.sidebar.header(
"⚙️ Forecast Settings"
)

# ============================================================

# FILE UPLOAD

# ============================================================

uploaded_file = st.sidebar.file_uploader(
"Upload Excel file",
type=["xlsx"]
)

# ============================================================

# FORECAST HORIZON

# ============================================================

horizon = st.sidebar.slider(

```
"Forecast Horizon (business days)",

min_value=1,

max_value=60,

value=30
```

)

# ============================================================

# BACKTEST DAYS

# ============================================================

backtest_days = st.sidebar.selectbox(

```
"Historical Backtest",

[30, 60],

index=0
```

)

# ============================================================

# BUTTONS

# ============================================================

run_forecast = st.sidebar.button(

```
"🚀 Run Forecast",

use_container_width=True
```

)

run_backtest = st.sidebar.button(

```
"🔍 Run Backtest",

use_container_width=True
```

)

# ============================================================

# FILE CHECK

# ============================================================

if (
uploaded_file is None
and not os.path.exists(DATA_FILE)
):

```
st.error(
    f"Excel file not found: {DATA_FILE}"
)

st.info(
    "Please upload your Excel file from the sidebar."
)

st.stop()
```

# ============================================================

# LOAD DATA

# ============================================================

try:

```
history = load_data(
    uploaded_file
)
```

except Exception as e:

```
st.error(
    f"Error loading Excel file: {e}"
)

st.stop()
```

# ============================================================

# DATA INFORMATION

# ============================================================

st.subheader(
"📊 Historical Data"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

```
st.metric(
    "Rows",
    len(history)
)
```

with col2:

```
st.metric(
    "Start Date",
    history["Date"]
    .min()
    .strftime("%d-%m-%Y")
)
```

with col3:

```
st.metric(
    "End Date",
    history["Date"]
    .max()
    .strftime("%d-%m-%Y")
)
```

with col4:

```
latest_close = float(
    history["Close"].iloc[-1]
)

st.metric(
    "Latest Close",
    f"₹{latest_close:,.2f}"
)
```

# ============================================================

# HISTORICAL CHART

# ============================================================

st.subheader(
"📈 Historical Reliance Closing Price"
)

fig, ax = plt.subplots(
figsize=(14, 5)
)

recent_history = history.tail(250)

ax.plot(

```
recent_history["Date"],

recent_history["Close"],

label="Actual Close",

linewidth=2
```

)

ax.set_xlabel(
"Date"
)

ax.set_ylabel(
"Price (₹)"
)

ax.set_title(
"Reliance Industries Historical Closing Price"
)

ax.legend()

ax.grid(
alpha=0.3
)

plt.xticks(
rotation=45
)

st.pyplot(
fig
)

plt.close(
fig
)

# ============================================================

# BACKTEST

# ============================================================

if run_backtest:

```
st.subheader(
    f"🔍 Last {backtest_days}-Trading-Day ARIMA Backtest"
)

with st.spinner(
    f"Testing ARIMA on the last {backtest_days} trading days..."
):

    try:

        backtest_result, metrics = arima_backtest(

            history,

            backtest_days

        )

        # ------------------------------------------------
        # METRICS
        # ------------------------------------------------

        c1, c2, c3 = st.columns(3)


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


        # ------------------------------------------------
        # INTERPRETATION
        # ------------------------------------------------

        if metrics["MAPE"] > 20:

            st.error(
                "⚠️ ARIMA has high forecasting error "
                "on this historical period. "
                "Future forecasts should be treated "
                "as low-confidence."
            )

        elif metrics["MAPE"] > 10:

            st.warning(
                "⚠️ ARIMA has moderate forecasting error "
                "on this historical period."
            )

        else:

            st.success(
                "✅ ARIMA shows relatively low "
                "historical forecasting error."
            )


        # ------------------------------------------------
        # BACKTEST CHART
        # ------------------------------------------------

        st.subheader(
            "📉 Actual vs ARIMA Predicted Price"
        )


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

            label="ARIMA Predicted Price",

            linestyle="--",

            linewidth=2
        )


        ax2.fill_between(

            backtest_result["Date"],

            backtest_result["ARIMA_Lower_95"],

            backtest_result["ARIMA_Upper_95"],

            alpha=0.15,

            label="95% Confidence Interval"
        )


        ax2.set_title(

            f"ARIMA Backtest - Last {backtest_days} Trading Days"
        )


        ax2.set_xlabel(
            "Date"
        )


        ax2.set_ylabel(
            "Price (₹)"
        )


        ax2.legend()


        ax2.grid(
            alpha=0.3
        )


        plt.xticks(
            rotation=45
        )


        st.pyplot(
            fig2
        )


        plt.close(
            fig2
        )


        # ------------------------------------------------
        # TABLE
        # ------------------------------------------------

        st.subheader(
            "📋 Backtest Results"
        )


        st.dataframe(

            backtest_result.style.format({

                "Actual_Close":
                    "₹{:.2f}",

                "Predicted_Close":
                    "₹{:.2f}",

                "ARIMA_Lower_95":
                    "₹{:.2f}",

                "ARIMA_Upper_95":
                    "₹{:.2f}",

                "Error":
                    "₹{:.2f}",

                "Absolute_Error":
                    "₹{:.2f}",

                "Percentage_Error":
                    "{:.2f}%"

            }),

            use_container_width=True

        )


        # ------------------------------------------------
        # DOWNLOAD
        # ------------------------------------------------

        csv = backtest_result.to_csv(
            index=False
        )


        st.download_button(

            "⬇️ Download Backtest CSV",

            csv,

            file_name=(
                f"reliance_arima_backtest_"
                f"{backtest_days}_days.csv"
            ),

            mime="text/csv"

        )


    except Exception as e:

        st.error(
            f"Backtest error: {e}"
        )
```

# ============================================================

# FUTURE FORECAST

# ============================================================

if run_forecast:

```
st.subheader(
    f"🔮 Next {horizon} Business-Day ARIMA Forecast"
)

with st.spinner(
    "Generating ARIMA forecast..."
):

    try:

        # ------------------------------------------------
        # ARIMA FORECAST
        # ------------------------------------------------

        forecast_result = arima_forecast(

            history,

            horizon

        )


        # ------------------------------------------------
        # FORECAST METRICS
        # ------------------------------------------------

        latest_price = float(
            history["Close"].iloc[-1]
        )


        next_day_price = float(

            forecast_result[
                "Predicted_Close_ARIMA"
            ].iloc[0]

        )


        final_price = float(

            forecast_result[
                "Predicted_Close_ARIMA"
            ].iloc[-1]

        )


        expected_change = (

            (
                final_price
                - latest_price
            )

            / latest_price

            * 100

        )


        # ------------------------------------------------
        # TOP METRICS
        # ------------------------------------------------

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


        # =================================================
        # FORECAST CHART
        # =================================================

        st.subheader(
            "📈 ARIMA Forecast Chart"
        )


        fig3, ax3 = plt.subplots(
            figsize=(15, 7)
        )


        chart_history = history.tail(120)


        # Historical price

        ax3.plot(

            chart_history["Date"],

            chart_history["Close"],

            label="Historical Close",

            linewidth=2

        )


        # Forecast

        ax3.plot(

            forecast_result["Date"],

            forecast_result[
                "Predicted_Close_ARIMA"
            ],

            label="ARIMA Forecast",

            linewidth=2,

            linestyle="--"

        )


        # Confidence interval

        ax3.fill_between(

            forecast_result["Date"],

            forecast_result[
                "ARIMA_Lower_95"
            ],

            forecast_result[
                "ARIMA_Upper_95"
            ],

            alpha=0.15,

            label="ARIMA 95% Confidence Interval"

        )


        # Forecast start

        ax3.axvline(

            history["Date"].max(),

            linestyle="--",

            linewidth=1.5,

            label="Forecast Start"

        )


        ax3.set_title(

            f"Reliance Industries - "
            f"{horizon} Business-Day ARIMA Forecast"

        )


        ax3.set_xlabel(
            "Date"
        )


        ax3.set_ylabel(
            "Price (₹)"
        )


        ax3.legend()


        ax3.grid(
            alpha=0.3
        )


        plt.xticks(
            rotation=45
        )


        st.pyplot(
            fig3
        )


        plt.close(
            fig3
        )


        # =================================================
        # FORECAST TABLE
        # =================================================

        st.subheader(
            "📋 Forecast Values"
        )


        display_forecast = (
            forecast_result.copy()
        )


        st.dataframe(

            display_forecast.style.format({

                "Predicted_Close_ARIMA":
                    "₹{:.2f}",

                "ARIMA_Lower_95":
                    "₹{:.2f}",

                "ARIMA_Upper_95":
                    "₹{:.2f}"

            }),

            use_container_width=True

        )


        # =================================================
        # DOWNLOAD
        # =================================================

        csv = forecast_result.to_csv(
            index=False
        )


        st.download_button(

            "⬇️ Download Forecast CSV",

            csv,

            file_name=(

                f"reliance_arima_forecast_"
                f"{horizon}_days.csv"

            ),

            mime="text/csv"

        )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader(
            "🧠 Forecast Interpretation"
        )


        if expected_change > 5:

            st.success(

                f"📈 The ARIMA model forecasts an "
                f"approximately {expected_change:.2f}% "
                f"increase over the next "
                f"{horizon} business days."

            )


        elif expected_change < -5:

            st.warning(

                f"📉 The ARIMA model forecasts an "
                f"approximately {abs(expected_change):.2f}% "
                f"decrease over the next "
                f"{horizon} business days."

            )


        else:

            st.info(

                f"➡️ The ARIMA model forecasts a "
                f"relatively stable movement over "
                f"the next {horizon} business days."

            )


        # =================================================
        # CONFIDENCE INFORMATION
        # =================================================

        lower_final = float(

            forecast_result[
                "ARIMA_Lower_95"
            ].iloc[-1]

        )


        upper_final = float(

            forecast_result[
                "ARIMA_Upper_95"
            ].iloc[-1]

        )


        st.info(

            f"📊 Day {horizon} ARIMA estimate: "
            f"₹{final_price:,.2f}. "
            f"The approximate 95% confidence interval "
            f"is ₹{lower_final:,.2f} to "
            f"₹{upper_final:,.2f}."

        )


        # =================================================
        # WARNING
        # =================================================

        st.warning(

            "⚠️ Stock-market forecasts are uncertain. "
            "ARIMA uses historical price patterns and "
            "cannot predict unexpected news, market "
            "events, corporate announcements, global "
            "economic changes, or sudden volatility. "
            "This application is for educational and "
            "research purposes only."

        )


    except Exception as e:

        st.error(
            f"Forecast error: {e}"
        )
```

# ============================================================

# FOOTER

# ============================================================

st.markdown("---")

st.caption(

```
"Reliance Industries Stock Price Forecasting | "
"ARIMA Time Series Analysis | "
"Educational Project"
```

)
