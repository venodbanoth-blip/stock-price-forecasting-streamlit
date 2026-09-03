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
# FILE CONFIGURATION
# ============================================================

DATA_FILE = "Company stock prices (1).xlsx"

REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

    # --------------------------------------------------------
    # READ EXCEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # DATE CONVERSION
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
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
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Date",
            "Close"
        ]
    )

    # --------------------------------------------------------
    # SORT BY DATE
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE DATES
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # MINIMUM DATA CHECK
    # --------------------------------------------------------

    if len(df) < 60:

        raise ValueError(
            "At least 60 valid historical rows are required."
        )

    return df


# ============================================================
# PREPARE CLOSE VALUES FOR ARIMA
# ============================================================

def prepare_close_series(history):

    df = history.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Date",
            "Close"
        ]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if len(df) < 30:

        raise ValueError(
            "At least 30 observations are required for ARIMA."
        )

    # --------------------------------------------------------
    # IMPORTANT ARIMA FIX
    #
    # Return NumPy array instead of Pandas Series with
    # DatetimeIndex.
    #
    # This prevents:
    # ValueError: No supported index is available.
    # --------------------------------------------------------

    close_values = (
        df["Close"]
        .astype(float)
        .to_numpy()
    )

    return close_values


# ============================================================
# FIT ARIMA MODEL
# ============================================================

def fit_arima_model(close_values):

    # --------------------------------------------------------
    # DATA CHECK
    # --------------------------------------------------------

    if close_values is None:

        raise ValueError(
            "Close price data is empty."
        )

    if len(close_values) < 30:

        raise ValueError(
            "At least 30 observations are required for ARIMA."
        )

    # --------------------------------------------------------
    # REMOVE INVALID NUMBERS
    # --------------------------------------------------------

    close_values = np.asarray(
        close_values,
        dtype=float
    )

    if not np.all(
        np.isfinite(close_values)
    ):

        raise ValueError(
            "Close price data contains invalid values."
        )

    # --------------------------------------------------------
    # ARIMA MODEL
    # --------------------------------------------------------

    model = ARIMA(
        close_values,
        order=(5, 1, 0)
    )

    fitted_model = model.fit()

    return fitted_model


# ============================================================
# FUTURE ARIMA FORECAST
# ============================================================

def arima_forecast(
    history,
    n_days
):

    # --------------------------------------------------------
    # VALIDATE HORIZON
    # --------------------------------------------------------

    if n_days < 1:

        raise ValueError(
            "Forecast horizon must be at least 1 day."
        )

    # --------------------------------------------------------
    # PREPARE CLOSE VALUES
    # --------------------------------------------------------

    close_values = prepare_close_series(
        history
    )

    # --------------------------------------------------------
    # FIT MODEL
    # --------------------------------------------------------

    fitted_model = fit_arima_model(
        close_values
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_result = fitted_model.get_forecast(
        steps=n_days
    )

    # --------------------------------------------------------
    # PREDICTED VALUES
    # --------------------------------------------------------

    forecast = np.asarray(
        forecast_result.predicted_mean,
        dtype=float
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    confidence = forecast_result.conf_int(
        alpha=0.05
    )

    lower_bound = np.asarray(
        confidence.iloc[:, 0],
        dtype=float
    )

    upper_bound = np.asarray(
        confidence.iloc[:, 1],
        dtype=float
    )

    # --------------------------------------------------------
    # LAST HISTORICAL DATE
    # --------------------------------------------------------

    last_date = pd.to_datetime(
        history["Date"].max()
    )

    # --------------------------------------------------------
    # FUTURE BUSINESS DATES
    #
    # NOTE:
    # bdate_range excludes Saturday/Sunday.
    # It does not account for NSE holidays.
    # --------------------------------------------------------

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days
    )

    # --------------------------------------------------------
    # RESULT DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame({

        "Date": future_dates,

        "Predicted_Close_ARIMA": forecast,

        "ARIMA_Lower_95": lower_bound,

        "ARIMA_Upper_95": upper_bound

    })

    return result


# ============================================================
# ARIMA BACKTEST
# ============================================================

def arima_backtest(
    history,
    test_days
):

    # --------------------------------------------------------
    # CHECK DATA SIZE
    # --------------------------------------------------------

    if len(history) <= test_days + 30:

        raise ValueError(
            f"Not enough data for a {test_days}-day backtest. "
            f"At least {test_days + 31} rows are recommended."
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
        actual_data["Date"],
        errors="coerce"
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

    actual_data = actual_data.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # PREPARE TRAINING DATA
    # --------------------------------------------------------

    train_values = prepare_close_series(
        train_data
    )

    # --------------------------------------------------------
    # FIT ARIMA
    # --------------------------------------------------------

    fitted_model = fit_arima_model(
        train_values
    )

    # --------------------------------------------------------
    # FORECAST TEST PERIOD
    # --------------------------------------------------------

    forecast_result = fitted_model.get_forecast(
        steps=test_days
    )

    predictions = np.asarray(
        forecast_result.predicted_mean,
        dtype=float
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    confidence = forecast_result.conf_int(
        alpha=0.05
    )

    lower_bound = np.asarray(
        confidence.iloc[:, 0],
        dtype=float
    )

    upper_bound = np.asarray(
        confidence.iloc[:, 1],
        dtype=float
    )

    # --------------------------------------------------------
    # PROTECT AGAINST LENGTH MISMATCH
    # --------------------------------------------------------

    usable_length = min(
        len(actual_data),
        len(predictions)
    )

    actual_data = actual_data.iloc[
        :usable_length
    ].copy()

    predictions = predictions[
        :usable_length
    ]

    lower_bound = lower_bound[
        :usable_length
    ]

    upper_bound = upper_bound[
        :usable_length
    ]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = pd.DataFrame({

        "Date": actual_data["Date"].to_numpy(),

        "Actual_Close":
            actual_data["Close"].to_numpy(),

        "Predicted_Close":
            predictions,

        "ARIMA_Lower_95":
            lower_bound,

        "ARIMA_Upper_95":
            upper_bound

    })

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

    # --------------------------------------------------------
    # PERCENTAGE ERROR
    # --------------------------------------------------------

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

    valid_mape = (
        result["Percentage_Error"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(valid_mape) > 0:

        mape = valid_mape.mean()

    else:

        mape = np.nan

    metrics = {

        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape)

    }

    return result, metrics


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

    "Forecast Horizon (business days)",

    min_value=1,

    max_value=60,

    value=30
)


# ============================================================
# BACKTEST DAYS
# ============================================================

backtest_days = st.sidebar.selectbox(

    "Historical Backtest",

    [30, 60],

    index=0
)


# ============================================================
# BUTTONS
# ============================================================

run_forecast = st.sidebar.button(

    "🚀 Run Forecast",

    use_container_width=True
)

run_backtest = st.sidebar.button(

    "🔍 Run Backtest",

    use_container_width=True
)


# ============================================================
# FILE CHECK
# ============================================================

if (
    uploaded_file is None
    and not os.path.exists(DATA_FILE)
):

    st.error(
        f"Excel file not found: {DATA_FILE}"
    )

    st.info(
        "Please upload your Excel file from the sidebar."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    history = load_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Error loading Excel file: {e}"
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.subheader(
    "📊 Historical Data"
)

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# ROWS
# ------------------------------------------------------------

with col1:

    st.metric(
        "Rows",
        len(history)
    )


# ------------------------------------------------------------
# START DATE
# ------------------------------------------------------------

with col2:

    st.metric(

        "Start Date",

        history["Date"]
        .min()
        .strftime("%d-%m-%Y")

    )


# ------------------------------------------------------------
# END DATE
# ------------------------------------------------------------

with col3:

    st.metric(

        "End Date",

        history["Date"]
        .max()
        .strftime("%d-%m-%Y")

    )


# ------------------------------------------------------------
# LATEST CLOSE
# ------------------------------------------------------------

with col4:

    latest_close = float(
        history["Close"].iloc[-1]
    )

    st.metric(
        "Latest Close",
        f"₹{latest_close:,.2f}"
    )


# ============================================================
# HISTORICAL CHART
# ============================================================

st.subheader(
    "📈 Historical Reliance Closing Price"
)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

recent_history = history.tail(
    min(250, len(history))
)

ax.plot(

    recent_history["Date"],

    recent_history["Close"],

    label="Actual Close",

    linewidth=2

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

plt.tight_layout()

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

                if np.isfinite(
                    metrics["MAPE"]
                ):

                    st.metric(

                        "MAPE",

                        f"{metrics['MAPE']:.2f}%"

                    )

                else:

                    st.metric(
                        "MAPE",
                        "N/A"
                    )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            if np.isfinite(
                metrics["MAPE"]
            ):

                if metrics["MAPE"] > 20:

                    st.error(

                        "⚠️ ARIMA has high forecasting "
                        "error on this historical period. "
                        "Future forecasts should be treated "
                        "as low-confidence."

                    )

                elif metrics["MAPE"] > 10:

                    st.warning(

                        "⚠️ ARIMA has moderate forecasting "
                        "error on this historical period."

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

                backtest_result["ARIMA_Lower_95"].to_numpy(),
                backtest_result["ARIMA_Upper_95"].to_numpy(),

                alpha=0.15,

                label="95% Confidence Interval"

            )

            ax2.set_title(

                f"ARIMA Backtest - Last "
                f"{backtest_days} Trading Days"

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

            plt.tight_layout()

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
            # DOWNLOAD BACKTEST
            # ------------------------------------------------

            backtest_csv = (
                backtest_result
                .to_csv(index=False)
            )

            st.download_button(

                "⬇️ Download Backtest CSV",

                backtest_csv,

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


# ============================================================
# FUTURE FORECAST
# ============================================================

if run_forecast:

    st.subheader(
        f"🔮 Next {horizon} Business-Day ARIMA Forecast"
    )

    with st.spinner(
        "Generating ARIMA forecast..."
    ):

        try:

            # ------------------------------------------------
            # GENERATE FORECAST
            # ------------------------------------------------

            forecast_result = arima_forecast(

                history,

                horizon

            )


            # ------------------------------------------------
            # CURRENT PRICE
            # ------------------------------------------------

            latest_price = float(

                history["Close"].iloc[-1]

            )


            # ------------------------------------------------
            # NEXT DAY PRICE
            # ------------------------------------------------

            next_day_price = float(

                forecast_result[
                    "Predicted_Close_ARIMA"
                ].iloc[0]

            )


            # ------------------------------------------------
            # FINAL FORECAST PRICE
            # ------------------------------------------------

            final_price = float(

                forecast_result[
                    "Predicted_Close_ARIMA"
                ].iloc[-1]

            )


            # ------------------------------------------------
            # EXPECTED CHANGE
            # ------------------------------------------------

            if latest_price != 0:

                expected_change = (

                    (
                        final_price
                        - latest_price
                    )

                    / latest_price

                    * 100

                )

            else:

                expected_change = np.nan


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

                if np.isfinite(
                    expected_change
                ):

                    st.metric(

                        "Expected Change",

                        f"{expected_change:+.2f}%"

                    )

                else:

                    st.metric(
                        "Expected Change",
                        "N/A"
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

            chart_history = history.tail(
                min(120, len(history))
            )


            # ------------------------------------------------
            # HISTORICAL PRICE
            # ------------------------------------------------

            ax3.plot(

                chart_history["Date"],

                chart_history["Close"],

                label="Historical Close",

                linewidth=2

            )


            # ------------------------------------------------
            # FORECAST
            # ------------------------------------------------

            ax3.plot(

                forecast_result["Date"],

                forecast_result[
                    "Predicted_Close_ARIMA"
                ],

                label="ARIMA Forecast",

                linewidth=2,

                linestyle="--"

            )


            # ------------------------------------------------
            # CONFIDENCE INTERVAL
            # ------------------------------------------------

            ax3.fill_between(

                forecast_result["Date"],

                forecast_result[
                    "ARIMA_Lower_95"
                ].to_numpy(),

                forecast_result[
                    "ARIMA_Upper_95"
                ].to_numpy(),

                alpha=0.15,

                label="ARIMA 95% Confidence Interval"

            )


            # ------------------------------------------------
            # FORECAST START
            # ------------------------------------------------

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

            plt.tight_layout()

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

            st.dataframe(

                forecast_result.style.format({

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
            # DOWNLOAD FORECAST
            # =================================================

            forecast_csv = (

                forecast_result
                .to_csv(index=False)

            )

            st.download_button(

                "⬇️ Download Forecast CSV",

                forecast_csv,

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

            if np.isfinite(
                expected_change
            ):

                if expected_change > 5:

                    st.success(

                        f"📈 The ARIMA model forecasts "
                        f"an approximately "
                        f"{expected_change:.2f}% increase "
                        f"over the next {horizon} "
                        f"business days."

                    )

                elif expected_change < -5:

                    st.warning(

                        f"📉 The ARIMA model forecasts "
                        f"an approximately "
                        f"{abs(expected_change):.2f}% decrease "
                        f"over the next {horizon} "
                        f"business days."

                    )

                else:

                    st.info(

                        f"➡️ The ARIMA model forecasts "
                        f"relatively stable movement "
                        f"over the next {horizon} "
                        f"business days."

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


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(

    "Reliance Industries Stock Price Forecasting | "
    "ARIMA Time Series Analysis | "
    "Educational Project"

)
