import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Reliance Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Reliance Industries Stock Price Forecasting")

st.caption(
    "Automatic ARIMA Model Selection | "
    "MAE | RMSE | MAPE | Historical Backtesting | Future Forecast"
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
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
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
    # REMOVE NON-POSITIVE CLOSE
    # --------------------------------------------------------

    df = df[
        df["Close"] > 0
    ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by="Date"
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
# PREPARE CLOSE VALUES
# ============================================================

def prepare_close_values(history):

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
    # REMOVE NON-POSITIVE VALUES
    # --------------------------------------------------------

    df = df[
        df["Close"] > 0
    ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by="Date"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    # --------------------------------------------------------
    # CHECK LENGTH
    # --------------------------------------------------------

    if len(df) < 30:

        raise ValueError(
            "At least 30 observations are required."
        )

    # --------------------------------------------------------
    # RETURN NUMPY ARRAY
    # --------------------------------------------------------

    values = np.asarray(
        df["Close"],
        dtype=float
    )

    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(values)
    ):

        raise ValueError(
            "Close price contains invalid values."
        )

    return values


# ============================================================
# ARIMA ORDERS TO TEST
# ============================================================

ARIMA_ORDERS = [

    (0, 1, 0),

    (0, 1, 1),

    (1, 1, 0),

    (1, 1, 1),

    (1, 1, 2),

    (2, 1, 0),

    (2, 1, 1),

    (2, 1, 2),

    (3, 1, 0),

    (3, 1, 1),

    (3, 1, 2),

    (4, 1, 0),

    (4, 1, 1),

    (4, 1, 2),

    (5, 1, 0),

    (5, 1, 1),

    (5, 1, 2)

]


# ============================================================
# FIT ARIMA
# ============================================================

def fit_arima_model(
    close_values,
    order
):

    close_values = np.asarray(
        close_values,
        dtype=float
    )

    if len(close_values) < 30:

        raise ValueError(
            "At least 30 observations are required."
        )

    if not np.all(
        np.isfinite(close_values)
    ):

        raise ValueError(
            "Invalid values found in close prices."
        )

    model = ARIMA(
        close_values,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit()

    return fitted_model


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    length = min(
        len(actual),
        len(predicted)
    )

    actual = actual[:length]

    predicted = predicted[:length]

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    error = (
        predicted
        - actual
    )

    absolute_error = np.abs(
        error
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        absolute_error
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    non_zero_mask = (
        actual != 0
    )

    if np.any(
        non_zero_mask
    ):

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    - predicted[non_zero_mask]
                )
                / actual[non_zero_mask]
            )
        ) * 100

    else:

        mape = np.nan

    return {

        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape)

    }


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def get_confidence_interval(
    forecast_result
):

    confidence = forecast_result.conf_int(
        alpha=0.05
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert DataFrame or NumPy array to NumPy.
    # --------------------------------------------------------

    confidence = np.asarray(
        confidence,
        dtype=float
    )

    if confidence.ndim != 2:

        raise ValueError(
            "Unexpected confidence interval format."
        )

    if confidence.shape[1] < 2:

        raise ValueError(
            "Confidence interval does not contain two columns."
        )

    lower_bound = confidence[:, 0]

    upper_bound = confidence[:, 1]

    return (
        lower_bound,
        upper_bound
    )


# ============================================================
# AUTOMATIC ARIMA MODEL SELECTION
# ============================================================

def select_best_arima(
    train_values,
    validation_days
):

    train_values = np.asarray(
        train_values,
        dtype=float
    )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if len(train_values) <= validation_days + 30:

        raise ValueError(
            "Not enough observations for ARIMA model selection."
        )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    model_train = train_values[
        :-validation_days
    ]

    validation_actual = train_values[
        -validation_days:
    ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # TEST EACH ARIMA MODEL
    # --------------------------------------------------------

    for order in ARIMA_ORDERS:

        try:

            model = fit_arima_model(
                model_train,
                order
            )

            forecast_result = model.get_forecast(
                steps=validation_days
            )

            predictions = np.asarray(
                forecast_result.predicted_mean,
                dtype=float
            )

            metrics = calculate_metrics(
                validation_actual,
                predictions
            )

            results.append({

                "ARIMA_Order":
                    str(order),

                "p":
                    order[0],

                "d":
                    order[1],

                "q":
                    order[2],

                "MAE":
                    metrics["MAE"],

                "RMSE":
                    metrics["RMSE"],

                "MAPE":
                    metrics["MAPE"],

                "Status":
                    "Success"

            })

        except Exception as e:

            results.append({

                "ARIMA_Order":
                    str(order),

                "p":
                    order[0],

                "d":
                    order[1],

                "q":
                    order[2],

                "MAE":
                    np.nan,

                "RMSE":
                    np.nan,

                "MAPE":
                    np.nan,

                "Status":
                    f"Failed: {str(e)[:60]}"

            })

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # REMOVE FAILED MODELS
    # --------------------------------------------------------

    valid_results = comparison[
        comparison["MAPE"].notna()
    ].copy()

    if len(valid_results) == 0:

        raise ValueError(
            "None of the tested ARIMA models could be fitted."
        )

    # --------------------------------------------------------
    # SORT BY MAPE
    # --------------------------------------------------------

    valid_results = valid_results.sort_values(
        by=[
            "MAPE",
            "RMSE",
            "MAE"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # BEST ORDER
    # --------------------------------------------------------

    best_order = (

        int(
            valid_results.loc[
                0,
                "p"
            ]
        ),

        int(
            valid_results.loc[
                0,
                "d"
            ]
        ),

        int(
            valid_results.loc[
                0,
                "q"
            ]
        )

    )

    return (
        best_order,
        valid_results
    )


# ============================================================
# FINAL FUTURE FORECAST
# ============================================================

def generate_future_forecast(
    history,
    horizon,
    best_order
):

    close_values = prepare_close_values(
        history
    )

    # --------------------------------------------------------
    # FIT BEST MODEL ON ALL AVAILABLE DATA
    # --------------------------------------------------------

    fitted_model = fit_arima_model(
        close_values,
        best_order
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_result = fitted_model.get_forecast(
        steps=horizon
    )

    forecast = np.asarray(
        forecast_result.predicted_mean,
        dtype=float
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    (
        lower_bound,
        upper_bound
    ) = get_confidence_interval(
        forecast_result
    )

    # --------------------------------------------------------
    # LAST DATE
    # --------------------------------------------------------

    last_date = pd.to_datetime(
        history["Date"].max()
    )

    # --------------------------------------------------------
    # FUTURE BUSINESS DAYS
    # --------------------------------------------------------

    future_dates = pd.bdate_range(

        start=(
            last_date
            + pd.Timedelta(days=1)
        ),

        periods=horizon

    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = pd.DataFrame({

        "Date":
            future_dates,

        "Predicted_Close":
            forecast,

        "Lower_95":
            lower_bound,

        "Upper_95":
            upper_bound

    })

    return result


# ============================================================
# HISTORICAL BACKTEST USING BEST MODEL
# ============================================================

def final_backtest(
    history,
    test_days,
    best_order
):

    # --------------------------------------------------------
    # PREPARE VALUES
    # --------------------------------------------------------

    values = prepare_close_values(
        history
    )

    if len(values) <= test_days + 30:

        raise ValueError(
            f"Not enough data for {test_days}-day backtest."
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_values = values[
        :-test_days
    ]

    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    actual_values = values[
        -test_days:
    ]

    # --------------------------------------------------------
    # FIT BEST MODEL
    # --------------------------------------------------------

    model = fit_arima_model(
        train_values,
        best_order
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_result = model.get_forecast(
        steps=test_days
    )

    predictions = np.asarray(
        forecast_result.predicted_mean,
        dtype=float
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    (
        lower_bound,
        upper_bound
    ) = get_confidence_interval(
        forecast_result
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual_values,
        predictions
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    clean_history = history.copy()

    clean_history["Date"] = pd.to_datetime(
        clean_history["Date"],
        errors="coerce"
    )

    clean_history["Close"] = pd.to_numeric(
        clean_history["Close"],
        errors="coerce"
    )

    clean_history = clean_history.dropna(
        subset=[
            "Date",
            "Close"
        ]
    )

    clean_history = clean_history[
        clean_history["Close"] > 0
    ]

    clean_history = clean_history.sort_values(
        by="Date"
    )

    clean_history = clean_history.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    actual_dates = clean_history[
        "Date"
    ].iloc[
        -test_days:
    ].to_numpy()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = pd.DataFrame({

        "Date":
            actual_dates,

        "Actual_Close":
            actual_values,

        "Predicted_Close":
            predictions,

        "Lower_95":
            lower_bound,

        "Upper_95":
            upper_bound

    })

    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    result["Error"] = (
        result["Predicted_Close"]
        - result["Actual_Close"]
    )

    result["Absolute_Error"] = (
        result["Error"].abs()
    )

    result["Percentage_Error"] = np.where(

        result["Actual_Close"] != 0,

        (
            result["Absolute_Error"]
            / result["Actual_Close"]
            * 100
        ),

        np.nan

    )

    return (
        result,
        metrics
    )


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
# MODEL VALIDATION DAYS
# ============================================================

validation_days = st.sidebar.selectbox(

    "Model Validation Period",

    [20, 30, 60],

    index=1

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

run_model_selection = st.sidebar.button(

    "🤖 Find Best ARIMA Model",

    use_container_width=True

)

run_backtest = st.sidebar.button(

    "🔍 Run Backtest",

    use_container_width=True

)

run_forecast = st.sidebar.button(

    "🚀 Run Forecast",

    use_container_width=True

)


# ============================================================
# FILE CHECK
# ============================================================

if (

    uploaded_file is None

    and not os.path.exists(
        DATA_FILE
    )

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


with col1:

    st.metric(
        "Rows",
        len(history)
    )


with col2:

    st.metric(

        "Start Date",

        history["Date"]
        .min()
        .strftime("%d-%m-%Y")

    )


with col3:

    st.metric(

        "End Date",

        history["Date"]
        .max()
        .strftime("%d-%m-%Y")

    )


with col4:

    latest_price = float(
        history["Close"].iloc[-1]
    )

    st.metric(

        "Latest Close",

        f"₹{latest_price:,.2f}"

    )


# ============================================================
# HISTORICAL DATA PREVIEW
# ============================================================

with st.expander(
    "📋 View Historical Data"
):

    st.dataframe(

        history,

        use_container_width=True

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
    min(
        250,
        len(history)
    )
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
# MODEL SELECTION
# ============================================================

if run_model_selection:

    st.subheader(
        "🤖 Automatic ARIMA Model Selection"
    )

    with st.spinner(
        "Testing multiple ARIMA models..."
    ):

        try:

            values = prepare_close_values(
                history
            )

            (
                best_order,
                comparison
            ) = select_best_arima(

                values,

                validation_days

            )

            # ------------------------------------------------
            # SAVE IN SESSION
            # ------------------------------------------------

            st.session_state[
                "best_order"
            ] = best_order

            st.session_state[
                "model_comparison"
            ] = comparison

            # ------------------------------------------------
            # BEST MODEL
            # ------------------------------------------------

            best_row = comparison.iloc[0]

            st.success(

                f"✅ Best ARIMA model: "
                f"ARIMA{best_order}"

            )

            # ------------------------------------------------
            # BEST MODEL METRICS
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(

                    "Validation MAE",

                    f"₹{best_row['MAE']:,.2f}"

                )

            with c2:

                st.metric(

                    "Validation RMSE",

                    f"₹{best_row['RMSE']:,.2f}"

                )

            with c3:

                st.metric(

                    "Validation MAPE",

                    f"{best_row['MAPE']:.2f}%"

                )

            # ------------------------------------------------
            # COMPARISON TABLE
            # ------------------------------------------------

            st.subheader(
                "📊 ARIMA Model Comparison"
            )

            display_comparison = comparison.copy()

            display_comparison[
                "ARIMA_Order"
            ] = display_comparison[
                "ARIMA_Order"
            ].astype(str)

            st.dataframe(

                display_comparison.style.format({

                    "MAE":
                        "₹{:.2f}",

                    "RMSE":
                        "₹{:.2f}",

                    "MAPE":
                        "{:.2f}%"

                }),

                use_container_width=True

            )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            best_mape = float(
                best_row["MAPE"]
            )

            if best_mape <= 10:

                st.success(

                    "✅ The selected ARIMA model "
                    "has relatively low validation error."

                )

            elif best_mape <= 20:

                st.warning(

                    "⚠️ The selected ARIMA model "
                    "has moderate validation error. "
                    "Stock forecasts remain uncertain."

                )

            else:

                st.error(

                    "⚠️ The selected ARIMA model "
                    "has high validation error. "
                    "Use the forecast with caution."

                )

        except Exception as e:

            st.error(
                f"Model selection error: {e}"
            )


# ============================================================
# CHECK BEST MODEL
# ============================================================

if (
    "best_order"
    not in st.session_state
):

    st.info(

        "ℹ️ Please click "
        "'🤖 Find Best ARIMA Model' "
        "before running the backtest or forecast."

    )


# ============================================================
# BACKTEST
# ============================================================

if run_backtest:

    if (
        "best_order"
        not in st.session_state
    ):

        st.warning(

            "Please run automatic ARIMA model "
            "selection first."

        )

    else:

        best_order = st.session_state[
            "best_order"
        ]

        st.subheader(

            f"🔍 {backtest_days}-Trading-Day "
            f"Backtest Using ARIMA{best_order}"

        )

        with st.spinner(
            "Running historical backtest..."
        ):

            try:

                (
                    backtest_result,
                    metrics
                ) = final_backtest(

                    history,

                    backtest_days,

                    best_order

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

                if metrics["MAPE"] <= 10:

                    st.success(

                        "✅ The selected ARIMA model "
                        "shows relatively low historical error."

                    )

                elif metrics["MAPE"] <= 20:

                    st.warning(

                        "⚠️ The selected ARIMA model "
                        "has moderate historical error."

                    )

                else:

                    st.error(

                        "⚠️ The selected ARIMA model "
                        "has high historical error."

                    )

                # ------------------------------------------------
                # BACKTEST CHART
                # ------------------------------------------------

                st.subheader(
                    "📉 Actual vs Predicted Price"
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

                    label=f"ARIMA{best_order} Predicted",

                    linestyle="--",

                    linewidth=2

                )

                ax2.fill_between(

                    backtest_result["Date"],

                    backtest_result[
                        "Lower_95"
                    ].to_numpy(),

                    backtest_result[
                        "Upper_95"
                    ].to_numpy(),

                    alpha=0.15,

                    label="95% Confidence Interval"

                )

                ax2.set_title(

                    f"ARIMA{best_order} "
                    f"Backtest - Last "
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

                        "Lower_95":
                            "₹{:.2f}",

                        "Upper_95":
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

                csv_data = (
                    backtest_result
                    .to_csv(
                        index=False
                    )
                )

                st.download_button(

                    "⬇️ Download Backtest CSV",

                    csv_data,

                    file_name=(

                        f"reliance_arima"
                        f"{best_order}_backtest_"
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

    if (
        "best_order"
        not in st.session_state
    ):

        st.warning(

            "Please run automatic ARIMA model "
            "selection first."

        )

    else:

        best_order = st.session_state[
            "best_order"
        ]

        st.subheader(

            f"🔮 Next {horizon} Business-Day "
            f"ARIMA{best_order} Forecast"

        )

        with st.spinner(
            "Generating future forecast..."
        ):

            try:

                forecast_result = generate_future_forecast(

                    history,

                    horizon,

                    best_order

                )

                # ------------------------------------------------
                # CURRENT PRICE
                # ------------------------------------------------

                latest_price = float(

                    history[
                        "Close"
                    ].iloc[-1]

                )

                # ------------------------------------------------
                # NEXT DAY
                # ------------------------------------------------

                next_day_price = float(

                    forecast_result[
                        "Predicted_Close"
                    ].iloc[0]

                )

                # ------------------------------------------------
                # FINAL DAY
                # ------------------------------------------------

                final_price = float(

                    forecast_result[
                        "Predicted_Close"
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
                # METRICS
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

                # ------------------------------------------------
                # MODEL INFORMATION
                # ------------------------------------------------

                st.success(

                    f"🤖 Forecast generated using "
                    f"automatically selected "
                    f"ARIMA{best_order} model."

                )

                # =================================================
                # FORECAST CHART
                # =================================================

                st.subheader(
                    "📈 Future ARIMA Forecast"
                )

                fig3, ax3 = plt.subplots(
                    figsize=(15, 7)
                )

                chart_history = history.tail(
                    min(
                        120,
                        len(history)
                    )
                )

                # ------------------------------------------------
                # HISTORY
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
                        "Predicted_Close"
                    ],

                    label=f"ARIMA{best_order} Forecast",

                    linewidth=2,

                    linestyle="--"

                )

                # ------------------------------------------------
                # CONFIDENCE INTERVAL
                # ------------------------------------------------

                ax3.fill_between(

                    forecast_result["Date"],

                    forecast_result[
                        "Lower_95"
                    ].to_numpy(),

                    forecast_result[
                        "Upper_95"
                    ].to_numpy(),

                    alpha=0.15,

                    label="95% Confidence Interval"

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
                    f"{horizon} Business-Day "
                    f"ARIMA{best_order} Forecast"

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

                        "Predicted_Close":
                            "₹{:.2f}",

                        "Lower_95":
                            "₹{:.2f}",

                        "Upper_95":
                            "₹{:.2f}"

                    }),

                    use_container_width=True

                )

                # =================================================
                # DOWNLOAD
                # =================================================

                forecast_csv = (

                    forecast_result
                    .to_csv(
                        index=False
                    )

                )

                st.download_button(

                    "⬇️ Download Forecast CSV",

                    forecast_csv,

                    file_name=(

                        f"reliance_arima"
                        f"{best_order}_forecast_"
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

                            f"📈 ARIMA{best_order} "
                            f"forecasts an approximately "
                            f"{expected_change:.2f}% increase "
                            f"over the next {horizon} "
                            f"business days."

                        )

                    elif expected_change < -5:

                        st.warning(

                            f"📉 ARIMA{best_order} "
                            f"forecasts an approximately "
                            f"{abs(expected_change):.2f}% decrease "
                            f"over the next {horizon} "
                            f"business days."

                        )

                    else:

                        st.info(

                            f"➡️ ARIMA{best_order} "
                            f"forecasts relatively stable "
                            f"movement over the next "
                            f"{horizon} business days."

                        )

                # =================================================
                # FINAL CONFIDENCE INTERVAL
                # =================================================

                lower_final = float(

                    forecast_result[
                        "Lower_95"
                    ].iloc[-1]

                )

                upper_final = float(

                    forecast_result[
                        "Upper_95"
                    ].iloc[-1]

                )

                st.info(

                    f"📊 Day {horizon} estimate: "
                    f"₹{final_price:,.2f}. "
                    f"Approximate 95% confidence interval: "
                    f"₹{lower_final:,.2f} to "
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
    "Automatic ARIMA Model Selection | "
    "MAE | RMSE | MAPE | "
    "Educational Project"

)
