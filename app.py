# app.py — ANN · ARIMA · ARIMAX Forecasting for All Wells
import streamlit as st, pandas as pd, numpy as np, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
import plotly.express as px

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    _TF = True
except:
    _TF = False

st.set_page_config(page_title="Groundwater Forecasts (ANN / ARIMA / ARIMAX)", layout="wide")
st.title("Groundwater Forecasting — ANN · ARIMA · ARIMAX")

DATA_PATH = "GW data (missing filled).csv"
H = 60
FORECAST_YEARS = list(range(2025, 2030))

@st.cache_data(show_spinner=False)
def load_data():
    if not Path(DATA_PATH).exists(): return None
    df = pd.read_csv(DATA_PATH)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df.dropna(subset=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, col):
    s = df[col].copy()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.where(s.between(q1 - 3*iqr, q3 + 3*iqr)).interpolate(limit_direction="both")

def clip_bounds(series):
    return float(series.min()), float(series.max())

def add_lags(df, col, lags):
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df[col].shift(i)
    return df.dropna()

def train_ann(df, col, layers, lags, scaler_type, lo, hi):
    df = df.copy()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, RobustScaler
    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df[col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler() if scaler_type == "Standard" else RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(lags,)))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, callbacks=[EarlyStopping(patience=5)])
    y_pred = model.predict(X_test).flatten()
    r2_train = model.evaluate(X_train, y_train, verbose=0)
    r2_test = model.evaluate(X_test, y_test, verbose=0)
    rmse_train = np.sqrt(np.mean((model.predict(X_train).flatten() - y_train) ** 2))
    rmse_test = np.sqrt(np.mean((y_pred - y_test) ** 2))
    metrics = {
        "R2_train": r2_train,
        "RMSE_train": rmse_train,
        "R2_test": r2_test,
        "RMSE_test": rmse_test
    }
    last_known = df.iloc[-1]
    history = df[[f"lag_{i}" for i in range(1, lags + 1)]].values[-1].tolist()
    future_preds = []
    for _ in range(H):
        x_input = np.array(history[-lags:]).reshape(1, -1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        yhat = np.clip(yhat, lo, hi)
        future_preds.append(yhat)
        history.append(yhat)
    future_dates = pd.date_range(df["Date"].max() + pd.DateOffset(months=1), periods=H, freq="MS")
    future_df = pd.DataFrame({"Date": future_dates, "Depth": future_preds})
    hist_df = df[["Date"]].copy()
    hist_df["pred"] = model.predict(X).flatten()
    return metrics, hist_df, future_df

raw = load_data()
if raw is None:
    st.error("CSV not found or invalid. Upload below.")
    st.stop()

all_wells = [c for c in raw.columns if c.startswith("W")]
exog_vars = [c for c in raw.columns if c not in ["Date"] + all_wells]

model_choice = st.sidebar.radio("Model", ["ARIMA", "ARIMAX", "🔮 ANN"])

if model_choice == "🔮 ANN" and _TF:
    lags = st.sidebar.slider("Lag steps", 1, 24, 12)
    layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
    scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])

    rows = []
    for well in all_wells:
        clean = clean_series(raw, well)
        lo, hi = clip_bounds(clean)
        if len(clean) < lags * 10:
            continue
        feat = pd.DataFrame({"Date": raw["Date"], well: clean})
        feat = add_lags(feat, well, lags)
        try:
            metrics, hist, future = train_ann(feat, well, layers, lags, scaler_choice, lo, hi)
            row = {
                "Well": well,
                "R²_train": round(metrics.get("R2_train", 0), 4),
                "RMSE_train": round(metrics.get("RMSE_train", 0), 4),
                "R²_test": round(metrics.get("R2_test", 0), 4),
                "RMSE_test": round(metrics.get("RMSE_test", 0), 4)
            }
            annual = future.resample("A", on="Date").mean()
            for year in range(2025, 2030):
                val = annual[annual.index.year == year]["Depth"]
                row[str(year)] = round(val.values[0], 2) if not val.empty else None
            rows.append(row)
        except Exception as e:
            st.warning(f"{well} failed: {e}")
            continue
    result_df = pd.DataFrame(rows)
    st.subheader("📊 ANN Forecast Summary (All Wells)")
    st.dataframe(result_df, use_container_width=True)

elif model_choice == "ARIMA":
    results = []
    for well in all_wells:
        s = clean_series(raw, well)
        s.index = raw["Date"]
        model = ARIMA(s, order=(1,1,1)).fit()
        pred = model.forecast(steps=H)
        f = pd.Series(pred.values, index=pd.date_range(s.index[-1]+pd.DateOffset(months=1), periods=H, freq='MS'))
        yearly = f.resample("A").mean()
        row = {"Well": well}
        for y in FORECAST_YEARS:
            sel = yearly[yearly.index.year == y]
            row[str(y)] = round(sel.iloc[0],2) if not sel.empty else np.nan
        results.append(row)
    st.subheader("ARIMA Forecast — 2025 to 2029")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

elif model_choice == "ARIMAX":
    results = []
    for well in all_wells:
        s = clean_series(raw, well)
        s.index = raw["Date"]
        exog = raw[exog_vars].fillna(method='ffill').fillna(method='bfill')
        exog.index = raw["Date"]
        exog = exog.loc[s.index]
        model = SARIMAX(s, exog=exog, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False).fit()
        future_exog = exog[-H:].reset_index(drop=True)
        pred = model.forecast(steps=H, exog=future_exog)
        f = pd.Series(pred.values, index=pd.date_range(s.index[-1]+pd.DateOffset(months=1), periods=H, freq='MS'))
        yearly = f.resample("A").mean()
        row = {"Well": well}
        for y in FORECAST_YEARS:
            sel = yearly[yearly.index.year == y]
            row[str(y)] = round(sel.iloc[0],2) if not sel.empty else np.nan
        results.append(row)
    st.subheader("ARIMAX Forecast — 2025 to 2029")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
