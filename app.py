# app.py — ANN · ARIMA · ARIMAX Forecasting for All Wells
import streamlit as st, pandas as pd, numpy as np, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed

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

def forecast_arima(series):
    model = ARIMA(series, order=(1,1,1)).fit()
    pred = model.forecast(steps=H)
    return pd.Series(pred.values, index=pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=H, freq='MS'))

def forecast_arimax(series, exog):
    model = SARIMAX(series, exog=exog, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False).fit()
    future_exog = exog[-H:].reset_index(drop=True)
    pred = model.forecast(steps=H, exog=future_exog)
    return pd.Series(pred.values, index=pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=H, freq='MS'))

def forecast_ann(series):
    sc = MinMaxScaler()
    data_scaled = sc.fit_transform(series.values.reshape(-1,1)).flatten()
    X, y = [], []
    lags = 12
    for i in range(lags, len(data_scaled)):
        X.append(data_scaled[i-lags:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([Dense(64, activation='relu', input_shape=(lags,)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=8, verbose=0, callbacks=[EarlyStopping(patience=3)])
    hist = list(data_scaled[-lags:])
    fc_scaled = []
    for _ in range(H):
        x_input = np.array(hist[-lags:]).reshape(1,-1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        fc_scaled.append(yhat)
        hist.append(yhat)
    return pd.Series(sc.inverse_transform(np.array(fc_scaled).reshape(-1,1)).flatten(), index=pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=H, freq='MS'))

raw = load_data()
if raw is None:
    st.error("CSV not found or invalid. Upload below.")
    st.stop()

all_wells = [c for c in raw.columns if c.startswith("W")]
exog_vars = [c for c in raw.columns if c not in ["Date"] + all_wells]

model_choice = st.sidebar.radio("Model", ["ARIMA", "ARIMAX", "ANN"])

results = []

for well in all_wells:
    s = clean_series(raw, well)
    s.index = raw["Date"]
    if model_choice == "ARIMA":
        f = forecast_arima(s)
    elif model_choice == "ARIMAX":
        exog = raw[exog_vars].fillna(method='ffill').fillna(method='bfill')
        exog.index = raw["Date"]
        exog = exog.loc[s.index]
        f = forecast_arimax(s, exog)
    elif model_choice == "ANN" and _TF:
        f = forecast_ann(s)
    else:
        continue
    yearly = f.resample("A").mean()
    row = {"Well": well}
    for y in FORECAST_YEARS:
        sel = yearly[yearly.index.year == y]
        row[str(y)] = round(sel.iloc[0],2) if not sel.empty else np.nan
    results.append(row)

st.subheader(f"{model_choice} Forecast — 2025 to 2029")
st.dataframe(pd.DataFrame(results), use_container_width=True)
