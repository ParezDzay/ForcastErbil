from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# Fixed bounding box
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30

# ---------------------------------------------------------------------------
# GitHub raw URLs
# ---------------------------------------------------------------------------
LEVELS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/Monthly_Sea_Level_Data.csv"
COORDS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/wells.csv"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_levels() -> pd.DataFrame:
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])

@st.cache_data
def load_coords() -> pd.DataFrame:
    df = pd.read_csv(COORDS_URL)
    id_syn = {"well", "no", "id", "well_name"}
    lat_syn = {"lat", "latitude", "y", "northing"}
    lon_syn = {"lon", "lng", "longitude", "x", "easting"}
    rename = {}
    for col in df.columns:
        c = col.lower()
        if c in id_syn:
            rename[col] = "well"
        elif c in lat_syn:
            rename[col] = "lat"
        elif c in lon_syn:
            rename[col] = "lon"
    df = df.rename(columns=rename)
    return df[["well", "lat", "lon"]]

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    page = st.sidebar.selectbox("Select Page", ["🔮 Forecast (Average Water Table)"])

    if page == "🔮 Forecast (Average Water Table)":
        st.header("Forecasting + Rate of Decline Using Random Forest")
        levels = load_levels()
        well_cols = [c for c in levels.columns if c.upper().startswith("W")]
        levels["Year"] = levels["Date"].dt.year
        levels["Month"] = levels["Date"].dt.month
        levels["Mean_Level"] = levels[well_cols].mean(axis=1)

        X = levels[["Year", "Month"]]
        y = levels["Mean_Level"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        st.markdown(f"**Model Performance** – R²: {r2:.3f}, RMSE: {rmse:.3f} m")

        forecast_years = st.slider("Forecast up to year", 2025, 2030, 2029)
        future_df = pd.DataFrame({
            "Year": np.repeat(range(2025, forecast_years + 1), 12),
            "Month": list(range(1, 13)) * (forecast_years - 2024),
        })
        future_df["Forecast"] = model.predict(future_df)

        annual_pred = future_df.groupby("Year")["Forecast"].mean().reset_index()

        # Calculate slope using linear regression
        X_year = annual_pred["Year"].values.reshape(-1, 1)
        y_level = annual_pred["Forecast"].values
        reg = LinearRegression().fit(X_year, y_level)
        slope = reg.coef_[0]

        st.subheader("📉 Estimated Rate of Change (2025–{})".format(forecast_years))
        st.write(f"**Rate of change:** {slope:.2f} meters/year")
        if slope < 0:
            st.warning("⚠️ Declining groundwater trend")
        else:
            st.success("✅ Improving groundwater trend")

        # Visualization
        fig, ax = plt.subplots()
        ax.plot(X_year.flatten(), y_level, 'o-', label="Forecasted Mean Level")
        ax.plot(X_year.flatten(), reg.predict(X_year), 'r--', label=f"Trend line ({slope:.2f} m/year)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Groundwater Level (m)")
        ax.set_title("Forecast Trend (Random Forest)")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
