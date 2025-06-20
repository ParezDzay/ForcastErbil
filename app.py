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
# RBF Interpolation
# ---------------------------------------------------------------------------

def rbf_surface(lon, lat, z, res):
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g

def draw_frame(lon_arr, lat_arr, z_arr, date_label, grid_res, n_levels):
    lon_g, lat_g, z_g = rbf_surface(lon_arr, lat_arr, z_arr, grid_res)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.75)
    ax.scatter(lon_arr, lat_arr, c=z_arr, edgecolors="black", s=60, label="Wells")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water Table — {date_label}")
    fig.colorbar(cf, ax=ax, label="Level")
    ax.legend(loc="upper right")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    page = st.sidebar.selectbox("Select Page", ["📊 Water-Table Map", "🔮 Forecast (Average Water Table)"])

    if page == "📊 Water-Table Map":
        levels = load_levels(); coords = load_coords()
        well_cols = [c for c in levels.columns if c.upper().startswith("W")]
        if not well_cols: st.error("No W1…Wn columns found."); st.stop()

        levels["Year"] = levels["Date"].dt.year
        annual_levels = levels.groupby("Year")[well_cols].mean().reset_index()

        st.sidebar.header("Controls")
        year_opts = annual_levels["Year"].astype(str)
        year_sel = st.sidebar.selectbox("Year", year_opts, index=len(year_opts) - 1)
        grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 300, 50)
        n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
        make_gif = st.sidebar.button("Generate GIF (all years)")

        year_row = annual_levels[annual_levels["Year"].astype(str) == year_sel][well_cols].iloc[0]
        year_df = (
            year_row.rename_axis("well")
            .reset_index(name="level")
            .merge(coords, on="well", how="inner")
            .dropna(subset=["lat", "lon", "level"])
        )
        if year_df.empty: st.warning("No matching wells."); st.stop()

        lon = year_df["lon"].to_numpy(float)
        lat = year_df["lat"].to_numpy(float)
        z = year_df["level"].to_numpy(float)
        lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)

        fig, ax = plt.subplots()
        cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.75)
        ax.scatter(lon, lat, c=z, edgecolors="black", s=80, label="Wells")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Water-Table Surface — {year_sel}")
        fig.colorbar(cf, ax=ax, label="Level")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        with st.expander("Raw data for this year"):
            st.dataframe(
                year_df[["well", "lat", "lon", "level"]]
                .set_index("well")
                .sort_index(),
                use_container_width=True,
            )

        if make_gif:
            with st.spinner("Generating GIF…"):
                frames = []
                for _, row in annual_levels.iterrows():
                    year = str(int(row["Year"]))
                    frame_df = (
                        row[well_cols]
                        .rename_axis("well")
                        .reset_index(name="level")
                        .merge(coords, on="well", how="inner")
                        .dropna(subset=["lat", "lon", "level"])
                    )
                    if frame_df.empty: continue
                    img = draw_frame(
                        frame_df["lon"].to_numpy(float),
                        frame_df["lat"].to_numpy(float),
                        frame_df["level"].to_numpy(float),
                        year,
                        grid_res,
                        n_levels,
                    )
                    frames.append(img)

                if not frames: st.error("No frames generated."); return
                gif_bytes = io.BytesIO()
                frames[0].save(
                    gif_bytes, format="GIF",
                    save_all=True, append_images=frames[1:], duration=500, loop=0,
                )
                gif_bytes.seek(0)
            st.subheader("Time-Series Animation")
            st.image(gif_bytes.getvalue())
            st.download_button(
                "Download GIF",
                data=gif_bytes.getvalue(),
                file_name="water_table_animation_annual.gif",
                mime="image/gif",
            )

    elif page == "🔮 Forecast (Average Water Table)":
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

        # Trend direction based on RF predictions
        X_year = annual_pred["Year"].values.reshape(-1, 1)
        y_level = annual_pred["Forecast"].values
        reg = LinearRegression().fit(X_year, y_level)
        slope = reg.coef_[0]

        st.subheader("📉 Estimated Rate of Change (2025–{})".format(forecast_years))
        st.write(f"**Rate of change in predicted water levels:** {slope:.2f} meters/year")

        # Groundwater interpretation: increase in level → water table decline
        if slope > 0:
            st.warning("⚠️ Predicted groundwater is **declining** (water level rising).")
        else:
            st.success("✅ Predicted groundwater is **improving** (water level decreasing).")

        # Visualization
        fig, ax = plt.subplots()
        ax.plot(X_year.flatten(), y_level, 'o-', label="Predicted Mean Level")
        ax.plot(X_year.flatten(), reg.predict(X_year), 'r--', label=f"Trend line ({slope:.2f} m/year)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted Avg. Water Level")
        ax.set_title("Forecast Trend (2025–{})".format(forecast_years))
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
