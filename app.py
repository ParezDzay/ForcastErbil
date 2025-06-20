from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
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
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_levels() -> pd.DataFrame:
    """Monthly water levels."""
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])

@st.cache_data
def load_coords() -> pd.DataFrame:
    """Well coordinates with standard column names."""
    df = pd.read_csv(COORDS_URL)

    id_syn  = {"well", "no", "id", "well_name"}
    lat_syn = {"lat", "latitude", "y", "northing"}
    lon_syn = {"lon", "lng", "longitude", "x", "easting"}

    rename = {}
    id_done = lat_done = lon_done = False
    for col in df.columns:
        c = col.lower()
        if c in id_syn and not id_done:
            rename[col] = "well"
            id_done = True
        elif c in lat_syn and not lat_done:
            rename[col] = "lat"
            lat_done = True
        elif c in lon_syn and not lon_done:
            rename[col] = "lon"
            lon_done = True

    df = df.rename(columns=rename).loc[:, ~df.columns.duplicated(keep="first")]
    missing = {"well", "lat", "lon"} - set(df.columns)
    if missing:
        st.error(f"`wells.csv` is missing column(s): {', '.join(sorted(missing))}")
        st.stop()

    return df[["well", "lat", "lon"]]

# ---------------------------------------------------------------------------
# RBF Interpolation
# ---------------------------------------------------------------------------

def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g

# ---------------------------------------------------------------------------
# Draw single frame
# ---------------------------------------------------------------------------

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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    page = st.sidebar.selectbox("Select Page", ["📊 Water-Table Map", "🔮 Forecast"])

    if page == "📊 Water-Table Map":
        levels = load_levels()
        coords = load_coords()
        well_cols = [c for c in levels.columns if c.upper().startswith("W")]

        if not well_cols:
            st.error("No W1…Wn columns found in the levels CSV.")
            st.stop()

        # Resample to annual averages
        levels["Year"] = levels["Date"].dt.year
        annual_levels = levels.groupby("Year")[well_cols].mean().reset_index()

        st.sidebar.header("Controls")
        year_opts = annual_levels["Year"].astype(str)
        year_sel = st.sidebar.selectbox("Year", year_opts, index=len(year_opts) - 1)
        grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 300, 50)
        n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
        make_gif = st.sidebar.button("Generate GIF (all years)")

        # Display selected year
        year_row = annual_levels[annual_levels["Year"].astype(str) == year_sel][well_cols].iloc[0]
        year_df = (
            year_row.rename_axis("well")
            .reset_index(name="level")
            .merge(coords, on="well", how="inner")
            .dropna(subset=["lat", "lon", "level"])
        )

        if year_df.empty:
            st.warning("No matching wells between the two CSV files.")
            st.stop()

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
            with st.spinner("Generating GIF… this may take a minute"):
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
                    if frame_df.empty:
                        continue
                    img = draw_frame(
                        frame_df["lon"].to_numpy(float),
                        frame_df["lat"].to_numpy(float),
                        frame_df["level"].to_numpy(float),
                        year,
                        grid_res,
                        n_levels,
                    )
                    frames.append(img)

                if not frames:
                    st.error("No frames could be generated.")
                    return

                gif_bytes = io.BytesIO()
                frames[0].save(
                    gif_bytes,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,
                    loop=0,
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

    elif page == "🔮 Forecast":
        st.header("Random Forest Forecasting")
        levels = load_levels()
        well_cols = [c for c in levels.columns if c.upper().startswith("W")]
        levels["Year"] = levels["Date"].dt.year
        levels["Month"] = levels["Date"].dt.month

        well = st.selectbox("Select Well", well_cols)
        forecast_years = st.slider("Forecast up to year", 2025, 2030, 2029)

        X = levels[["Year", "Month"]]
        y = levels[well]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        st.markdown(f"**R²**: {r2:.3f}, **RMSE**: {rmse:.3f}")

        future_df = pd.DataFrame({
            "Year": np.repeat(range(2025, forecast_years + 1), 12),
            "Month": list(range(1, 13)) * (forecast_years - 2024),
        })

        future_pred = model.predict(future_df)
        future_df["Forecast"] = future_pred
        st.line_chart(future_df.groupby("Year")["Forecast"].mean())

if __name__ == "__main__":
    main()
