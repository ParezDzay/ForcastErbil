from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image

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
    df = pd.read_csv(LEVELS_URL)

    # 🔧 Replace with the correct date column name if available
    for col in df.columns:
        if col.lower() == "date":
            df["Date"] = pd.to_datetime(df[col])
            break
    else:
        df["Date"] = pd.NaT  # If no date column, assign NaT

    return df

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

    levels = load_levels()
    coords = load_coords()
    well_cols = [c for c in levels.columns if c.upper().startswith("W")]
    if not well_cols:
        st.error("No W1…Wn columns found.")
        st.stop()

    if "Date" not in levels.columns or pd.isna(levels["Date"]).all():
        st.error("Date column not found or invalid. Please update the CSV.")
        st.stop()

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
    if year_df.empty:
        st.warning("No matching wells.")
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
                st.error("No frames generated.")
                return
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

if __name__ == "__main__":
    main()
