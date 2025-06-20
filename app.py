"""
2-D groundwater surface (thin-plate-spline RBF) – Streamlit
with GIF animation over time.

Data (raw from GitHub):
  • Monthly_Sea_Level_Data.csv  – water levels by month (Date, W1…Wn)
  • wells.csv                   – well IDs + coordinates

Map bounds are fixed to:
  lat  35.80 – 36.40
  lon  43.60 – 44.30
"""

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
LEVELS_URL = (
    "https://raw.githubusercontent.com/"
    "hawkarabdulhaq/waterdemo/main/Monthly_Sea_Level_Data.csv"
)
COORDS_URL = (
    "https://raw.githubusercontent.com/"
    "hawkarabdulhaq/waterdemo/main/wells.csv"
)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_levels() -> pd.DataFrame:
    """Monthly water levels."""
    return pd.read_csv(LEVELS_URL, parse_dates=["Date"])


@st.cache_data
def load_coords() -> pd.DataFrame:
    """
    Well coordinates – normalise column names and drop duplicates.

    ID synonyms  : well | no | id | well_name
    Lat synonyms : lat  | latitude | y | northing
    Lon synonyms : lon  | lng | longitude | x | easting
    """
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
# Interpolation helper
# ---------------------------------------------------------------------------


def rbf_surface(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, res: int):
    """Return meshgrid (lon_g, lat_g, z_g) on fixed bounds using thin-plate RBF."""
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g


# ---------------------------------------------------------------------------
# Plot single frame
# ---------------------------------------------------------------------------


def draw_frame(
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
    z_arr: np.ndarray,
    date_label: str,
    grid_res: int,
    n_levels: int,
) -> Image.Image:
    """Return a PIL Image of the contour plot for one month."""
    lon_g, lat_g, z_g = rbf_surface(lon_arr, lat_arr, z_arr, grid_res)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    cf = ax.contourf(
        lon_g,
        lat_g,
        z_g,
        levels=n_levels,
        cmap="viridis",
        alpha=0.75,
    )
    ax.scatter(
        lon_arr,
        lat_arr,
        c=z_arr,
        edgecolors="black",
        s=60,
        label="Wells",
    )
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
# Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("2-D Water-Table Map (RBF) + GIF Animation")

    # ---- Load data --------------------------------------------------------
    levels = load_levels()
    coords = load_coords()
    well_cols = [c for c in levels.columns if c.upper().startswith("W")]

    if not well_cols:
        st.error("No W1…Wn columns found in the levels CSV.")
        st.stop()

    # ---- Sidebar controls -------------------------------------------------
    st.sidebar.header("Controls")
    date_opts = levels["Date"].dt.strftime("%Y-%m-%d")
    date_sel = st.sidebar.selectbox("Month", date_opts, index=len(date_opts) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)

    make_gif = st.sidebar.button("Generate GIF (all months)")

    # ---- Prepare selected month ------------------------------------------
    levels_row = levels.loc[date_opts == date_sel, well_cols].iloc[0]
    month_df = (
        levels_row.rename_axis("well")
        .reset_index(name="level")
        .merge(coords, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if month_df.empty:
        st.warning("No matching wells between the two CSV files.")
        st.stop()

    lon = month_df["lon"].to_numpy(float)
    lat = month_df["lat"].to_numpy(float)
    z   = month_df["level"].to_numpy(float)

    # ---- Draw current month ----------------------------------------------
    lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)

    fig, ax = plt.subplots()
    cf = ax.contourf(
        lon_g,
        lat_g,
        z_g,
        levels=n_levels,
        cmap="viridis",
        alpha=0.75,
    )
    ax.scatter(
        lon,
        lat,
        c=z,
        edgecolors="black",
        s=80,
        label="Wells",
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {date_sel}")
    fig.colorbar(cf, ax=ax, label="Level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # ---- Table -----------------------------------------------------------
    with st.expander("Raw data for this month"):
        st.dataframe(
            month_df[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )

    # ---- GIF generation ---------------------------------------------------
    if make_gif:
        with st.spinner("Generating GIF… this may take a minute"):
            frames: list[Image.Image] = []
            for idx, (ts, row) in enumerate(levels[["Date"] + well_cols].iterrows()):
                date_str = row["Date"].strftime("%Y-%m-%d")
                frame_df = (
                    row[well_cols]
                    .rename_axis("well")
                    .reset_index(name="level")
                    .merge(coords, on="well", how="inner")
                    .dropna(subset=["lat", "lon", "level"])
                )
                if frame_df.empty:
                    continue
                frame_img = draw_frame(
                    frame_df["lon"].to_numpy(float),
                    frame_df["lat"].to_numpy(float),
                    frame_df["level"].to_numpy(float),
                    date_str,
                    grid_res,
                    n_levels,
                )
                frames.append(frame_img)

            if not frames:
                st.error("No frames could be generated (missing data).")
                return

            gif_bytes = io.BytesIO()
            frames[0].save(
                gif_bytes,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=500,   # ms per frame
                loop=0,
            )
            gif_bytes.seek(0)

        st.subheader("Time-Series Animation")
        st.image(gif_bytes.getvalue())

        st.download_button(
            "Download GIF",
            data=gif_bytes.getvalue(),
            file_name="water_table_animation.gif",
            mime="image/gif",
        )


if __name__ == "__main__":
    main()
