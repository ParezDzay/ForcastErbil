from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image

# ---------------------------------------------------------------------------
# Toggle extra prints (set to True if you need to inspect merge keys)
# ---------------------------------------------------------------------------
SHOW_DEBUG = False   # <- change to True for st.write() debug outputs

# ---------------------------------------------------------------------------
# Bounding box of the Erbil study area
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30

# ---------------------------------------------------------------------------
# Raw-file URLs (adjust if you move the files)
# ---------------------------------------------------------------------------
LEVELS_URL = (
    "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/Monthly_Sea_Level_Data.csv"
)
COORDS_URL = (
    "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/wells.csv"
)

# ---------------------------------------------------------------------------
# Load annual groundwater levels (Year,W1-W20)
# ---------------------------------------------------------------------------
@st.cache_data
def load_levels() -> pd.DataFrame:
    df = pd.read_csv(LEVELS_URL)
    # Force well columns to a uniform upper-case, no-space style
    df.columns = [c.strip().upper() for c in df.columns]
    return df

# ---------------------------------------------------------------------------
# Load well coordinates, enforce unique well/lat/lon, unify case
# ---------------------------------------------------------------------------
@st.cache_data
def load_coords() -> pd.DataFrame:
    df = pd.read_csv(COORDS_URL)

    # Lower-case col names for matching
    df.columns = [c.lower() for c in df.columns]

    # Rename flexibly
    rename: dict[str, str] = {}
    for col in df.columns:
        if any(k in col for k in ("well", "id", "no")):
            rename[col] = "well"
        elif any(k in col for k in ("lat", "north", "y")):
            rename[col] = "lat"
        elif any(k in col for k in ("lon", "long", "east", "x")):
            rename[col] = "lon"
    df = df.rename(columns=rename)

    # Remove duplicate column labels
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    # Verify required columns
    req = {"well", "lat", "lon"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(
            f"wells.csv is missing required column(s): {', '.join(missing)}"
        )

    # Keep only wanted columns, drop NaNs & dups
    df = (
        df[["well", "lat", "lon"]]
        .dropna()
        .drop_duplicates(subset="well", keep="first")
    )

    # Standardise values
    df["well"] = df["well"].astype(str).str.strip().str.upper()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Clip to bounding box (optional safety)
    df = df[
        (df["lat"].between(LAT_MIN, LAT_MAX))
        & (df["lon"].between(LON_MIN, LON_MAX))
    ]

    return df

# ---------------------------------------------------------------------------
# Thin-plate-spline interpolation
# ---------------------------------------------------------------------------
def rbf_surface(lon, lat, z, res):
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    z_g = rbf(lon_g, lat_g)
    return lon_g, lat_g, z_g

# ---------------------------------------------------------------------------
# Create a contour frame (single year or for GIF)
# ---------------------------------------------------------------------------
def draw_frame(lon_arr, lat_arr, z_arr, label, grid_res, n_levels):
    lon_g, lat_g, z_g = rbf_surface(lon_arr, lat_arr, z_arr, grid_res)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.75)
    ax.scatter(lon_arr, lat_arr, c=z_arr, edgecolors="black", s=90, label="Wells")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water Table — {label}")
    fig.colorbar(cf, ax=ax, label="Level")
    ax.legend(loc="upper right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# Streamlit main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    # Load data
    levels = load_levels()
    coords = load_coords()

    # Identify well columns (after upper-case)
    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No columns named W1…Wn found in level file.")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Controls")
    year_opts = levels["YEAR"].astype(str)
    year_sel = st.sidebar.selectbox("Year", year_opts, index=len(year_opts) - 1)
    grid_res = st.sidebar.slider("Grid resolution (pixels)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif = st.sidebar.button("Generate GIF (all years)")

    # Row for selected year
    row = levels.loc[levels["YEAR"].astype(str) == year_sel, well_cols].iloc[0]

    # Build dataframe (well, level) → merge with coords
    year_df = (
        row.rename_axis("well")
        .reset_index(name="level")
    )
    # Standardise well names before merge
    year_df["well"] = year_df["well"].astype(str).str.strip().str.upper()

    # (Optional debug)
    if SHOW_DEBUG:
        st.write("Wells in levels for year", year_sel, ":", year_df["well"].tolist())
        st.write("Coords wells:", coords["well"].tolist())

    year_df = (
        year_df.merge(coords, on="well", how="inner")
        .dropna(subset=["lat", "lon", "level"])
    )

    if year_df.empty:
        st.error("No wells matched between level data and coordinates.")
        st.stop()

    # Plot
    lon = year_df["lon"].to_numpy(float)
    lat = year_df["lat"].to_numpy(float)
    z = year_df["level"].to_numpy(float)

    lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)

    fig, ax = plt.subplots()
    cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.75)
    ax.scatter(lon, lat, c=z, edgecolors="black", s=90, label="Wells")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Water-Table Surface — {year_sel}")
    fig.colorbar(cf, ax=ax, label="Level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # Raw data view
    with st.expander("Raw data for this year"):
        st.dataframe(
            year_df[["well", "lat", "lon", "level"]]
            .set_index("well")
            .sort_index(),
            use_container_width=True,
        )

    # GIF option
    if make_gif:
        with st.spinner("Generating GIF…"):
            frames: list[Image.Image] = []
            for _, row in levels.iterrows():
                yr = str(int(row["YEAR"]))
                frame_df = (
                    row[well_cols]
                    .rename_axis("well")
                    .reset_index(name="level")
                )
                frame_df["well"] = frame_df["well"].astype(str).str.strip().str.upper()
                frame_df = (
                    frame_df.merge(coords, on="well", how="inner")
                    .dropna(subset=["lat", "lon", "level"])
                )
                if frame_df.empty:
                    continue
                img = draw_frame(
                    frame_df["lon"].to_numpy(float),
                    frame_df["lat"].to_numpy(float),
                    frame_df["level"].to_numpy(float),
                    yr,
                    grid_res,
                    n_levels,
                )
                frames.append(img)

            if not frames:
                st.error("No frames generated — check data coverage.")
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

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
