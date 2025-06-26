from __future__ import annotations

import io, unicodedata, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image

# ──────────────────────────────────────────────────────────────
#  CONFIG – adjust if you move the files or bounding box
# ──────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30
LEVELS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/Monthly_Sea_Level_Data.csv"
COORDS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/wells.csv"

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────
def normalise_well(name: str) -> str:
    """Convert variations like ' w01 ' or 'Well-1' to 'W1'."""
    s = unicodedata.normalize("NFKD", str(name).strip().upper())
    digits = re.findall(r"\d+", s)
    return f"W{digits[0].lstrip('0')}" if digits else s

def rbf_surface(lon, lat, z, res):
    rbf = Rbf(lon, lat, z, function="thin_plate")
    lon_g, lat_g = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, res),
        np.linspace(LAT_MIN, LAT_MAX, res),
    )
    return lon_g, lat_g, rbf(lon_g, lat_g)

# ──────────────────────────────────────────────────────────────
#  DATA LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_levels() -> pd.DataFrame:
    df = pd.read_csv(LEVELS_URL)
    # Normalise any header that looks like W#  (just in case of W01 etc.)
    df.columns = [
        normalise_well(c) if c.strip().upper().startswith("W") else c
        for c in df.columns
    ]
    return df

@st.cache_data
def load_coords() -> pd.DataFrame:
    df = pd.read_csv(COORDS_URL)
    # Strip spaces/newlines in header names
    df.columns = [c.strip().lower() for c in df.columns]

    # Keep only needed cols & tidy
    df = df.rename(columns={"well": "well", "lat": "lat", "lon": "lon"})
    df = df[["well", "lat", "lon"]].dropna()
    df["well"] = df["well"].apply(normalise_well)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df.drop_duplicates(subset="well")

# ──────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    levels = load_levels()
    coords  = load_coords()

    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No W1…Wn columns found in the level file.")
        return

    # Sidebar
    years = levels["Year"].astype(str)
    year_sel = st.sidebar.selectbox("Year", years, index=len(years) - 1)
    grid_res = st.sidebar.slider("Grid resolution (px)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif = st.sidebar.button("Generate GIF (all years)")

    # ------------------- Plot chosen year ----------------------
    row = levels.loc[levels["Year"].astype(str) == year_sel, well_cols].iloc[0]
    df_year = (
        row.rename_axis("well").reset_index(name="level")
    )
    df_year["well"] = df_year["well"].apply(normalise_well)

    merged = (
        df_year.merge(coords, on="well", how="inner")
               .dropna(subset=["lat", "lon", "level"])
    )

    if merged.empty:
        st.error("No wells matched between level data and wells.csv.")
        return

    lon = merged["lon"].to_numpy(float)
    lat = merged["lat"].to_numpy(float)
    z   = merged["level"].to_numpy(float)

    # If <3 wells, skip interpolation (Rbf needs ≥3 for a surface)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    if len(merged) >= 3:
        lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)
        ax.contourf(lon_g, lat_g, z_g, levels=n_levels,
                    cmap="viridis", alpha=0.75)

    ax.scatter(lon, lat, c=z, edgecolors="black", s=120, label="Wells")
    ax.set(xlim=(LON_MIN, LON_MAX), ylim=(LAT_MIN, LAT_MAX),
           aspect="equal",
           xlabel="Longitude", ylabel="Latitude",
           title=f"Water Table — {year_sel}")
    fig.colorbar(ax.collections[0] if len(ax.collections) else ax.scatter(lon, lat),
                 ax=ax, label="Level")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    with st.expander("Raw data"):
        st.dataframe(merged.set_index("well"), use_container_width=True)

    # -------------------- GIF (all years) ----------------------
    if make_gif:
        with st.spinner("Building GIF…"):
            frames: list[Image.Image] = []
            for _, row in levels.iterrows():
                yr = int(row["Year"])
                df = row[well_cols].rename_axis("well").reset_index(name="level")
                df["well"] = df["well"].apply(normalise_well)
                df = df.merge(coords, on="well", how="inner").dropna()
                if len(df) < 3:
                    continue  # skip years with <3 matching wells
                img = draw_frame(df["lon"], df["lat"], df["level"],
                                 str(yr), grid_res, n_levels)
                frames.append(img)

            if not frames:
                st.error("No frames generated – need ≥3 matching wells each year.")
                return

            buf = io.BytesIO()
            frames[0].save(buf, format="GIF",
                           save_all=True, append_images=frames[1:],
                           duration=500, loop=0)
            buf.seek(0)

        st.subheader("Time-series animation")
        st.image(buf.getvalue())
        st.download_button("Download GIF", data=buf.getvalue(),
                           file_name="water_table_animation.gif",
                           mime="image/gif")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
