from __future__ import annotations

import io, unicodedata, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30
LEVELS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/Monthly_Sea_Level_Data.csv"
COORDS_URL = "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/wells.csv"

# ── HELPERS ────────────────────────────────────────────────────────
def normalise_well(name: str) -> str:
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

def draw_frame(lon, lat, z, label, grid_res, n_levels) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    if len(lon) >= 3:
        lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)
        cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.75)
        fig.colorbar(cf, ax=ax, label="Level")
    else:
        sc = ax.scatter(lon, lat, c=z, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Level")
    ax.scatter(lon, lat, c=z, edgecolors="black", s=120, label="Wells")
    ax.set(
        xlim=(LON_MIN, LON_MAX),
        ylim=(LAT_MIN, LAT_MAX),
        aspect="equal",
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"Water Table — {label}",
    )
    ax.legend(loc="upper right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ── LOADERS ────────────────────────────────────────────────────────
@st.cache_data
def load_levels() -> pd.DataFrame:
    df = pd.read_csv(LEVELS_URL)
    df.columns = [
        normalise_well(c) if c.strip().upper().startswith("W") else c
        for c in df.columns
    ]
    return df

@st.cache_data
def load_coords() -> pd.DataFrame:
    df = pd.read_csv(COORDS_URL)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"well": "well", "lat": "lat", "lon": "lon"})
    df = df[["well", "lat", "lon"]].dropna()
    df["well"] = df["well"].apply(normalise_well)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df.drop_duplicates(subset="well")

# ── APP ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    # Sidebar cache‐clear
    if st.sidebar.button("Clear cache"):
        st.legacy_caching.clear_cache()
        st.experimental_rerun()

    # Load data
    levels = load_levels()
    coords = load_coords()

    # ─ Debug: show which years we actually loaded ──────────────────
    st.write("Years loaded from CSV:", levels["Year"].unique().tolist())

    # Identify well columns
    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No W1…Wn columns in level file.")
        return

    # Sidebar controls
    years = levels["Year"].astype(str)
    yr_sel = st.sidebar.selectbox("Year", years, index=len(years) - 1)
    grid_res = st.sidebar.slider("Grid resolution (px)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif = st.sidebar.button("Generate GIF (all years)")

    # Single‐year plot
    row = levels.loc[levels["Year"].astype(str) == yr_sel, well_cols].iloc[0]
    df_year = row.rename_axis("well").reset_index(name="level")
    df_year["well"] = df_year["well"].apply(normalise_well)
    merged = df_year.merge(coords, on="well", how="inner").dropna()

    if merged.empty:
        st.error("No wells matched between files.")
        return

    tag = "forecast" if int(yr_sel) >= 2025 else "observed"
    fig_img = draw_frame(
        merged["lon"].to_numpy(float),
        merged["lat"].to_numpy(float),
        merged["level"].to_numpy(float),
        f"{yr_sel} ({tag})",
        grid_res,
        n_levels,
    )
    st.image(fig_img)

    with st.expander("Raw data"):
        st.dataframe(merged.set_index("well"), use_container_width=True)

    # GIF across all years
    if make_gif:
        with st.spinner("Building GIF…"):
            frames: list[Image.Image] = []
            for _, row in levels.iterrows():
                yr = int(row["Year"])
                df = row[well_cols].rename_axis("well").reset_index(name="level")
                df["well"] = df["well"].apply(normalise_well)
                df = df.merge(coords, on="well", how="inner").dropna()
                if df.empty:
                    continue
                tag = "forecast" if yr >= 2025 else "observed"
                img = draw_frame(
                    df["lon"].to_numpy(float),
                    df["lat"].to_numpy(float),
                    df["level"].to_numpy(float),
                    f"{yr} ({tag})",
                    grid_res,
                    n_levels,
                )
                frames.append(img)
            if not frames:
                st.error("No frames produced.")
                return
            buf = io.BytesIO()
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=500,
                loop=0,
            )
            buf.seek(0)
        st.subheader("Time-series animation")
        st.image(buf.getvalue())
        st.download_button(
            "Download GIF",
            data=buf.getvalue(),
            file_name="water_table_animation.gif",
            mime="image/gif",
        )

if __name__ == "__main__":
    main()
