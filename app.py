from __future__ import annotations

import io, unicodedata, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import Rbf
from PIL import Image

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 35.80, 36.40
LON_MIN, LON_MAX = 43.60, 44.30

LEVELS_URL = (
    "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/"
    "Monthly_Sea_Level_Data.csv"
)
COORDS_URL = (
    "https://raw.githubusercontent.com/parezdzay/ForcastErbil/main/wells.csv"
)

# ──────────────────────────────────────────────────────────
# HELPERS (unchanged)
# ──────────────────────────────────────────────────────────
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

def draw_frame(lon, lat, z, title, grid_res, n_levels) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    if len(lon) >= 3:
        lon_g, lat_g, z_g = rbf_surface(lon, lat, z, grid_res)
        cf = ax.contourf(lon_g, lat_g, z_g, levels=n_levels,
                         cmap="viridis", alpha=0.75)
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
        title=title,
    )
    ax.legend(loc="upper right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ──────────────────────────────────────────────────────────
# LOADERS (unchanged)
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_levels_raw() -> pd.DataFrame:
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

# ──────────────────────────────────────────────────────────
# MAIN APP (with full 2004–2029 and Period tagging)
# ──────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    # Load raw observed data
    raw = load_levels_raw()
    coords = load_coords()

    # Build a DataFrame of all years 2004–2029
    all_years = pd.DataFrame({"Year": list(range(2004, 2030))})

    # Left-join your observed data onto the full range
    levels = (
        all_years
        .merge(raw, on="Year", how="left")
        .assign(
            Period=lambda df: np.where(df["Year"] >= 2025, "forecast", "observed")
        )
    )

    # Identify well columns W1…W20
    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No W1…Wn columns found in your data.")
        return

    # Sidebar controls
    years_str = levels["Year"].astype(str)
    yr_sel = st.sidebar.selectbox("Year", years_str, index=len(years_str) - 1)
    grid_res = st.sidebar.slider("Grid resolution (px)", 100, 500, 300, 50)
    n_levels = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif = st.sidebar.button("Generate GIF (all years)")

    # Single-year slice
    year_int = int(yr_sel)
    row = levels.loc[levels["Year"] == year_int, well_cols].iloc[0]
    period = levels.loc[levels["Year"] == year_int, "Period"].iloc[0].capitalize()
    df_year = (
        row.rename_axis("well")
        .reset_index(name="level")
    )
    df_year["well"] = df_year["well"].apply(normalise_well)

    merged = df_year.merge(coords, on="well", how="inner").dropna(subset=["level"])

    if merged.empty:
        st.warning(
            f"No well-level data for {yr_sel} ({period})."
            " If this is a forecast year, please populate your CSV."
        )
    else:
        title = f"{period.upper()} — {yr_sel}"
        st.subheader(f"{period.capitalize()} data — {yr_sel}")
        st.image(
            draw_frame(
                merged["lon"].to_numpy(),
                merged["lat"].to_numpy(),
                merged["level"].to_numpy(),
                title,
                grid_res,
                n_levels,
            )
        )

        with st.expander("Raw data for this year"):
            st.dataframe(
                merged.set_index("well"),
                use_container_width=True,
            )

    # GIF across all years
    if make_gif:
        with st.spinner("Building GIF…"):
            frames: list[Image.Image] = []
            for _, yr_row in levels.iterrows():
                yr = int(yr_row["Year"])
                period = yr_row["Period"].capitalize()
                df_f = (
                    yr_row[well_cols]
                    .rename_axis("well")
                    .reset_index(name="level")
                )
                df_f["well"] = df_f["well"].apply(normalise_well)
                df_f = (
                    df_f.merge(coords, on="well", how="inner")
                    .dropna(subset=["level"])
                )
                if df_f.empty:
                    continue
                label = f"{period.upper()} — {yr}"
                frames.append(
                    draw_frame(
                        df_f["lon"].to_numpy(),
                        df_f["lat"].to_numpy(),
                        df_f["level"].to_numpy(),
                        label,
                        grid_res,
                        n_levels,
                    )
                )

            if not frames:
                st.error("No frames could be generated.")
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
