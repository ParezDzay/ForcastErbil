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
# HELPERS
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

def draw_frame(
    lon, lat, z, labels, title: str, grid_res: int, n_levels: int
) -> Image.Image:
    """
    lon, lat, z: arrays of floats
    labels: list/array of well names (strings), same length
    """
    # Convert + clean
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    z_arr   = np.asarray(z,   dtype=float)
    mask    = (~np.isnan(lon_arr)) & (~np.isnan(lat_arr)) & (~np.isnan(z_arr))
    lon_arr, lat_arr, z_arr = lon_arr[mask], lat_arr[mask], z_arr[mask]
    lbls    = np.asarray(labels)[mask]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    # Interpolate surface if possible
    if len(lon_arr) >= 3:
        lon_g, lat_g, z_g = rbf_surface(lon_arr, lat_arr, z_arr, grid_res)
        cf = ax.contourf(
            lon_g, lat_g, z_g, levels=n_levels, cmap="viridis", alpha=0.6
        )
        fig.colorbar(cf, ax=ax, label="Level")
    else:
        sc = ax.scatter(lon_arr, lat_arr, c=z_arr, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Level")

    # Scatter wells & annotate
    ax.scatter(lon_arr, lat_arr, c=z_arr, edgecolors="black", s=70, label="Wells")
    for x, y, name in zip(lon_arr, lat_arr, lbls):
        ax.text(
            x, y,
            name,
            fontsize=8,
            ha="right",
            va="bottom",
            color="black",
            weight="bold",
            alpha=0.8,
        )

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
# MAIN APP
# ──────────────────────────────────────────────────────────
def main():
    st.set_page_config(layout="wide")
    st.title("Groundwater Dashboard")

    raw    = load_levels_raw()
    coords = load_coords()

    # Guarantee full 2004–2029
    all_years = pd.DataFrame({"Year": list(range(2004, 2030))})
    levels    = (
        all_years
        .merge(raw, on="Year", how="left")
        .assign(
            Period=lambda df: np.where(df["Year"] >= 2025, "forecast", "observed")
        )
    )

    well_cols = [c for c in levels.columns if c.startswith("W")]
    if not well_cols:
        st.error("No W# columns found.")
        return

    # Sidebar
    years_str = levels["Year"].astype(str)
    yr_sel    = st.sidebar.selectbox("Year", years_str, index=len(years_str)-1)
    grid_res  = st.sidebar.slider("Grid resolution (px)", 100, 500, 300, 50)
    n_levels  = st.sidebar.slider("Contour levels", 5, 30, 15, 1)
    make_gif  = st.sidebar.button("Generate GIF")

    # Single year
    year_int = int(yr_sel)
    row      = levels.loc[levels["Year"] == year_int, well_cols].iloc[0]
    period   = levels.loc[levels["Year"] == year_int, "Period"].iloc[0].capitalize()
    df_year  = row.rename_axis("well").reset_index(name="level")
    df_year["well"] = df_year["well"].apply(normalise_well)
    merged   = df_year.merge(coords, on="well", how="inner").dropna(subset=["level","lat","lon"])

    if merged.empty:
        st.warning(f"No data for {period} {yr_sel}.")
    else:
        title = f"{period.upper()} — {yr_sel}"
        st.subheader(f"{period.capitalize()} data — {yr_sel}")
        st.image(draw_frame(
            merged["lon"],
            merged["lat"],
            merged["level"],
            merged["well"],   # pass labels here
            title,
            grid_res,
            n_levels,
        ))
        with st.expander("Raw data"):
            st.dataframe(merged.set_index("well"), use_container_width=True)

    # GIF
    if make_gif:
        with st.spinner("Building GIF…"):
            frames: list[Image.Image] = []
            for _, row in levels.iterrows():
                yr     = int(row["Year"])
                period= row["Period"].capitalize()
                df_f   = row[well_cols].rename_axis("well").reset_index(name="level")
                df_f["well"] = df_f["well"].apply(normalise_well)
                df_f       = df_f.merge(coords, on="well", how="inner").dropna(subset=["level","lat","lon"])
                if df_f.empty:
                    continue
                label = f"{period.upper()} — {yr}"
                frames.append(draw_frame(
                    df_f["lon"],
                    df_f["lat"],
                    df_f["level"],
                    df_f["well"],   # labels here too
                    label,
                    grid_res,
                    n_levels,
                ))

            if not frames:
                st.error("No frames generated.")
                return

            buf = io.BytesIO()
            frames[0].save(
                buf, format="GIF", save_all=True,
                append_images=frames[1:], duration=1200, loop=0
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
