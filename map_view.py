"""
Interactive map visualization for property data.
Uses Folium with district-level coordinates.
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

# Approximate lat/lng for Egyptian districts
DISTRICT_COORDS = {
    "New Cairo": (30.0074, 31.4913),
    "Maadi": (29.9602, 31.2569),
    "Heliopolis": (30.0911, 31.3225),
    "Nasr City": (30.0626, 31.3386),
    "Sheikh Zayed": (30.0444, 30.9419),
    "6th October": (29.9722, 30.9175),
    "Rehab City": (30.0584, 31.4908),
    "Madinaty": (30.1070, 31.6388),
    "New Capital City": (30.0194, 31.7606),
    "Shorouk City": (30.1167, 31.6167),
    "Mostakbal City": (30.0833, 31.7000),
    "Alexandria": (31.2001, 29.9187),
    "Smouha": (31.2156, 29.9401),
    "Downtown Cairo": (30.0444, 31.2357),
    "Garden City": (30.0358, 31.2314),
    "Zamalek": (30.0616, 31.2207),
    "Mokattam": (30.0116, 31.3001),
    "Noor City": (30.0500, 31.7500),
    "El Nozha": (30.1110, 31.3510),
    "Badr City": (30.1333, 31.7167),
    "El Manial": (30.0143, 31.2286),
}

# Center of Cairo for default map view
CAIRO_CENTER = (30.0444, 31.2357)


def create_property_map(df, map_type="markers"):
    """
    Create an interactive Folium map.

    Args:
        df: DataFrame with property data
        map_type: 'markers' for circle markers, 'heat' for heat map
    Returns:
        folium.Map object
    """
    m = folium.Map(
        location=CAIRO_CENTER,
        zoom_start=10,
        tiles="CartoDB dark_matter",
    )

    # Compute district-level stats
    district_stats = (
        df[df["district"].isin(DISTRICT_COORDS.keys())]
        .groupby("district")
        .agg(
            count=("unified_price", "size"),
            median_price=("unified_price", "median"),
            mean_ppsm=("price_per_sqm", "mean"),
            mean_roi=("estimated_roi_percent", "mean"),
            mean_area=("unified_area", "mean"),
        )
        .reset_index()
    )

    if map_type == "heat":
        return _create_heat_map(m, df)
    else:
        return _create_marker_map(m, district_stats)


def _create_marker_map(m, district_stats):
    """Create map with sized/colored circle markers per district."""
    if district_stats.empty:
        return m

    max_count = district_stats["count"].max()
    min_price = district_stats["median_price"].min()
    max_price = district_stats["median_price"].max()
    price_range = max_price - min_price if max_price != min_price else 1

    for _, row in district_stats.iterrows():
        district = row["district"]
        if district not in DISTRICT_COORDS:
            continue

        lat, lng = DISTRICT_COORDS[district]
        count = row["count"]
        median_price = row["median_price"]
        mean_ppsm = row["mean_ppsm"]
        mean_roi = row["mean_roi"]
        mean_area = row["mean_area"]

        # Size based on listing count
        radius = max(8, min(35, (count / max_count) * 35))

        # Color intensity based on price
        price_norm = (median_price - min_price) / price_range
        r = int(255 * price_norm)
        g = int(255 * (1 - price_norm))
        b = 180
        color = f"#{r:02x}{g:02x}{b:02x}"

        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px; color: #000;">
            <h4 style="margin: 0 0 8px 0; border-bottom: 2px solid #000;">{district}</h4>
            <table style="width: 100%; font-size: 13px;">
                <tr><td><b>Listings</b></td><td style="text-align:right;">{count:,}</td></tr>
                <tr><td><b>Median Price</b></td><td style="text-align:right;">{median_price/1e6:.2f}M EGP</td></tr>
                <tr><td><b>Avg Price/sqm</b></td><td style="text-align:right;">{mean_ppsm:,.0f} EGP</td></tr>
                <tr><td><b>Avg Area</b></td><td style="text-align:right;">{mean_area:.0f} sqm</td></tr>
                <tr><td><b>Est. ROI</b></td><td style="text-align:right;">{mean_roi:.1f}%</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=(lat, lng),
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{district}: {count:,} listings",
            fill=True,
            fill_color=color,
            color="#FFFFFF",
            weight=2,
            fill_opacity=0.7,
        ).add_to(m)

    return m


def _create_heat_map(m, df):
    """Create heat map based on property density and price."""
    # Filter to only rows with known district coordinates (vectorized)
    known = df[df["district"].isin(DISTRICT_COORDS.keys())].copy()

    if known.empty:
        return m

    # Map district names to lat/lng in bulk
    known["lat"] = known["district"].map(lambda d: DISTRICT_COORDS[d][0])
    known["lng"] = known["district"].map(lambda d: DISTRICT_COORDS[d][1])

    # Add jitter to spread properties within each district
    rng = np.random.default_rng(42)
    n = len(known)
    known["lat"] = known["lat"] + rng.normal(0, 0.012, n)
    known["lng"] = known["lng"] + rng.normal(0, 0.012, n)

    # Weight by price (capped)
    known["weight"] = known["unified_price"].clip(upper=50e6) / 1e6

    heat_data = known[["lat", "lng", "weight"]].values.tolist()

    if heat_data:
        HeatMap(
            heat_data,
            radius=20,
            blur=25,
            max_zoom=13,
            min_opacity=0.3,
            gradient={
                "0.0": "#000033",
                "0.2": "#0044AA",
                "0.4": "#00AAFF",
                "0.6": "#44DDFF",
                "0.8": "#FF6699",
                "1.0": "#FFFFFF",
            },
        ).add_to(m)

    return m
