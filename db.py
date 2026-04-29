"""
Local CSV connection helper for the Property Analysis Dashboard.
Replaces the MongoDB Atlas architecture with local Pandas operations.
"""

import pandas as pd
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "Cleaning Data", "final_unified_property_data.csv")
_df_cache = None

def get_all_properties():
    """Load all properties as a DataFrame."""
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = pd.read_csv(CSV_PATH)
        except Exception:
            _df_cache = pd.DataFrame()
    return _df_cache


def get_district_list():
    """Get sorted list of unique districts."""
    df = get_all_properties()
    if df.empty or "district" not in df.columns:
        return []
    districts = df["district"].dropna().unique()
    districts = [d for d in districts if str(d).lower() not in ["unknown", "other"]]
    return sorted(districts)


def get_district_stats():
    """Aggregation pipeline equivalent for district-level statistics."""
    df = get_all_properties()
    if df.empty or "district" not in df.columns:
        return pd.DataFrame()
        
    valid_df = df[~df["district"].isin(["unknown", "Other", "Unknown"])]
    
    stats = valid_df.groupby("district").agg(
        count=("unified_price", "count"),
        mean_price=("unified_price", "mean"),
        median_price=("unified_price", "median"),
        mean_ppsm=("price_per_sqm", "mean"),
        mean_area=("unified_area", "mean"),
        mean_rooms=("unified_rooms", "mean"),
        mean_roi=("estimated_roi_percent", "mean"),
        std_price=("unified_price", "std"),
    ).reset_index()
    
    stats = stats.sort_values("count", ascending=False)
    stats["cv"] = stats["std_price"] / stats["mean_price"]
    return stats


def search_properties(
    district=None,
    min_price=None,
    max_price=None,
    min_area=None,
    max_area=None,
    min_rooms=None,
    max_rooms=None,
    min_bathrooms=None,
    max_bathrooms=None,
    amenities=None,
    limit=100,
):
    """Search properties with multiple filter criteria using Pandas."""
    df = get_all_properties()
    if df.empty:
        return df
        
    filtered = df.copy()
    
    if district and district != "All":
        filtered = filtered[filtered["district"] == district]
        
    if min_price is not None:
        filtered = filtered[filtered["unified_price"] >= min_price]
    if max_price is not None:
        filtered = filtered[filtered["unified_price"] <= max_price]
        
    if min_area is not None:
        filtered = filtered[filtered["unified_area"] >= min_area]
    if max_area is not None:
        filtered = filtered[filtered["unified_area"] <= max_area]
        
    if min_rooms is not None:
        filtered = filtered[filtered["unified_rooms"] >= min_rooms]
    if max_rooms is not None:
        filtered = filtered[filtered["unified_rooms"] <= max_rooms]
        
    if min_bathrooms is not None:
        filtered = filtered[filtered["unified_bathrooms"] >= min_bathrooms]
    if max_bathrooms is not None:
        filtered = filtered[filtered["unified_bathrooms"] <= max_bathrooms]

    if amenities:
        for amenity in amenities:
            col = f"has_{amenity.lower()}"
            if col in filtered.columns:
                # Assuming flags are 1/0 or True/False
                filtered = filtered[filtered[col] == 1]

    if limit > 0:
        filtered = filtered.head(limit)
        
    return filtered


def check_connection():
    """Check if the CSV file is accessible."""
    return os.path.exists(CSV_PATH)


def get_document_count():
    """Get total number of properties in the dataset."""
    df = get_all_properties()
    return len(df)
