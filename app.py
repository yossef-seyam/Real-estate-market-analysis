"""
Property Market Analysis Dashboard
Modern black and white theme - Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Property Market Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for black/white theme, no emojis --
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0E1117; }

    .stMetric {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2A2A3E;
    }
    .stMetric label { color: #888 !important; font-size: 14px !important; }
    .stMetric .metric-value { color: #FFF !important; }

    h1, h2, h3, h4 { color: #FAFAFA !important; letter-spacing: -0.5px; }
    h1 { font-weight: 700 !important; font-size: 2.2rem !important; }
    h2 { font-weight: 600 !important; font-size: 1.6rem !important; border-bottom: 1px solid #2A2A3E; padding-bottom: 8px; }
    h3 { font-weight: 500 !important; font-size: 1.2rem !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A2E;
        border-radius: 8px;
        color: #888;
        padding: 10px 20px;
        border: 1px solid #2A2A3E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #000 !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a14 0%, #111128 100%);
        border-right: 1px solid #2A2A3E;
    }

    .recommendation-card {
        background: #1A1A2E;
        border: 1px solid #2A2A3E;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid #2A2A3E;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .kpi-card h3 { margin: 0; font-size: 2rem !important; color: #FFF !important; }
    .kpi-card p { margin: 4px 0 0 0; color: #888; font-size: 13px; }

    .search-result {
        background: #1A1A2E;
        border: 1px solid #2A2A3E;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }

    div[data-testid="stDataFrame"] { border: 1px solid #2A2A3E; border-radius: 8px; }

    /* Buttons - vibrant and readable */
    .stButton > button {
        background: linear-gradient(135deg, #534AB7 0%, #7C6BF0 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7C6BF0 0%, #9D8FFF 100%) !important;
        box-shadow: 0 4px 16px rgba(124, 107, 240, 0.4) !important;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #0F6E56 0%, #2DD4A8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2DD4A8 0%, #5AEFC4 100%) !important;
        box-shadow: 0 4px 16px rgba(45, 212, 168, 0.4) !important;
    }

    /* Radio buttons */
    .stRadio > div { gap: 8px; }
    .stRadio label {
        background: #1A1A2E !important;
        border: 1px solid #2A2A3E !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        color: #CCC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -- Bypass MongoDB and Load CSV Natively --
import os

CSV_PATH = os.path.join("Cleaning Data", "final_unified_property_data.csv")

# -- Data Loading Helper (Option B: Local CSV) --
@st.cache_data
def load_data_from_csv():
    csv_path = "Cleaning Data/final_unified_property_data.csv"
    try:
        df = pd.read_csv(csv_path)
        # Ensure ROI column exists if not present
        if "estimated_roi_percent" not in df.columns:
            df["estimated_roi_percent"] = 7.0 # Default value
        # Ensure price_per_sqm exists
        if "price_per_sqm" not in df.columns:
            df["price_per_sqm"] = df["unified_price"] / df["unified_area"]
        return df
    except Exception as e:
        st.error(f"Error loading local CSV: {e}")
        return pd.DataFrame()

# Local implementation of DB functions
def get_district_list_local(df):
    districts = df["district"].unique()
    districts = [d for d in districts if d and str(d).lower() not in ["unknown", "other", "nan"]]
    return sorted(districts)

def get_district_stats_local(df):
    stats = df.groupby("district").agg(
        count=("unified_price", "size"),
        mean_price=("unified_price", "mean"),
        mean_ppsm=("price_per_sqm", "mean"),
        mean_roi=("estimated_roi_percent", "mean"),
        std_price=("unified_price", "std")
    ).reset_index()
    stats = stats[~stats["district"].isin(["unknown", "Other"])]
    stats["cv"] = stats["std_price"] / stats["mean_price"]
    return stats.sort_values("count", ascending=False)

def search_properties_local(df, district=None, min_price=None, max_price=None, min_area=None, max_area=None, min_rooms=None, max_rooms=None, min_bathrooms=None, max_bathrooms=None, amenities=None, limit=100):
    filtered = df.copy()
    if district and district != "All":
        filtered = filtered[filtered["district"] == district]
    if min_price: filtered = filtered[filtered["unified_price"] >= min_price]
    if max_price: filtered = filtered[filtered["unified_price"] <= max_price]
    if min_area: filtered = filtered[filtered["unified_area"] >= min_area]
    if max_area: filtered = filtered[filtered["unified_area"] <= max_area]
    if min_rooms: filtered = filtered[filtered["unified_rooms"] >= min_rooms]
    if max_rooms: filtered = filtered[filtered["unified_rooms"] <= max_rooms]
    if min_bathrooms: filtered = filtered[filtered["unified_bathrooms"] >= min_bathrooms]
    if max_bathrooms: filtered = filtered[filtered["unified_bathrooms"] <= max_bathrooms]
    
    if amenities:
        for amenity in amenities:
            col = f"has_{amenity.lower()}"
            if col in filtered.columns:
                filtered = filtered[filtered[col] == 1]
    
    return filtered.head(limit)

from charts import (
    chart_price_by_district,
    chart_ppsm_by_district,
    chart_opportunity_matrix,
    chart_amenity_forest_plot,
    chart_amenity_premium_rank,
    chart_correlation_heatmap,
    chart_price_distribution,
    chart_price_by_rooms,
    chart_roi_by_district,
    chart_risk_consistency_map,
    chart_model_comparison,
)
from map_view import create_property_map
from ai_engine import predict_price, get_investment_recommendation, get_gemini_recommendation, model_available


# -- Sidebar Navigation --
st.sidebar.markdown("# Property Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Market Analysis",
        "Property Map",
        "Property Search",
        "AI Recommendations",
        "Data Explorer",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")


# -- Load Local Data (Option B) --
df = load_data_from_csv()

if df.empty:
    st.error("The local data file was not found or is empty. Please ensure 'Cleaning Data/final_unified_property_data.csv' exists.")
    st.stop()

st.sidebar.success(f"**Local Data Mode**: {len(df):,} properties loaded")


# ============================================================
# PAGE: Dashboard
# ============================================================
if page == "Dashboard":
    st.title("Property Market Dashboard")
    st.markdown("Overview of the Egyptian real estate market based on aggregated listing data.")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        median_price = df["unified_price"].median()
        st.metric("Median Price", f"{median_price/1e6:.2f}M EGP")
    with col3:
        avg_area = df["unified_area"].mean()
        st.metric("Avg Area", f"{avg_area:.0f} sqm")
    with col4:
        avg_ppsm = df["price_per_sqm"].mean()
        st.metric("Avg Price/sqm", f"{avg_ppsm:,.0f} EGP")
    with col5:
        avg_roi = df["estimated_roi_percent"].mean()
        st.metric("Avg ROI", f"{avg_roi:.1f}%")

    st.markdown("---")

    # Two column layout for overview charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(chart_price_distribution(df), use_container_width=True)

    with col_right:
        st.plotly_chart(chart_price_by_district(df), use_container_width=True)

    st.markdown("---")

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.plotly_chart(chart_ppsm_by_district(df), use_container_width=True)

    with col_right2:
        st.plotly_chart(chart_opportunity_matrix(df), use_container_width=True)

    # District summary table
    st.markdown("## District Summary")
    stats = get_district_stats_local(df)
    if not stats.empty:
        display_stats = stats[["district", "count", "mean_price", "mean_ppsm", "mean_roi", "cv"]].copy()
        display_stats.columns = ["District", "Listings", "Avg Price (EGP)", "Avg PPSM (EGP)", "Avg ROI (%)", "Price CV"]
        display_stats["Avg Price (EGP)"] = display_stats["Avg Price (EGP)"].apply(lambda x: f"{x:,.0f}")
        display_stats["Avg PPSM (EGP)"] = display_stats["Avg PPSM (EGP)"].apply(lambda x: f"{x:,.0f}")
        display_stats["Avg ROI (%)"] = display_stats["Avg ROI (%)"].apply(lambda x: f"{x:.1f}")
        display_stats["Price CV"] = display_stats["Price CV"].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_stats, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: Market Analysis
# ============================================================
elif page == "Market Analysis":
    st.title("Market Analysis")
    st.markdown("Interactive charts exploring price patterns, amenity impacts, and market structure.")

    tab1, tab2, tab3 = st.tabs(["Pricing", "Amenities", "Statistical"])

    with tab1:
        st.plotly_chart(chart_price_by_district(df), use_container_width=True)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_ppsm_by_district(df), use_container_width=True)
        with col2:
            st.plotly_chart(chart_opportunity_matrix(df), use_container_width=True)

        st.markdown("---")
        st.plotly_chart(chart_price_distribution(df), use_container_width=True)

        st.markdown("---")
        st.plotly_chart(chart_price_by_rooms(df), use_container_width=True)

    with tab2:
        st.plotly_chart(chart_amenity_forest_plot(df), use_container_width=True)
        st.markdown("---")
        st.plotly_chart(chart_amenity_premium_rank(df), use_container_width=True)

    with tab3:
        st.plotly_chart(chart_correlation_heatmap(df), use_container_width=True)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_roi_by_district(df), use_container_width=True)
        with col2:
            st.plotly_chart(chart_risk_consistency_map(df), use_container_width=True)

        st.markdown("---")
        st.plotly_chart(chart_model_comparison(), use_container_width=True)


# ============================================================
# PAGE: Property Map
# ============================================================
elif page == "Property Map":
    st.title("Property Map")
    st.markdown("Geographic visualization of property listings across Egyptian districts.")

    map_type = st.radio(
        "Map Type",
        ["District Markers", "Price Heat Map"],
        horizontal=True,
    )

    map_mode = "markers" if map_type == "District Markers" else "heat"

    with st.spinner("Loading map..."):
        from streamlit_folium import st_folium
        m = create_property_map(df, map_type=map_mode)
        st_folium(m, width=None, height=600, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Note**: Properties are mapped to their district center coordinates. "
        "Marker size represents listing count. Color intensity represents median price level. "
        "Click a marker for detailed district statistics."
    )


# ============================================================
# PAGE: Property Search
# ============================================================
elif page == "Property Search":
    st.title("Property Search")
    st.markdown("Find properties matching your criteria. Results are queried directly from the database.")

    col1, col2, col3 = st.columns(3)

    with col1:
        districts = ["All"] + get_district_list_local(df)
        selected_district = st.selectbox("District", districts)

        min_area = st.number_input("Min Area (sqm)", min_value=0, value=50, step=10)
        max_area = st.number_input("Max Area (sqm)", min_value=0, value=500, step=10)

    with col2:
        min_price = st.number_input(
            "Min Price (EGP)", min_value=0, value=100000, step=100000,
            help="Minimum property price"
        )
        max_price = st.number_input(
            "Max Price (EGP)", min_value=0, value=50000000, step=500000,
            help="Maximum property price"
        )
        min_rooms = st.number_input("Min Rooms", min_value=0, value=1, step=1)
        max_rooms = st.number_input("Max Rooms", min_value=0, value=5, step=1)

    with col3:
        min_bath = st.number_input("Min Bathrooms", min_value=0, value=1, step=1)
        max_bath = st.number_input("Max Bathrooms", min_value=0, value=5, step=1)

        amenity_options = ["Elevator", "Security", "Balcony", "Pool", "Garden", "Parking"]
        selected_amenities = st.multiselect("Required Amenities", amenity_options)

    max_results = st.slider("Max Results", min_value=10, max_value=500, value=100, step=10)

    if st.button("Search", type="primary", use_container_width=True):
        with st.spinner("Searching local data..."):
            results = search_properties_local(
                df,
                district=selected_district,
                min_price=min_price if min_price > 0 else None,
                max_price=max_price if max_price > 0 else None,
                min_area=min_area if min_area > 0 else None,
                max_area=max_area if max_area > 0 else None,
                min_rooms=min_rooms if min_rooms > 0 else None,
                max_rooms=max_rooms if max_rooms > 0 else None,
                min_bathrooms=min_bath if min_bath > 0 else None,
                max_bathrooms=max_bath if max_bath > 0 else None,
                amenities=[a.lower() for a in selected_amenities] if selected_amenities else None,
                limit=max_results,
            )

        if results.empty:
            st.warning("No properties found matching your criteria. Try adjusting the filters.")
        else:
            st.markdown(f"### Found {len(results):,} properties")

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Median Price", f"{results['unified_price'].median()/1e6:.2f}M")
            with c2:
                st.metric("Avg Area", f"{results['unified_area'].mean():.0f} sqm")
            with c3:
                st.metric("Avg Price/sqm", f"{results['price_per_sqm'].mean():,.0f}")
            with c4:
                st.metric("Avg ROI", f"{results['estimated_roi_percent'].mean():.1f}%")

            # Results table
            display_cols = [
                "district", "unified_price", "unified_area", "price_per_sqm",
                "unified_rooms", "unified_bathrooms", "estimated_roi_percent",
                "has_elevator", "has_security", "has_balcony", "has_pool",
                "has_garden", "has_parking",
            ]
            existing_cols = [c for c in display_cols if c in results.columns]
            display_df = results[existing_cols].copy()

            rename_map = {
                "district": "District",
                "unified_price": "Price (EGP)",
                "unified_area": "Area (sqm)",
                "price_per_sqm": "Price/sqm",
                "unified_rooms": "Rooms",
                "unified_bathrooms": "Bathrooms",
                "estimated_roi_percent": "ROI %",
                "has_elevator": "Elevator",
                "has_security": "Security",
                "has_balcony": "Balcony",
                "has_pool": "Pool",
                "has_garden": "Garden",
                "has_parking": "Parking",
            }
            display_df = display_df.rename(columns=rename_map)

            if "Price (EGP)" in display_df.columns:
                display_df["Price (EGP)"] = display_df["Price (EGP)"].apply(lambda x: f"{x:,.0f}")
            if "Price/sqm" in display_df.columns:
                display_df["Price/sqm"] = display_df["Price/sqm"].apply(lambda x: f"{x:,.0f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: AI Recommendations
# ============================================================
elif page == "AI Recommendations":
    st.title("AI-Powered Analysis")
    st.markdown(
        "Enter property details below and the trained machine learning model will "
        "predict the fair market value and provide investment recommendations."
    )

    if not model_available():
        st.error("Model file not found (rf_model.pkl). The AI features require the trained model.")
        st.stop()

    # API key input in sidebar
    api_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key",
                                     help="Enter your Google Gemini API key for AI-powered recommendations")

    st.markdown("## Property Details")

    col1, col2 = st.columns(2)

    with col1:
        districts = get_district_list_local(df)
        ai_district = st.selectbox("District", districts, key="ai_district")
        ai_area = st.number_input("Area (sqm)", min_value=10, max_value=2000, value=120, step=10)
        ai_rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
        ai_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)

    with col2:
        st.markdown("**Amenities**")
        ai_elevator = 1 if st.checkbox("Elevator") else 0
        ai_security = 1 if st.checkbox("Security") else 0
        ai_balcony = 1 if st.checkbox("Balcony") else 0
        ai_pool = 1 if st.checkbox("Pool") else 0
        ai_garden = 1 if st.checkbox("Garden") else 0
        ai_parking = 1 if st.checkbox("Parking") else 0

    if st.button("Get AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Running model prediction..."):
            result = predict_price(
                area=ai_area,
                rooms=ai_rooms,
                bathrooms=ai_bathrooms,
                district=ai_district,
                has_elevator=ai_elevator,
                has_security=ai_security,
                has_balcony=ai_balcony,
                has_pool=ai_pool,
                has_garden=ai_garden,
                has_parking=ai_parking,
            )

        if result is None:
            st.error("Model could not make a prediction. Check if rf_model.pkl is valid.")
        elif "error" in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            st.markdown("---")
            st.markdown("## Prediction Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="kpi-card"><h3>{result["predicted_price"]/1e6:.2f}M EGP</h3>'
                    f'<p>Estimated Fair Market Value</p></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="kpi-card"><h3>{result["price_per_sqm"]:,.0f} EGP</h3>'
                    f'<p>Predicted Price per sqm</p></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="kpi-card"><h3>{result["estimated_monthly_rent"]/1e3:.1f}K EGP</h3>'
                    f'<p>Estimated Monthly Rent</p></div>',
                    unsafe_allow_html=True,
                )

            # Detailed breakdown
            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### Financial Summary")
                fin_data = {
                    "Metric": [
                        "Predicted Price",
                        "Price per sqm",
                        "District Avg Price/sqm",
                        "Estimated Annual Rent",
                        "Estimated Monthly Rent",
                        "Estimated ROI",
                    ],
                    "Value": [
                        f"{result['predicted_price']:,.0f} EGP",
                        f"{result['price_per_sqm']:,.0f} EGP",
                        f"{result['district_avg_ppsm']:,.0f} EGP",
                        f"{result['estimated_annual_rent']:,.0f} EGP",
                        f"{result['estimated_monthly_rent']:,.0f} EGP",
                        f"{result['estimated_roi']:.1f}%",
                    ],
                }
                st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)

            with col_b:
                st.markdown("### AI Recommendations")

                if api_key:
                    with st.spinner("Generating AI-powered analysis..."):
                        ai_text = get_gemini_recommendation(
                            api_key=api_key,
                            district=ai_district,
                            area=ai_area,
                            rooms=ai_rooms,
                            bathrooms=ai_bathrooms,
                            amenities={
                                "elevator": ai_elevator, "security": ai_security,
                                "balcony": ai_balcony, "pool": ai_pool,
                                "garden": ai_garden, "parking": ai_parking,
                            },
                            prediction=result,
                            df=df,
                        )
                    if ai_text and not ai_text.startswith("ERROR:"):
                        st.markdown(
                            f'<div class="recommendation-card">{ai_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        error_msg = ai_text.replace("ERROR: ", "") if ai_text else "Unknown error"
                        st.warning(f"AI recommendation failed: {error_msg}")
                else:
                    # Fallback to data-driven recommendations
                    recommendations = get_investment_recommendation(
                        df, ai_district, ai_area, ai_rooms, result["predicted_price"]
                    )
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(
                            f'<div class="recommendation-card">'
                            f'<strong>{i}.</strong> {rec}</div>',
                            unsafe_allow_html=True,
                        )
                    st.info("Enter a Gemini API key in the sidebar for AI-powered recommendations.")


# ============================================================
# PAGE: Data Explorer
# ============================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Browse the raw property dataset with filtering and sorting.")

    # Quick filters
    col1, col2 = st.columns(2)
    with col1:
        filter_district = st.selectbox("Filter by District", ["All"] + get_district_list_local(df), key="explorer_district")
    with col2:
        sort_by = st.selectbox("Sort by", ["unified_price", "unified_area", "price_per_sqm", "estimated_roi_percent"])

    filtered = df.copy()
    if filter_district != "All":
        filtered = filtered[filtered["district"] == filter_district]

    sort_order = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True)
    filtered = filtered.sort_values(sort_by, ascending=(sort_order == "Ascending"))

    st.markdown(f"Showing {len(filtered):,} of {len(df):,} properties")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_properties.csv",
        mime="text/csv",
    )
