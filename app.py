import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from recommender import get_recommendations

# 1. Basic Page Configuration
st.set_page_config(page_title="Aqarmap Market Intelligence", page_icon="🏢", layout="wide")

# ==========================================
# 💎 FUTURISTIC & PROFESSIONAL UI INJECTION
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 20, 29) 0%, rgb(24, 26, 40) 90%);
        color: #E2E8F0;
    }
    
    h1, h2, h3 {
        color: #38bdf8 !important;
        text-shadow: 0px 4px 15px rgba(56, 189, 248, 0.4);
    }

    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border: 1px solid #38bdf8;
        box-shadow: 0 4px 30px rgba(56, 189, 248, 0.3);
    }
    
    div[data-testid="metric-container"] > div > div {
        color: #f8fafc;
        font-weight: bold;
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 1.1rem;
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏢 Real Estate Market Intelligence Dashboard")
st.markdown("Advanced intelligent analysis for real estate pricing using extracted data and AI technology.")

# Helper for formatting numbers cleanly (e.g., Millions and Thousands)
def format_currency(val):
    if pd.isna(val): return "N/A"
    if val >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M EGP"
    elif val >= 1_000:
        return f"{val / 1_000:,.1f}K EGP"
    return f"{val:,.0f} EGP"

# 2. Data Loading (Using Cache)
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_properties.csv")  # The generated raw CSV
    
    df = df.dropna(subset=['latitude', 'longitude'])
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Restrict to geographical boundaries of Egypt to fix map outliers (Cyprus issue)
    # The absolute northern limit of Egypt's coast is approx 31.36. 
    # Anything above 31.4 is almost certainly in the sea.
    df = df[(df['latitude'] >= 22.0) & (df['latitude'] <= 31.42)]
    df = df[(df['longitude'] >= 24.0) & (df['longitude'] <= 37.0)]
    
    import re
    def extract_info(url):
        if not isinstance(url, str): return 'Unknown', 'Unknown'
        # Flexible Regex: We split on for-sale and take the segments.
        # This recovers all properties while we clean the "Type" in the next step.
        match = re.search(r'for-sale-([a-z0-9-]+?)-([a-z0-9-]+.*)', url)
        if match:
            prop_type = match.group(1).title() 
            # If the type is actually a city name (like Cairo/Alexandria), we label it 'Property'
            if prop_type in ['Cairo', 'Alexandria', 'Giza', 'North', 'The']:
                prop_type = 'Property'
                
            location_raw = match.group(2).split('?')[0].strip('/').replace('-', ' ').title()
            return prop_type, location_raw
            
        return 'Property', 'Unknown'
        
    df[['property_type', 'district']] = df['url'].apply(lambda u: pd.Series(extract_info(u)))
    
    # Recalculate price per meter numerically safely
    df['price_per_meter'] = pd.to_numeric(df['unified_price'] / df['unified_area'], errors='coerce')
    df = df.dropna(subset=['price_per_meter'])
    
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("🔍 Filters & Controls")

min_price = int(df['unified_price'].min())
max_price = int(df['unified_price'].max()) # Use absolute max instead of quantile to catch premium properties

selected_price_range = st.sidebar.slider(
    "Select Budget (EGP):",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, int(df['unified_price'].quantile(0.70))), # Default value ignores super outliers
    step=250000
)

# Convert to string/title for UI
prop_options = sorted([p for p in df['property_type'].unique() if str(p) != 'Unknown'])
selected_props = st.sidebar.multiselect("Property Types:", options=prop_options, default=prop_options)

districts_opt = sorted([d for d in df['district'].unique() if str(d) != 'Unknown'])
selected_districts = st.sidebar.multiselect("District / Area:", options=districts_opt, default=districts_opt)

# Area (Square Meters) Filter
min_area = int(df['unified_area'].min())
max_area = int(df['unified_area'].quantile(0.99)) # Drop 1% extreme mansions so slider is usable
selected_area_range = st.sidebar.slider(
    "Select Area Range (m²):",
    min_value=min_area,
    max_value=max_area,
    value=(min_area, max_area),
    step=10
)

# Rooms options must be integers instead of floats (10.0 -> 10)
rooms_options = sorted([int(r) for r in df['unified_rooms'].dropna().unique()])
selected_rooms = st.sidebar.multiselect("Number of Rooms:", options=rooms_options, default=rooms_options[:3])

# Handling empty selections gracefully => if user unselects all, assume ALL items to prevent 0 results
active_districts = selected_districts if selected_districts else districts_opt
active_props = selected_props if selected_props else prop_options
active_rooms = selected_rooms if selected_rooms else rooms_options

# 4. Filter Data
filtered_df = df[
    (df['unified_price'] >= selected_price_range[0]) &
    (df['unified_price'] <= selected_price_range[1]) &
    (df['unified_area'] >= selected_area_range[0]) &
    (df['unified_area'] <= selected_area_range[1]) &
    (df['unified_rooms'].isin(active_rooms)) &
    (df['district'].isin(active_districts)) &
    (df['property_type'].isin(active_props))
]

# 5. Fast KPIs
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Listings", f"{len(filtered_df):,}")
with col2:
    st.metric("Average Price", format_currency(filtered_df['unified_price'].mean()))
with col3:
    st.metric("Median Price", format_currency(filtered_df['unified_price'].median()))
with col4:
    avg_price_meter = filtered_df['price_per_meter'].mean()
    st.metric("Avg Price/m²", f"{avg_price_meter:,.0f} EGP" if not pd.isna(avg_price_meter) else "N/A")
with col5:
    avg_area = filtered_df['unified_area'].mean()
    st.metric("Average Area", f"{avg_area:,.0f} m²" if not pd.isna(avg_area) else "N/A")

st.markdown("---")

# 6. Tabs
tab1, tab2 = st.tabs(["📊 Market Analytics", "🤖 Smart Recommender (AI)"])

with tab1:
    st.subheader("📍 Property Distribution Map")
    
    if not filtered_df.empty:
        fig = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="unified_price",
            size="unified_area",
            hover_name="title",
            hover_data={"latitude": False, "longitude": False, "unified_price": True, "unified_rooms": True},
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=15,
            zoom=10
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("⚠️ No properties match these filters. Try adjusting your budget or room requirements.")

    st.markdown("---")
    
    st.subheader("📊 Average Price by District (Bar Chart)")
    if not filtered_df.empty:
        dist_filtered = filtered_df[filtered_df['district'] != 'Unknown']
        if not dist_filtered.empty:
            district_price = dist_filtered.groupby('district')['price_per_meter'].mean().reset_index()
            district_price = district_price.sort_values(by='price_per_meter', ascending=False)
            
            fig_bar = px.bar(
                district_price.head(30), 
                x='district', 
                y='price_per_meter', 
                title="Top 30 Most Expensive Districts by Price/m²",
                labels={'district': 'District', 'price_per_meter': 'Price per m² (EGP)'},
                color='price_per_meter', 
                color_continuous_scale='Viridis',
                template='plotly_dark' # Matches our modern UI
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, width="stretch")
    
    st.markdown("---")
    
    st.subheader("📦 Feature Impact on Price (Box Plot)")
    st.markdown("""
    **How to read this chart:** 
    A Box Plot displays the statistical distribution of property prices based on the presence of a specific feature. 
    - The middle line is the **Median** price.
    - The box represents the **middle 50%** of the properties (Interquartile Range).
    - The lines (whiskers) show the **full range**, excluding extreme outliers.
    
    👉 **Action:** Select an amenity below to see how it statistically shifts the property values!
    """)
    
    if not filtered_df.empty:
        available_features = [col for col in filtered_df.columns if col.startswith('has_')]
        
        if available_features:
            feature_to_check = st.selectbox("Select a Feature to Analyze:", available_features, format_func=lambda x: str(x).replace('has_', 'Has ').title())
            
            box_df = filtered_df.copy()
            box_df[feature_to_check] = box_df[feature_to_check].map({1: 'Yes', 0: 'No'}).fillna('Unknown')
            
            # Drop top 5% just for the box plot visualization so the boxes aren't completely squished by billionaires
            q_high = box_df['unified_price'].quantile(0.95)
            box_df_filtered = box_df[box_df['unified_price'] <= q_high]
            
            fig_box = px.box(
                box_df_filtered, 
                x=feature_to_check, 
                y='unified_price', 
                color=feature_to_check,
                title=f"Statistical Impact of '{str(feature_to_check).replace('has_', '').title()}' on Prices (Excluding top 5% outliers)",
                labels={feature_to_check: 'Feature Presence', 'unified_price': 'Total Price (EGP)'},
                template='plotly_dark'
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, width="stretch")
        else:
            st.info("No amenity features are present in the current dataset.")

with tab2:
    st.subheader("🤖 AI Smart Recommender (KNN Algorithm)")
    st.markdown("Find your perfect property! This AI calculates exact distances across Geographical Location, Price, and Area using `MinMaxScaler` and custom weights to find the best matching neighbors.")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        target_price = st.number_input("Target Budget (EGP):", min_value=100000, value=int(selected_price_range[1]), step=50000)
        target_rooms = st.number_input("Minimum Rooms:", min_value=1, value=3)
        
    with rec_col2:
        district_coords = df.groupby('district')[['latitude', 'longitude']].mean().reset_index()
        valid_districts = sorted([d for d in district_coords['district'] if d != 'Unknown'])
        
        target_district = st.selectbox("Preferred Area:", valid_districts)
        
        dist_data = district_coords[district_coords['district'] == target_district]
        if not dist_data.empty:
            target_lat = float(dist_data.iloc[0]['latitude'])
            target_lon = float(dist_data.iloc[0]['longitude'])
        else:
            target_lat, target_lon = 30.01, 31.42 
        
    with rec_col3:
        st.write("Algorithm Weights (Importance):")
        w_loc = st.slider("Location Proximity (%)", 0, 100, 40) / 100.0
        w_price = st.slider("Price Match (%)", 0, 100, 40) / 100.0
        w_area = st.slider("Area & Amenities (%)", 0, 100, 20) / 100.0
        
    if st.button("🔍 Find Top 5 Matches", type="primary"):
        with st.spinner('Calculating distances with AI...'):
            try:
                recs = get_recommendations(df, target_lat, target_lon, target_price, min_rooms=target_rooms, 
                                           weight_loc=w_loc, weight_price=w_price, weight_area=w_area, weight_amenities=0.0)
                
                if not recs.empty:
                    st.success("🎉 Best matches found!")
                    for i, (_, row) in enumerate(recs.iterrows()):
                        clean_rooms = int(row['unified_rooms']) if not pd.isna(row['unified_rooms']) else 'N/A'
                        st.markdown(f"### #{i+1} - {row['title']}")
                        st.markdown(f"**Price:** {format_currency(row['unified_price'])} | **Area:** {row['unified_area']} m² | **Rooms:** {clean_rooms}")
                        st.markdown(f"[🔗 View Listing]({row['url']})")
                        st.markdown("---")
                else:
                    st.error("Could not find properties matching this minimum room requirement.")
            except Exception as e:
                st.error(f"Error running recommender: {e}")
                st.info("💡 Ensure scikit-learn is installed: pip install scikit-learn")