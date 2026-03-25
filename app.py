import streamlit as st
import pandas as pd
import plotly.express as px
from recommender import get_recommendations

# 1. إعدادات الصفحة الأساسية
st.set_page_config(page_title="Aqarmap Market Intelligence", page_icon="🏢", layout="wide")

# ==========================================
# 💎 FUTURISTIC & PROFESSIONAL UI INJECTION
# ==========================================
# Streamlit allows direct HTML/CSS mapping. We use dark gradients, "Glassmorphism" for the KPIs,
# and an elegant Arabic font 'Cairo' for a premium hyper-modern dashboard feel.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif !important;
    }
    
    /* Futuristic Space-Dark gradient for the main application */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 20, 29) 0%, rgb(24, 26, 40) 90%);
        color: #E2E8F0;
    }
    
    /* Neon Glow effect for all main Headings */
    h1, h2, h3 {
        color: #38bdf8 !important;
        text-shadow: 0px 4px 15px rgba(56, 189, 248, 0.4);
    }

    /* Redesign the KPI Metrics into interactive Glass Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    }
    
    /* Cyberpunk hover animation for the boxes */
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
    
    /* Sleek Sidebar Design */
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
st.markdown("تحليل ذكي ومتقدم لأسعار العقارات باستخدام مستخلصات البيانات وتكنولوجيا الـ AI.")

# 2. تحميل الداتا (استخدام Cache عشان الداشبورد تفضل سريعة ومتحملش الداتا كل شوية)
@st.cache_data
def load_data():
    # تأكد إن اسم الملف نفس الاسم اللي طلع من الـ cleaner
    df = pd.read_csv("cleaned_properties.csv")
    
    # تنظيف سريع للإحداثيات عشان الخريطة متضربش إيرور واستبعاد النقاط في البحر المتوسط أو خارج مصر
    df = df.dropna(subset=['latitude', 'longitude'])
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    # مصر تقع تقريباً بين خطي عرض (22 -> 31.5) وطول (24 -> 37)
    df = df[(df['latitude'] >= 22.0) & (df['latitude'] <= 31.6)]
    df = df[(df['longitude'] >= 24.0) & (df['longitude'] <= 37.0)]
    
    # استخراج نوع العقار واسم المنطقة من الرابط باستخدام Regex لمعالجة (apartment, villa, chalet)
    import re
    def extract_info(url):
        if not isinstance(url, str): return 'Unknown', 'Unknown'
        # Pattern looks like: ...-for-sale-(apartment|villa|chalet)-(cairo-new-cairo)
        match = re.search(r'for-sale-([a-z0-9-]+?)-([a-z0-9-]+.*)', url)
        if match:
            prop_type = match.group(1).title() 
            location_raw = match.group(2).strip('/').replace('-', ' ').title()
            return prop_type, location_raw
        return 'Unknown', 'Unknown'
        
    df[['property_type', 'district']] = df['url'].apply(lambda u: pd.Series(extract_info(u)))
    
    # تحضير سعر المتر بشكل جذري.
    # نتجاهل الأرقام النصية المعطوبة مثل "23,111 EGP/M²" القادمة من السكريبتر المباشر
    # ونعيد حسابها بدقة باستخدام الأعمدة الرقمية الأنقى لدينا.
    df['price_per_meter'] = pd.to_numeric(df['unified_price'] / df['unified_area'], errors='coerce')
    # إزالة أي قيم فارغة نتجت عن القسمة الرياضية
    df = df.dropna(subset=['price_per_meter'])
    
    return df

df = load_data()

# 3. القائمة الجانبية (Sidebar) للـ Filters
st.sidebar.header("🔍 الفلاتر (Filters)")

# فلتر السعر (البادجت)
min_price = int(df['unified_price'].min())
max_price = int(df['unified_price'].quantile(0.99)) # To avoid super outliers stretching the slider
selected_price_range = st.sidebar.slider(
    "حدد ميزانيتك (EGP):",
    min_value=min_price,
    max_value=int(max_price),
    value=(min_price, int(max_price / 2)), # قيمة افتراضية
    step=250000
)

# فلتر نوع العقار
prop_options = sorted([p for p in df['property_type'].unique() if str(p) != 'Unknown'])
selected_props = st.sidebar.multiselect("أنواع العقارات:", options=prop_options, default=prop_options)

# فلتر المدن/المناطق (District)
districts_opt = sorted([d for d in df['district'].unique() if str(d) != 'Unknown'])
selected_districts = st.sidebar.multiselect("المنطقة (District):", options=districts_opt, default=[])

# فلتر عدد الغرف
rooms_options = sorted(df['unified_rooms'].dropna().unique().tolist())
selected_rooms = st.sidebar.multiselect("عدد الغرف:", options=rooms_options, default=rooms_options[:3])

# Handle empty multi-selects by essentially assuming "ALL"
active_districts = selected_districts if selected_districts else districts_opt
active_props = selected_props if selected_props else prop_options

# 4. تطبيق الفلاتر على الداتا
filtered_df = df[
    (df['unified_price'] >= selected_price_range[0]) &
    (df['unified_price'] <= selected_price_range[1]) &
    (df['unified_rooms'].isin(selected_rooms)) &
    (df['district'].isin(active_districts)) &
    (df['property_type'].isin(active_props))
]

# 5. عرض إحصائيات سريعة (KPIs)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("إجمالي الشقق", f"{len(filtered_df):,}")
with col2:
    st.metric("متوسط السعر", f"{filtered_df['unified_price'].mean():,.0f} EGP")
with col3:
    st.metric("السعر الوسيط", f"{filtered_df['unified_price'].median():,.0f} EGP")
with col4:
    st.metric("متوسط سعر المتر", f"{filtered_df['price_per_meter'].mean():,.0f} EGP/m²")
with col5:
    st.metric("متوسط المساحة", f"{filtered_df['unified_area'].mean():,.0f} m²")

st.markdown("---")

# 6. إضافة التبويبات (Tabs) لفصل التحليل عن محرك التوصيات
tab1, tab2 = st.tabs(["📊 Market Analytics (تحليل السوق)", "🤖 Smart Recommender (محرك التوصيات الذكي)"])

with tab1:
    st.subheader("📍 خريطة توزيع العقارات والأسعار")
    
    if not filtered_df.empty:
        # استخدام Plotly لعمل Scatter Map
        fig = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="unified_price", # لون النقطة يتغير حسب السعر
            size="unified_area",   # حجم النقطة يتغير حسب المساحة
            hover_name="title",
            hover_data={"latitude": False, "longitude": False, "unified_price": True, "unified_rooms": True},
            color_continuous_scale=px.colors.sequential.Plasma,
            size_max=15,
            zoom=10
            #mapbox_style="carto-positron" # شكل الخريطة
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("⚠️ لا توجد عقارات تطابق هذه الفلاتر. جرب تغيير الميزانية أو عدد الغرف.")

    st.markdown("---")
    
    # --- Bar Chart: Average Price by District ---
    st.subheader("📊 تحليل سعر المتر حسب المنطقة (Bar Chart)")
    if not filtered_df.empty:
        # Group by district and calculate mean price per metere
        district_price = filtered_df.groupby('district')['price_per_meter'].mean().reset_index()
        district_price = district_price.sort_values(by='price_per_meter', ascending=False)
        
        # Display top 30 districts to avoid massive clutter
        fig_bar = px.bar(
            district_price.head(30), 
            x='district', 
            y='price_per_meter', 
            title="أغلى 30 منطقة من حيث متوسط سعر المتر",
            labels={'district': 'المنطقة', 'price_per_meter': 'سعر المتر (EGP)'},
            color='price_per_meter', 
            color_continuous_scale='Viridis',
            template='plotly_dark' # دمج مع الثيم الحديث
        )
        # إضفاء ستايل حديث على خلفية الرسم نفسه
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, width="stretch")
    
    st.markdown("---")
    
    # --- Box Plot: Feature Impact on Price ---
    st.subheader("📦 تأثير المميزات على سعر العقار (Box Plot)")
    st.markdown("يعرض هذا الرسم الإحصائي كيف يؤثر وجود ميزة معينة (مثل الجراج أو الأسانسير) على السعر الإجمالي.")
    
    if not filtered_df.empty:
        # Create a selectbox so user can toggle features dynamically
        available_features = [col for col in filtered_df.columns if col.startswith('has_')]
        
        if available_features:
            feature_to_check = st.selectbox("اختر الميزة لمقارنة الأسعار:", available_features, format_func=lambda x: x.replace('has_', 'يوجد '))
            
            box_df = filtered_df.copy()
            # Map robustly to prevent empty labels
            box_df[feature_to_check] = box_df[feature_to_check].map({1: 'يوجد (Yes)', 0: 'لا يوجد (No)'}).fillna('غير محدد')
            
            # To make box plots readable, remove extreme statistical outliers from visualization (e.g., upper 1%)
            q_high = box_df['unified_price'].quantile(0.95)
            box_df_filtered = box_df[box_df['unified_price'] <= q_high]
            
            fig_box = px.box(
                box_df_filtered, 
                x=feature_to_check, 
                y='unified_price', 
                color=feature_to_check,
                title=f"التأثير الإحصائي لميزة ({feature_to_check.replace('has_', '')}) للمنازل تحت {q_high:,.0f} EGP",
                labels={feature_to_check: 'التوفر', 'unified_price': 'السعر الإجمالي (EGP)'},
                template='plotly_dark' # دمج مع الثيم الحديث
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, width="stretch")
        else:
            st.info("لا توجد بيانات مميزات (Amenities) مسجلة في هذا الملف.")

with tab2:
    st.subheader("🤖 ابحث عن شقة أحلامك بالذكاء الاصطناعي (KNN Algoritm)")
    st.markdown("يقوم هذا المحرك بحساب أقرب الشقق لطلبك مع موازنة فرق السعر والمسافة الجغرافية في نفس الوقت باستخدام `MinMaxScaler` وتوزيع الأوزان.")
    
    # User Inputs for ML Recommender
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        target_price = st.number_input("الميزانية المستهدفة (EGP):", min_value=100000, value=int(selected_price_range[1]), step=50000)
        target_rooms = st.number_input("الحد الأدنى للغرف:", min_value=1, value=3)
        
    with rec_col2:
        # Precompute mean coordinates for each district accurately
        district_coords = df.groupby('district')[['latitude', 'longitude']].mean().reset_index()
        valid_districts = sorted([d for d in district_coords['district'] if d != 'Unknown'])
        
        target_district = st.selectbox("المنطقة المفضلة:", valid_districts)
        
        # Get coordinates dynamically from the chosen district
        dist_data = district_coords[district_coords['district'] == target_district]
        if not dist_data.empty:
            target_lat = float(dist_data.iloc[0]['latitude'])
            target_lon = float(dist_data.iloc[0]['longitude'])
        else:
            target_lat, target_lon = 30.01, 31.42
        
    with rec_col3:
        st.write("أوزان خوارزمية البحث (Weights):")
        w_loc = st.slider("أهمية الموقع (%)", 0, 100, 40) / 100.0
        w_price = st.slider("أهمية السعر (%)", 0, 100, 40) / 100.0
        w_area = st.slider("أهمية المساحة والمميزات (%)", 0, 100, 20) / 100.0
        
    if st.button("🔍 البحث عن أفضل 5 مطابقات", type="primary"):
        with st.spinner('جاري حساب المسافات بالذكاء الاصطناعي...'):
            try:
                # Run the recommender function
                recs = get_recommendations(df, target_lat, target_lon, target_price, min_rooms=target_rooms, 
                                           weight_loc=w_loc, weight_price=w_price, weight_area=w_area, weight_amenities=0.0)
                
                if not recs.empty:
                    st.success("🎉 تم العثور على أفضل المطابقات!")
                    for i, (_, row) in enumerate(recs.iterrows()):
                        st.markdown(f"### #{i+1} - {row['title']}")
                        st.markdown(f"**السعر:** {row['unified_price']:,.0f} EGP | **المساحة:** {row['unified_area']} m² | **الغرف:** {row['unified_rooms']}")
                        st.markdown(f"[🔗 رابط الشقة على الموقع]({row['url']})")
                        st.markdown("---")
                else:
                    st.error("لم نتمكن من العثور على شقق تطابق هذا الحد الأدنى من الغرف.")
            except Exception as e:
                st.error(f"حدث خطأ أثناء تشغيل خوارزمية التوصية: {e}")
                st.info("💡 تأكد من تثبيت مكتبة scikit-learn باستخدام: pip install scikit-learn")