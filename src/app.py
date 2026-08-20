import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PriceView · Vehicle Price Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CLASSIC & MINIMALIST DARK STYLING
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Dark Theme */
    .stApp {
        background-color: #09090b;
        color: #f4f4f5;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 960px;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Brand Header */
    .brand-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        padding-bottom: 14px;
        border-bottom: 1px solid #27272a;
        margin-bottom: 24px;
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .brand-sub {
        font-size: 13px;
        color: #71717a;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }

    .section-desc {
        font-size: 13px;
        color: #a1a1aa;
        margin-bottom: 18px;
    }

    label {
        color: #e4e4e7 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* Input Elements */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #52525b !important;
    }

    /* Default Buttons */
    .stButton > button {
        width: 100%;
        background-color: #18181b !important;
        color: #f4f4f5 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #27272a !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #27272a !important;
        color: #ffffff !important;
        border-color: #52525b !important;
    }

    /* Primary Action Button */
    .main-submit-btn > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        border: none !important;
    }

    .main-submit-btn > button:hover {
        background-color: #e4e4e7 !important;
        color: #000000 !important;
    }

    /* Info Cards */
    .info-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 18px 20px;
    }

    .info-title {
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .info-desc {
        font-size: 12px;
        color: #a1a1aa;
        line-height: 1.5;
    }

    .footer-text {
        text-align: center;
        color: #52525b;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #27272a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS & MODEL LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    """Load preprocessors and models."""
    reg_prep = joblib.load(os.path.join(BASE_DIR, "models", "regression", "preprocessor.pkl"))
    reg_model = joblib.load(os.path.join(BASE_DIR, "models", "regression", "best_regression_model.pkl"))
    cls_prep = joblib.load(os.path.join(BASE_DIR, "models", "classification", "preprocessor.pkl"))
    cls_model = joblib.load(os.path.join(BASE_DIR, "models", "classification", "best_classification_model.pkl"))
    return reg_prep, reg_model, cls_prep, cls_model

@st.cache_data
def load_dataset():
    """Load reference dataset."""
    data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception:
            return None
    return None

try:
    regression_preprocessor, regression_model, classification_preprocessor, classification_model = load_models()
    dataset = load_dataset()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def format_inr(amount):
    if amount >= 10000000:
        return f"₹ {amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹ {amount / 100000:.2f} Lakh"
    else:
        return f"₹ {amount:,.0f}"

def format_brand_name(b):
    b_str = str(b).strip()
    if b_str.lower() == 'bmw':
        return 'BMW'
    elif b_str.lower() == 'mg':
        return 'MG'
    return b_str.title()

if dataset is not None:
    all_brands = sorted([format_brand_name(b) for b in dataset['brand'].dropna().unique()])
    model_by_brand = {}
    for b in dataset['brand'].dropna().unique():
        sub = dataset[dataset['brand'] == b]
        brand_fmt = format_brand_name(b)
        model_by_brand[brand_fmt] = sorted([m.title() for m in sub['model'].dropna().unique()])
else:
    all_brands = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Mahindra", "Tata", "BMW", "Mercedes-Benz", "Audi"]
    model_by_brand = {b: ["Standard Model"] for b in all_brands}

# ============================================================
# STATE MANAGEMENT & CALLBACKS
# ============================================================

defaults = {
    "form_brand": "Maruti",
    "form_model": "Alto",
    "form_age": 5,
    "form_km": 45000,
    "form_fuel": "Petrol",
    "form_trans": "Manual",
    "form_seller": "Individual",
    "form_mileage": 19.5,
    "form_engine": 1197.0,
    "form_power": 82.0,
    "form_seats": 5
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(brand, model, age, km, fuel, trans, seller, mileage, engine, power, seats):
    """Callback to set preset values with case-insensitive matching."""
    matched_brand = brand
    for b in all_brands:
        if b.lower() == brand.lower():
            matched_brand = b
            break
            
    avail = model_by_brand.get(matched_brand, ["Standard Model"])
    matched_model = avail[0]
    for m in avail:
        if m.lower() == model.lower() or (len(model) > 1 and model.lower() in m.lower()) or (len(m) > 1 and m.lower() in model.lower()):
            matched_model = m
            break

    st.session_state["form_brand"] = matched_brand
    st.session_state["form_model"] = matched_model
    st.session_state["form_age"] = age
    st.session_state["form_km"] = km
    st.session_state["form_fuel"] = fuel
    st.session_state["form_trans"] = trans
    st.session_state["form_seller"] = seller
    st.session_state["form_mileage"] = mileage
    st.session_state["form_engine"] = engine
    st.session_state["form_power"] = power
    st.session_state["form_seats"] = seats

def reset_inputs():
    for k, v in defaults.items():
        st.session_state[k] = v

# ============================================================
# NAVBAR HEADER
# ============================================================

st.markdown("""
<div class="brand-header">
    <div class="brand-name">PriceView</div>
    <div class="brand-sub">Used vehicle price estimation</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# QUICK PRESET SHORTCUTS
# ============================================================

st.markdown('<div class="section-title">Vehicle Specifications</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Select specs or choose a quick preset.</div>', unsafe_allow_html=True)

preset_cols = st.columns(5)

with preset_cols[0]:
    st.button("Maruti Swift", key="preset_swift", on_click=apply_preset, args=("Maruti", "Swift", 4, 35000, "Petrol", "Manual", "Individual", 21.21, 1197.0, 81.8, 5))
with preset_cols[1]:
    st.button("Hyundai Creta", key="preset_creta", on_click=apply_preset, args=("Hyundai", "Creta", 3, 28000, "Diesel", "Automatic", "Dealer", 18.0, 1493.0, 113.4, 5))
with preset_cols[2]:
    st.button("Honda City", key="preset_city", on_click=apply_preset, args=("Honda", "City", 5, 45000, "Petrol", "Manual", "Individual", 17.8, 1497.0, 117.3, 5))
with preset_cols[3]:
    st.button("Tata Nexon", key="preset_nexon", on_click=apply_preset, args=("Tata", "Nexon", 2, 18000, "Petrol", "Manual", "Individual", 17.5, 1199.0, 118.3, 5))
with preset_cols[4]:
    st.button("BMW 3 Series", key="preset_bmw", on_click=apply_preset, args=("BMW", "3", 5, 42000, "Diesel", "Automatic", "Dealer", 19.6, 1995.0, 187.4, 5))

# ============================================================
# INPUT FORM
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    selected_brand = st.selectbox("Brand", all_brands, key="form_brand")
    avail_models = model_by_brand.get(selected_brand, ["Alto", "Swift", "Baleno"])
    if st.session_state.get("form_model") not in avail_models:
        st.session_state["form_model"] = avail_models[0]
    selected_model = st.selectbox("Model", avail_models, key="form_model")
    car_name_val = f"{selected_brand.lower()} {selected_model.lower()}"
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=40, key="form_age")

with col2:
    km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=500000, step=2000, key="form_km")
    fuel_options = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    fuel_type = st.selectbox("Fuel Type", fuel_options, key="form_fuel")
    trans_options = ["Manual", "Automatic"]
    transmission_type = st.selectbox("Transmission", trans_options, key="form_trans")

with col3:
    seller_options = ["Individual", "Dealer", "Trustmark Dealer"]
    seller_type = st.selectbox("Seller Type", seller_options, key="form_seller")
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=60.0, step=0.5, key="form_mileage")
        engine = st.number_input("Engine (CC)", min_value=50.0, max_value=6000.0, step=50.0, key="form_engine")
    with sub_c2:
        max_power = st.number_input("Max Power (BHP)", min_value=10.0, max_value=800.0, step=5.0, key="form_power")
        seats = st.number_input("Seats", min_value=2, max_value=14, key="form_seats")

# Action Buttons
btn_col1, btn_col2 = st.columns([3, 1])

with btn_col1:
    st.markdown('<div class="main-submit-btn">', unsafe_allow_html=True)
    predict_btn = st.button("Estimate Selling Price", key="estimate_action")
    st.markdown('</div>', unsafe_allow_html=True)

with btn_col2:
    st.button("Clear Inputs", key="clear_action", on_click=reset_inputs)

# ============================================================
# PREDICTION RESULT DISPLAY (PREMIUM MINIMALIST)
# ============================================================

if predict_btn:
    input_df = pd.DataFrame([{
        "car_name": car_name_val.lower(),
        "brand": selected_brand.lower(),
        "model": selected_model.lower(),
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type.lower(),
        "fuel_type": fuel_type.lower(),
        "transmission_type": transmission_type.lower(),
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }])

    try:
        reg_trans = regression_preprocessor.transform(input_df)
        predicted_price = float(regression_model.predict(reg_trans)[0])

        cls_trans = classification_preprocessor.transform(input_df)
        predicted_category = str(classification_model.predict(cls_trans)[0])

        lower_bound = max(10000, predicted_price * 0.925)
        upper_bound = predicted_price * 1.075

        badge_style = "background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);"
        if "Mid" in predicted_category or "mid" in predicted_category:
            badge_style = "background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3);"
        elif "Lux" in predicted_category or "lux" in predicted_category or "Premium" in predicted_category:
            badge_style = "background: rgba(217, 70, 239, 0.15); color: #f0abfc; border: 1px solid rgba(217, 70, 239, 0.3);"

        # Main Price Card
        st.markdown(f'''
        <div style="background:#121215; border:1px solid #27272a; border-radius:14px; padding:28px 32px; margin-top:24px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px;">
                <div>
                    <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#888888; margin-bottom:6px;">ESTIMATED FAIR MARKET VALUE</div>
                    <div style="font-size:46px; font-weight:800; color:#ffffff; letter-spacing:-1.5px; line-height:1.05;">₹ {predicted_price:,.0f}</div>
                    <div style="font-size:17px; font-weight:600; color:#34d399; margin-top:6px; margin-bottom:10px;">Approximately {format_inr(predicted_price)}</div>
                    <div style="font-size:13px; color:#999999;">Estimated Range: <b style="color:#ffffff;">{format_inr(lower_bound)}</b> – <b style="color:#ffffff;">{format_inr(upper_bound)}</b></div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#888888; margin-bottom:8px;">PRICE CATEGORY</div>
                    <span style="display:inline-block; padding:7px 18px; border-radius:20px; font-size:13px; font-weight:700; {badge_style}">
                        🏷️ {predicted_category.upper()} TIER
                    </span>
                    <div style="font-size:12px; color:#777777; margin-top:10px; max-width:240px;">
                        {vehicle_age}-yr-old {selected_brand} {selected_model}<br>({km_driven:,} km · {fuel_type} · {transmission_type})
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # 3 Key Metric Highlights
        km_per_year = km_driven / max(1, vehicle_age)
        usage_label = "Low Usage ✨" if km_per_year < 10000 else ("Moderate 🚘" if km_per_year < 20000 else "High Usage ⚠️")
        power_ratio = max_power / (engine / 1000.0) if engine > 0 else 0

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f'''
            <div style="background:#141417; border:1px solid #27272a; border-radius:10px; padding:14px 18px; text-align:center;">
                <div style="font-size:11px; font-weight:700; color:#888888; text-transform:uppercase;">ANNUAL RUNNING RATE</div>
                <div style="font-size:20px; font-weight:800; color:#ffffff; margin-top:4px;">{km_per_year:,.0f} km/yr</div>
                <div style="font-size:11px; color:#a1a1aa; margin-top:2px;">{usage_label}</div>
            </div>
            ''', unsafe_allow_html=True)

        with m2:
            st.markdown(f'''
            <div style="background:#141417; border:1px solid #27272a; border-radius:10px; padding:14px 18px; text-align:center;">
                <div style="font-size:11px; font-weight:700; color:#888888; text-transform:uppercase;">VEHICLE AGE FACTOR</div>
                <div style="font-size:20px; font-weight:800; color:#ffffff; margin-top:4px;">{vehicle_age} Years</div>
                <div style="font-size:11px; color:#a1a1aa; margin-top:2px;">Model Year ~{2026 - vehicle_age}</div>
            </div>
            ''', unsafe_allow_html=True)

        with m3:
            st.markdown(f'''
            <div style="background:#141417; border:1px solid #27272a; border-radius:10px; padding:14px 18px; text-align:center;">
                <div style="font-size:11px; font-weight:700; color:#888888; text-transform:uppercase;">ENGINE SPECIFIC OUTPUT</div>
                <div style="font-size:20px; font-weight:800; color:#ffffff; margin-top:4px;">{power_ratio:.1f} BHP/L</div>
                <div style="font-size:11px; color:#a1a1aa; margin-top:2px;">{engine:.0f} CC · {max_power:.0f} BHP</div>
            </div>
            ''', unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Error generating valuation: {err}")

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown('<div class="section-title" style="margin-top: 36px;">How it works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Three steps to your vehicle valuation.</div>', unsafe_allow_html=True)

info_cols = st.columns(3)

with info_cols[0]:
    st.markdown('<div class="info-card"><div class="info-title">01 · Input Specs</div><div class="info-desc">Provide vehicle details including age, km driven, engine CC, and fuel type.</div></div>', unsafe_allow_html=True)

with info_cols[1]:
    st.markdown('<div class="info-card"><div class="info-title">02 · ML Prediction</div><div class="info-desc">Trained regression and classification models process the specifications.</div></div>', unsafe_allow_html=True)

with info_cols[2]:
    st.markdown('<div class="info-card"><div class="info-title">03 · Valuation Result</div><div class="info-desc">Receive an estimated resale price range and market tier classification.</div></div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown('<div class="footer-text">PriceView · Vehicle Valuation Engine</div>', unsafe_allow_html=True)