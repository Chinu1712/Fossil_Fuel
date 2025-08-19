import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
USE_IMAGES = False

# ---------- Styles ----------
st.set_page_config(page_title="CO2 Emission Predictor & EV Awareness", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    body, .stApp { background-color: #0e1117; color: #ffffff; }
    .main-header { font-size: 3rem; color: #2E8B57; text-align: center; margin-bottom: 2rem; }
    .sub-header  { font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem; }
    .metric-card, .prediction-result, .highlight-box { background-color: #1e2130; color: #ffffff !important; }
    .highlight-box { border-left: 8px solid #2E8B57; padding: 1.5rem; border-radius: 0.5rem; margin: 2rem 0; }
    .impact-point { display: flex; align-items: center; margin-bottom: 1rem; }
    .impact-point .icon { font-size: 2rem; margin-right: 1rem; color: #2E8B57; }
    .stButton>button { background-color: #4CAF50; color: white; font-size: 1.2rem; padding: 0.8rem 1.5rem;
                       border-radius: 0.5rem; border: none; cursor: pointer; }
    .stButton>button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

# ---------- Loaders ----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(BASE_DIR / "co2_model.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        features = joblib.load(BASE_DIR / "model_features.pkl")
        return model, scaler, features
    except Exception:
        return None, None, None

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv(BASE_DIR / "owid-co2-data.csv")
    except Exception:
        return None

# ---------- Landing Page ----------
def show_landing_page():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness Platform</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.3rem;'>Leveraging Data Science to Combat Climate Change 🌱</p>", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1532187863570-877237497150?q=80&w=2070", use_column_width=True)

    st.markdown('<h2 class="sub-header">✨ Self-Explanatory Project Overview</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
        <div class="impact-point"><span class="icon">📈</span><b>Accurate CO2 Emission Prediction</b></div>
        <div class="impact-point"><span class="icon">🚗</span><b>Promoting Electric Vehicle Adoption</b></div>
        <div class="impact-point"><span class="icon">🌱</span><b>Spreading Environmental Awareness</b></div>
        <div class="impact-point"><span class="icon">💡</span><b>Inspiring Curiosity and Action</b></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Go to Main App"):
        st.session_state["page"] = "main"

# ---------- Main App ----------
def show_main_app():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 CO2 Predictor", "🚗 EV Benefits", "🌱 Environmental Impact", "📊 EV Statistics"])
    with tab1: show_co2_predictor()
    with tab2: show_ev_benefits()
    with tab3: show_environmental_impact()
    with tab4: show_ev_statistics()

    if st.button("⬅️ Back to Landing Page"):
        st.session_state["page"] = "landing"

# ---------- Tabs ----------
def show_co2_predictor():
    model, scaler, features = load_model()
    sample_data = load_sample_data()
    if model is None: 
        st.error("⚠️ Model not loaded.")
        return

    population = st.sidebar.number_input("Population", value=330000000)
    gdp = st.sidebar.number_input("GDP (billion USD)", value=25000.0)
    energy_per_capita = st.sidebar.number_input("Energy per Capita (kWh)", value=12000.0)
    primary_energy_consumption = st.sidebar.number_input("Primary Energy Consumption (TWh)", value=2500.0)
    year = st.sidebar.slider("Year", 1990, 2030, 2023)

    if st.sidebar.button("🔮 Predict CO2 Emissions"):
        x = pd.DataFrame([[year, population, gdp*1e9, energy_per_capita, primary_energy_consumption]], 
                         columns=["year","population","gdp","energy_per_capita","primary_energy_consumption"])
        x_scaled = scaler.transform(x)
        pred = float(model.predict(x_scaled)[0])
        st.success(f"🌍 Predicted CO2 Emissions: {pred:.2f} Mt")

def show_ev_benefits():
    st.markdown('<h2 class="sub-header">💰 Cost Comparison: EV vs Gasoline Car</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 30000, 12000, 1000)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 6.0, 3.50, 0.10)
        car_mpg = st.number_input("Your car\'s MPG", 15, 40, 25, 1)
        electricity_rate = st.number_input("Electricity rate (¢/kWh)", 8, 30, 12, 1)
    with c2:
        annual_gas_cost = (annual_miles / car_mpg) * gas_price
        annual_electricity_cost = (annual_miles / 3.5) * (electricity_rate / 100)
        annual_savings = annual_gas_cost - annual_electricity_cost
        st.markdown(f"""
        <div class="prediction-result">
          <h4>💰 Your Annual Savings</h4>
          <p><strong>${annual_savings:.0f}</strong> per year</p>
          <p><strong>${annual_savings*5:.0f}</strong> over 5 years</p>
          <p>Gas: ${annual_gas_cost:.0f} | Electric: ${annual_electricity_cost:.0f}</p>
        </div>
        """, unsafe_allow_html=True)

def show_environmental_impact():
    st.markdown('<h2 class="sub-header">🌱 Environmental Impact of EVs</h2>', unsafe_allow_html=True)
    st.info("🚗 Transportation is a major contributor to CO2. EVs reduce tailpipe emissions and can leverage renewables.")
    st.success("✅ Key Benefits: Zero tailpipe emissions, reduced urban pollution, lower lifecycle carbon footprint.")

def show_ev_statistics():
    years = list(range(2015, 2024))
    global_ev_sales = [0.4, 0.7, 1.2, 2.0, 2.2, 3.1, 6.6, 10.5, 14.1]
    fig = px.line(x=years, y=global_ev_sales, title="Global Electric Vehicle Sales (2015-2023)",
                  labels={'x': 'Year', 'y': 'Sales (Millions)'}, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Entrypoint ----------
if "page" not in st.session_state: st.session_state["page"] = "landing"
if st.session_state["page"] == "landing": show_landing_page()
else: show_main_app()
