import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ---------- Paths & toggles ----------
BASE_DIR = Path(__file__).resolve().parent
USE_IMAGES = False  # set True only after you add images under search_images/

def load_image_safe(rel_path: str):
    """Return PIL image if exists and USE_IMAGES=True, else None."""
    if not USE_IMAGES:
        return None
    try:
        p = BASE_DIR / rel_path
        if p.exists():
            return Image.open(p)
    except Exception:
        pass
    return None

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="CO2 Emission Predictor & EV Awareness",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styles ----------
st.markdown("""
<style>
    /* Headings */
    .main-header { font-size: 3rem; color: #2E8B57; text-align: center; margin-bottom: 2rem; }
    .sub-header  { font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem; }

    /* Light cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }

    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }

    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }
    .info-box    { background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }

    /* Gradient cards that should keep white text */
    .ev-gradient, .env-gradient { color: #ffffff !important; }
    .ev-gradient { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .env-gradient { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }

    /* --- FIX: readable text on light cards in dark theme --- */
    .metric-card, .metric-card *,
    .info-box, .info-box *,
    .fact-highlight, .fact-highlight * {
        color: #0f172a !important;  /* dark slate text */
    }

    .fact-highlight {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem; margin: 1rem 0; border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Loaders ----------
@st.cache_resource
def load_model():
    """Load ML artifacts."""
    try:
        model = joblib.load(BASE_DIR / "co2_model.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        features = joblib.load(BASE_DIR / "model_features.pkl")
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_sample_data():
    """Load sample/reference data for countries & charts."""
    try:
        df = pd.read_csv(BASE_DIR / "owid-co2-data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# ---------- Main ----------
def main():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:1.2rem;color:#ccc;">Predict CO2 emissions and discover the power of electric vehicles</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 CO2 Predictor", "🚗 EV Benefits", "🌱 Environmental Impact", "📊 EV Statistics"])
    with tab1: show_co2_predictor()
    with tab2: show_ev_benefits()
    with tab3: show_environmental_impact()
    with tab4: show_ev_statistics()

# ---------- Tab: Predictor ----------
def show_co2_predictor():
    model, scaler, features = load_model()
    sample_data = load_sample_data()

    if model is None or scaler is None or features is None:
        st.error("Failed to load the prediction model. Please check model/scaler/features files.")
        return

    # Sidebar inputs
    st.sidebar.markdown('<h2 class="sub-header">🔧 Input Parameters</h2>', unsafe_allow_html=True)

    if sample_data is not None and "country" in sample_data.columns:
        countries = sorted(sample_data['country'].dropna().unique().tolist())
        default_idx = countries.index('United States') if 'United States' in countries else 0
        selected_country = st.sidebar.selectbox("Select Country", countries, index=default_idx)
    else:
        selected_country = st.sidebar.text_input("Country", value="United States")

    year = st.sidebar.slider("Year", 1990, 2030, 2023, 1)

    st.sidebar.markdown("### 💰 Economic Indicators")
    population = st.sidebar.number_input("Population", min_value=1000, max_value=2_000_000_000, value=330_000_000, step=1_000_000)
    gdp = st.sidebar.number_input("GDP (billion USD)", min_value=0.1, max_value=30_000.0, value=25_000.0, step=100.0)

    st.sidebar.markdown("### ⚡ Energy Consumption")
    energy_per_capita = st.sidebar.number_input("Energy per Capita (kWh)", 0.0, 100_000.0, 12_000.0, 100.0)
    primary_energy_consumption = st.sidebar.number_input("Primary Energy Consumption (TWh)", 0.0, 10_000.0, 2_500.0, 10.0)

    st.sidebar.markdown("### 🏭 CO2 Sources")
    cement_co2 = st.sidebar.number_input("Cement CO2 (Mt)", 0.0, 1_000.0, 50.0, 1.0)
    coal_co2   = st.sidebar.number_input("Coal CO2 (Mt)",   0.0, 5_000.0, 1_200.0, 10.0)
    oil_co2    = st.sidebar.number_input("Oil CO2 (Mt)",    0.0, 3_000.0, 800.0, 10.0)
    gas_co2    = st.sidebar.number_input("Gas CO2 (Mt)",    0.0, 3_000.0, 600.0, 10.0)
    flaring_co2= st.sidebar.number_input("Flaring CO2 (Mt)",0.0, 200.0,   10.0, 1.0)

    st.sidebar.markdown("### 🌡️ Other Greenhouse Gases")
    methane       = st.sidebar.number_input("Methane (Mt CO2eq)", 0.0, 2_000.0, 300.0, 10.0)
    nitrous_oxide = st.sidebar.number_input("Nitrous Oxide (Mt CO2eq)", 0.0, 500.0, 100.0, 5.0)

    if st.sidebar.button("🔮 Predict CO2 Emissions", type="primary"):
        input_df = prepare_input_data(
            selected_country, year, population, gdp, energy_per_capita,
            primary_energy_consumption, cement_co2, coal_co2, oil_co2,
            gas_co2, flaring_co2, methane, nitrous_oxide, features
        )
        if input_df is not None:
            try:
                input_scaled = scaler.transform(input_df)
                prediction = float(model.predict(input_scaled)[0])
                display_prediction_results(prediction, selected_country, year, population)
                display_charts(prediction, cement_co2, coal_co2, oil_co2, gas_co2, flaring_co2)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Info + chart
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="sub-header">📊 About CO2 Emissions</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>🌍 Understanding CO2 Emissions</h4>
          <p>CO2 emissions are a key driver of climate change. This model predicts CO2 based on economic, energy, and industrial factors.</p>
          <h4>🔍 Key Factors</h4>
          <ul>
            <li><strong>Economic:</strong> GDP and population</li>
            <li><strong>Energy:</strong> Total energy use and per-capita consumption</li>
            <li><strong>Industry:</strong> Cement, coal, oil, gas, and flaring emissions</li>
            <li><strong>Other GHGs:</strong> Methane and nitrous oxide</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        if sample_data is not None and {"year","country","co2"}.issubset(sample_data.columns):
            st.markdown('<h3 class="sub-header">🌐 Global CO2 Emissions Overview</h3>', unsafe_allow_html=True)
            recent = (sample_data[sample_data["year"] >= 2020]
                      .groupby("country", as_index=True)["co2"]
                      .mean().sort_values(ascending=False).head(10))
            fig = px.bar(x=recent.values, y=recent.index, orientation='h',
                         title="Top 10 Countries by Average CO2 Emissions (2020+)",
                         labels={'x':'CO2 Emissions (Mt)','y':'Country'},
                         color=recent.values, color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<h2 class="sub-header">🚗 Why Choose Electric Vehicles?</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
          <h4>🌱 Environmental Benefits</h4>
          <ul>
            <li>Zero direct emissions</li>
            <li>Reduced air pollution</li>
            <li>Lower carbon footprint</li>
          </ul>
        </div>
        <div class="metric-card">
          <h4>💰 Economic Benefits</h4>
          <ul>
            <li>Lower fuel costs</li>
            <li>Reduced maintenance</li>
            <li>Potential incentives</li>
          </ul>
        </div>
        <div class="metric-card">
          <h4>⚡ Performance Benefits</h4>
          <ul>
            <li>Instant torque</li>
            <li>Quiet operation</li>
            <li>Advanced technology</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------- Tab: EV Benefits ----------
def show_ev_benefits():
    st.markdown('<h1 class="main-header">🚗 Electric Vehicle Benefits</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # optional image
        img = load_image_safe("search_images/JAAPlF95yGTw.jpg")
        if img: st.image(img, caption="Top Benefits of Electric Cars", use_column_width=True)

        st.markdown("""
        <div class="ev-gradient" style="padding:1.5rem;border-radius:1rem;margin:1rem 0;">
          <h3>🌟 Key EV Advantages</h3>
          <ul>
            <li><strong>Environmental:</strong> Zero tailpipe emissions</li>
            <li><strong>Lower Operating Costs:</strong> Electricity often cheaper than gasoline</li>
            <li><strong>Reduced Maintenance:</strong> Fewer moving parts</li>
            <li><strong>Performance:</strong> Instant torque, smooth acceleration</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        img2 = load_image_safe("search_images/QD5PXgCvPHNS.jpg")
        if img2: st.image(img2, caption="Life Cycle Emissions: Gasoline vs Electric", use_column_width=True)

        st.markdown("""
        <div class="fact-highlight">
          <h4>💡 Did You Know?</h4>
          <p>Even accounting for electricity generation and battery production, EVs can produce significantly fewer lifecycle emissions than gasoline cars.</p>
        </div>
        """, unsafe_allow_html=True)

    # Simple savings calc
    st.markdown('<h2 class="sub-header">💰 Cost Comparison: EV vs Gasoline Car</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 30000, 12000, 1000)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 6.0, 3.50, 0.10)
        car_mpg = st.number_input("Your car's MPG", 15, 40, 25, 1)
        electricity_rate = st.number_input("Electricity rate (¢/kWh)", 8, 30, 12, 1)
    with c2:
        annual_gas_cost = (annual_miles / car_mpg) * gas_price
        annual_electricity_cost = (annual_miles / 3.5) * (electricity_rate / 100)  # ~3.5 mi/kWh
        annual_savings = annual_gas_cost - annual_electricity_cost
        st.markdown(f"""
        <div class="prediction-result">
          <h4>💰 Your Annual Savings</h4>
          <p><strong>${annual_savings:.0f}</strong> per year</p>
          <p><strong>${annual_savings * 5:.0f}</strong> over 5 years</p>
          <p>Gas: ${annual_gas_cost:.0f} | Electric: ${annual_electricity_cost:.0f}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- Tab: Environmental Impact ----------
def show_environmental_impact():
    st.markdown('<h1 class="main-header">🌱 Environmental Impact</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        img = load_image_safe("search_images/PVmkLhcf3iJz.jpg")
        if img: st.image(img, caption="Going Green with Electric Vehicles", use_column_width=True)

        st.markdown("""
        <div class="env-gradient" style="padding:1.5rem;border-radius:1rem;margin:1rem 0;">
          <h3>🌍 Climate Impact</h3>
          <p>Transportation contributes substantially to greenhouse gas emissions.
          EVs help reduce tailpipe emissions and can be powered increasingly by renewables.</p>
          <h4>Key Environmental Benefits:</h4>
          <ul>
            <li>Zero tailpipe emissions</li>
            <li>Reduced urban air pollution</li>
            <li>Lower lifecycle carbon footprint</li>
            <li>Works well with renewable energy</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        img2 = load_image_safe("search_images/xqjndibmhIna.webp")
        if img2: st.image(img2, caption="EV Lifecycle Emissions Analysis", use_column_width=True)

        st.markdown("""
        <div class="fact-highlight">
          <h4>🔋 Battery Technology Progress</h4>
          <ul>
            <li>Energy density improving</li>
            <li>Costs far lower than a decade ago</li>
            <li>Recycling programs expanding</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------- Tab: EV Statistics ----------
def show_ev_statistics():
    st.markdown('<h1 class="main-header">📊 EV Market Statistics</h1>', unsafe_allow_html=True)

    years = list(range(2015, 2024))
    global_ev_sales = [0.4, 0.7, 1.2, 2.0, 2.2, 3.1, 6.6, 10.5, 14.1]  # in millions

    fig = px.line(x=years, y=global_ev_sales, title="Global Electric Vehicle Sales (2015-2023)",
                  labels={'x': 'Year', 'y': 'Sales (Millions)'}, markers=True)
    fig.update_traces(line_color='#2E8B57', line_width=3, marker_size=8)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Helpers ----------
def prepare_input_data(country, year, population, gdp, energy_per_capita,
                       primary_energy_consumption, cement_co2, coal_co2, oil_co2,
                       gas_co2, flaring_co2, methane, nitrous_oxide, features):
    """Build a one-row DataFrame with all required features."""
    try:
        x = {f: 0 for f in features}
        x['year'] = year
        x['population'] = population
        x['gdp'] = gdp * 1e9
        x['energy_per_capita'] = energy_per_capita
        x['primary_energy_consumption'] = primary_energy_consumption
        x['cement_co2'] = cement_co2
        x['coal_co2'] = coal_co2
        x['oil_co2'] = oil_co2
        x['gas_co2'] = gas_co2
        x['flaring_co2'] = flaring_co2
        x['methane'] = methane
        x['nitrous_oxide'] = nitrous_oxide

        # derived
        x['year_sq'] = year ** 2
        x['year_cub'] = year ** 3
        x['gdp_per_capita'] = x['gdp'] / max(population, 1)
        x['energy_per_capita_log'] = np.log1p(energy_per_capita)
        x['population_log'] = np.log1p(population)
        x['gdp_log'] = np.log1p(x['gdp'])
        x['cement_co2_log'] = np.log1p(cement_co2)
        x['coal_co2_log'] = np.log1p(coal_co2)
        x['flaring_co2_log'] = np.log1p(flaring_co2)
        x['gas_co2_log'] = np.log1p(gas_co2)
        x['oil_co2_log'] = np.log1p(oil_co2)
        x['gdp_x_energy'] = x['gdp'] * energy_per_capita
        x['population_x_gdp'] = population * x['gdp']

        # country dummy
        cfeat = f'country_{country}'
        if cfeat in x: x[cfeat] = 1

        return pd.DataFrame([x])
    except Exception as e:
        st.error(f"Error preparing input data: {e}")
        return None

def display_prediction_results(prediction, country, year, population):
    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Predicted CO2 Emissions", f"{prediction:.1f} Mt", help="Million tonnes of CO2 equivalent")
    per_capita_kg = (prediction * 1e6) / max(population, 1) * 1000
    with c2: st.metric("Per Capita Emissions", f"{per_capita_kg:.1f} kg", help="Kilograms of CO2 per person")
    global_avg_t = 4.8
    comparison = "Above" if per_capita_kg/1000 > global_avg_t else "Below"
    with c3: st.metric("vs Global Average", comparison, delta=f"{(per_capita_kg/1000 - global_avg_t):.1f}t")

    if prediction > 1000:
        level, color, msg = "Very High", "#ff4444", "Aggressive decarbonization strategies recommended."
    elif prediction > 500:
        level, color, msg = "High", "#ff8800", "Significant room for improvement via clean energy adoption."
    elif prediction > 100:
        level, color, msg = "Moderate", "#ffaa00", "Continue efforts to reduce emissions through sustainable practices."
    else:
        level, color, msg = "Low", "#44aa44", "Relatively low emissions. Keep up sustainable practices!"

    st.markdown(f"""
    <div class="prediction-result">
      <h3 style="color:{color};">Emission Level: {level}</h3>
      <p>{msg}</p>
      <p><strong>Country:</strong> {country} | <strong>Year:</strong> {year}</p>
    </div>
    """, unsafe_allow_html=True)

def display_charts(_, cement_co2, coal_co2, oil_co2, gas_co2, flaring_co2):
    st.markdown('<h2 class="sub-header">📈 CO2 Sources Breakdown</h2>', unsafe_allow_html=True)
    sources = ['Coal', 'Oil', 'Gas', 'Cement', 'Flaring']
    values = [coal_co2, oil_co2, gas_co2, cement_co2, flaring_co2]
    fig1 = px.pie(values=values, names=sources, title="CO2 Emissions by Source",
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    main()
