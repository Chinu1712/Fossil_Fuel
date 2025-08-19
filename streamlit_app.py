import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Basic setup
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CO2 Emission Predictor & EV Awareness",
    page_icon="🌍",
    layout="wide",
)

# Global dark theme helpers
st.markdown(
    """
<style>
  .stApp, body { background-color:#0e1117; color:#ffffff; }
  .main-header { font-size:3rem; color:#37c77f; text-align:center; margin:0 0 1.2rem 0; }
  .sub-header  { font-size:1.6rem; color:#8ab4f8; margin:1rem 0 .6rem 0; }
  .card {
      background:#1e2130;
      border-radius:12px;
      padding:1rem 1.25rem;
      color:#fff;
      border: 1px solid rgba(255,255,255,.06);
  }
  .accent { color:#37c77f; }
  .btn-primary > button {
    background-color:#37c77f !important; color:#0b1320 !important;
    border:none !important; font-weight:700 !important;
  }
  .btn-primary > button:hover { filter:brightness(0.92); }
  .pill {
    background:linear-gradient(90deg, #2563eb, #16a34a);
    height:14px; border-radius:999px; margin:.6rem 0;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(BASE_DIR / "co2_model.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        features = joblib.load(BASE_DIR / "model_features.pkl")
        return model, scaler, features
    except Exception as e:
        st.error(f"⚠️ Failed to load model/scaler/features: {e}")
        return None, None, None

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv(BASE_DIR / "owid-co2-data.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load sample data for charts: {e}")
        return None

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def prepare_input_data(
    country, year, population, gdp, energy_per_capita,
    primary_energy_consumption, cement_co2, coal_co2, oil_co2,
    gas_co2, flaring_co2, methane, nitrous_oxide, features
):
    """
    Build a one-row DataFrame with exactly the columns the scaler/model expect.
    Columns that aren't supplied are set to 0.
    """
    x = {f: 0 for f in features}

    # Base fields
    x["year"] = year
    x["population"] = population
    x["gdp"] = gdp * 1e9  # convert from billion USD → USD
    x["energy_per_capita"] = energy_per_capita
    x["primary_energy_consumption"] = primary_energy_consumption
    x["cement_co2"] = cement_co2
    x["coal_co2"] = coal_co2
    x["oil_co2"] = oil_co2
    x["gas_co2"] = gas_co2
    x["flaring_co2"] = flaring_co2
    x["methane"] = methane
    x["nitrous_oxide"] = nitrous_oxide

    # Engineered fields (match your earlier training)
    x["year_sq"] = year ** 2
    x["year_cub"] = year ** 3
    x["gdp_per_capita"] = x["gdp"] / max(population, 1)
    x["energy_per_capita_log"] = np.log1p(energy_per_capita)
    x["population_log"] = np.log1p(population)
    x["gdp_log"] = np.log1p(x["gdp"])
    x["cement_co2_log"] = np.log1p(cement_co2)
    x["coal_co2_log"] = np.log1p(coal_co2)
    x["flaring_co2_log"] = np.log1p(flaring_co2)
    x["gas_co2_log"] = np.log1p(gas_co2)
    x["oil_co2_log"] = np.log1p(oil_co2)
    x["gdp_x_energy"] = x["gdp"] * energy_per_capita
    x["population_x_gdp"] = population * x["gdp"]

    # Country one-hot (if present in model_features)
    cfeat = f"country_{country}"
    if cfeat in x:
        x[cfeat] = 1

    return pd.DataFrame([x])

def show_source_breakdown_charts(coal, oil, gas, cement, flaring):
    sources = ["Coal", "Oil", "Gas", "Cement", "Flaring"]
    vals = [coal, oil, gas, cement, flaring]
    df = pd.DataFrame({"Source": sources, "Emissions (Mt)": vals})

    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(
            df, names="Source", values="Emissions (Mt)",
            title="CO₂ Emissions by Source (input mix)",
            hole=0.35
        )
        pie.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        bar = px.bar(
            df, x="Source", y="Emissions (Mt)",
            title="Source Contribution (bar view)", text_auto=True
        )
        bar.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(bar, use_container_width=True)

def show_continent_stats(df):
    st.markdown('<h3 class="sub-header">🌍 Continent-level CO₂ Statistics</h3>', unsafe_allow_html=True)
    if df is None or "continent" not in df.columns or "co2" not in df.columns:
        st.info("Continental stats unavailable (missing columns).")
        return

    recent = df[(df["year"] >= 2020) & df["continent"].notna()]
    cont = (recent.groupby("continent", as_index=False)["co2"]
            .sum().sort_values("co2", ascending=False))
    fig = px.bar(
        cont, x="continent", y="co2",
        title="Total CO₂ Emissions by Continent (2020+)",
        labels={"continent": "Continent", "co2": "CO₂ Emissions (Mt)"},
        text_auto=True
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: stacked bar by source if columns exist
    needed = {"coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"}
    if needed.issubset(recent.columns):
        bysrc = (recent.groupby("continent")[list(needed)]
                 .sum().reset_index().melt("continent", var_name="Source", value_name="Mt"))
        bysrc["Source"] = bysrc["Source"].str.replace("_co2", "", regex=False).str.title()
        fig2 = px.bar(
            bysrc, x="continent", y="Mt", color="Source", barmode="stack",
            title="CO₂ by Source and Continent (2020+)",
            labels={"continent": "Continent", "Mt": "CO₂ Emissions (Mt)"}
        )
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Landing Page
# ---------------------------------------------------------
def show_landing_page():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card" style="text-align:center;">'
        '<p style="font-size:1.15rem; margin:.2rem 0;">'
        'Leverage data science to understand emissions, compare transport choices, and explore policies '
        'that accelerate the shift to a low-carbon future.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="sub-header">💡 What this platform offers</h2>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card">
  <ul>
    <li><b>Forecast-ready CO₂ predictor:</b> Enter macro & energy inputs and get model-backed estimates.</li>
    <li><b>Cost comparison (EV vs Gas):</b> Instantly see your yearly savings and 5-year impact.</li>
    <li><b>Clean visuals:</b> Source pie/bar charts, continent rollups, and global context.</li>
    <li><b>Actionable guidance:</b> Practical steps for individuals, campuses, and cities.</li>
  </ul>
  <div class="pill"></div>
  <p style="opacity:.9"><b>Tip:</b> Values beyond the model's training range will extrapolate—use them as scenarios, not certainties.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="card">
  <h4 class="accent">📈 Accurate Predictions</h4>
  <p>Model trained on OWID data with engineered features (logs, interactions, time powers) for signal capture.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card">
  <h4 class="accent">🚗 EV Advantage</h4>
  <p>Lower operating cost, fewer moving parts, and rapid grid decarbonization amplify lifecycle gains.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="card">
  <h4 class="accent">🌱 Climate Literacy</h4>
  <p>Simple, shareable narratives to build awareness and motivate policy-aligned behavior.</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Enter the App", key="enter", help="Open predictor, EV savings, and stats", use_container_width=True):
        st.session_state["page"] = "main"

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
def show_co2_predictor():
    model, scaler, features = load_model()
    sample = load_sample_data()
    if model is None or scaler is None or features is None:
        st.error("Model resources not available.")
        return

    st.markdown('<h2 class="sub-header">🔮 CO₂ Predictor</h2>', unsafe_allow_html=True)

    # Sidebar inputs (kept compact but complete)
    st.sidebar.markdown('<h3 class="sub-header">🔧 Input Parameters</h3>', unsafe_allow_html=True)
    year = st.sidebar.slider("Year", 1990, 2070, 2023, 1)
    population = st.sidebar.number_input("Population", min_value=1, value=330_000_000, step=1_000_000)
    gdp = st.sidebar.number_input("GDP (billion USD)", min_value=0.0, value=25_000.0, step=100.0)

    st.sidebar.markdown("**Energy & Sources**")
    energy_per_capita = st.sidebar.number_input("Energy per Capita (kWh)", 0.0, 100_000.0, 12_000.0, 100.0)
    primary_energy_consumption = st.sidebar.number_input("Primary Energy Consumption (TWh)", 0.0, 20_000.0, 2_500.0, 10.0)
    cement_co2 = st.sidebar.number_input("Cement CO₂ (Mt)", 0.0, 2_000.0, 50.0, 1.0)
    coal_co2 = st.sidebar.number_input("Coal CO₂ (Mt)", 0.0, 10_000.0, 1_200.0, 10.0)
    oil_co2 = st.sidebar.number_input("Oil CO₂ (Mt)", 0.0, 10_000.0, 800.0, 10.0)
    gas_co2 = st.sidebar.number_input("Gas CO₂ (Mt)", 0.0, 10_000.0, 600.0, 10.0)
    flaring_co2 = st.sidebar.number_input("Flaring CO₂ (Mt)", 0.0, 1_000.0, 10.0, 1.0)
    methane = st.sidebar.number_input("Methane (Mt CO₂e)", 0.0, 5_000.0, 300.0, 10.0)
    nitrous_oxide = st.sidebar.number_input("Nitrous Oxide (Mt CO₂e)", 0.0, 1_000.0, 100.0, 5.0)

    if year > 2035:
        st.info("ℹ️ You selected a year beyond typical training windows. Results are an extrapolation scenario.")

    if st.sidebar.button("🔮 Predict CO₂ Emissions", type="primary"):
        try:
            X = prepare_input_data(
                "United States", year, population, gdp, energy_per_capita,
                primary_energy_consumption, cement_co2, coal_co2, oil_co2,
                gas_co2, flaring_co2, methane, nitrous_oxide, features
            )
            Xs = scaler.transform(X)
            pred = float(model.predict(Xs)[0])  # Mt

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted CO₂ Emissions: **{pred:,.2f} Mt**")
            per_capita_tonnes = (pred * 1e6) / max(population, 1)
            per_capita_tonnes /= 1e3  # kg -> tonnes
            st.markdown(f"**Per Capita Emissions:** {per_capita_tonnes:,.2f} tonnes per person")
            st.markdown("</div>", unsafe_allow_html=True)

            # Source mix visuals
            st.markdown('<h3 class="sub-header">📈 Source Breakdown</h3>', unsafe_allow_html=True)
            show_source_breakdown_charts(coal_co2, oil_co2, gas_co2, cement_co2, flaring_co2)

            # Global/continent context
            show_continent_stats(sample)

            # Optional: show top regions/countries (already aggregated ones exist in OWID)
            if sample is not None and {"year", "country", "co2"}.issubset(sample.columns):
                recent = (
                    sample[sample["year"] >= 2020]
                    .groupby("country", as_index=True)["co2"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.markdown('<h3 class="sub-header">🌐 Top Regions by CO₂ (2020+)</h3>', unsafe_allow_html=True)
                fig = px.bar(
                    x=recent.values, y=recent.index, orientation="h",
                    title="Top 10 Regions by Total CO₂ (2020+)",
                    labels={"x": "CO₂ Emissions (Mt)", "y": "Region/Country"},
                    color=recent.values, color_continuous_scale="Reds"
                )
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff", height=460)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

def show_ev_benefits():
    st.markdown('<h2 class="sub-header">🚗 EV Benefits & Savings</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 30000, 12000, 1000)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 10.0, 3.50, 0.10)
        car_mpg = st.number_input("Your car's MPG", 10, 80, 25, 1)
        electricity_rate = st.number_input("Electricity rate (¢/kWh)", 4, 60, 12, 1)
    with c2:
        # Simple comparison: ~3.5 mi/kWh efficiency assumption for EV
        annual_gas_cost = (annual_miles / car_mpg) * gas_price
        annual_electricity_cost = (annual_miles / 3.5) * (electricity_rate / 100.0)
        annual_savings = annual_gas_cost - annual_electricity_cost

        st.markdown(
            f"""
<div class="card">
  <h4 class="accent">💰 Your Annual Savings</h4>
  <p style="font-size:1.2rem;"><b>${annual_savings:,.0f}</b> per year &nbsp;|&nbsp; <b>${annual_savings*5:,.0f}</b> over 5 years</p>
  <p>Gas: ${annual_gas_cost:,.0f} &nbsp;|&nbsp; Electric: ${annual_electricity_cost:,.0f}</p>
</div>
""",
            unsafe_allow_html=True,
        )

def show_environmental_impact():
    st.markdown('<h2 class="sub-header">🌱 Environmental Impact</h2>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h4 class="accent">Why EVs help</h4>
  <ul>
    <li><b>Zero tailpipe emissions</b> ⇒ improved city air quality (NOx, PM drop).</li>
    <li><b>Grid synergy</b> ⇒ as grids add solar/wind, EV lifecycle CO₂ keeps falling.</li>
    <li><b>Efficiency</b> ⇒ electric drivetrains convert ~75–90% of energy to motion vs ~20–30% for ICE.</li>
    <li><b>Noise reduction</b> ⇒ quieter streets, especially at low speeds.</li>
  </ul>
  <h4 class="accent">What about batteries?</h4>
  <ul>
    <li>Manufacturing emissions are front-loaded but offset during use in most regions.</li>
    <li>Recycling capacity is expanding; second-life storage is growing.</li>
  </ul>
  <div class="pill"></div>
  <h4 class="accent">Practical actions</h4>
  <ul>
    <li>Prefer public transit, biking, and walking for short trips.</li>
    <li>Choose smaller, efficient vehicles; drive smoothly to cut energy use.</li>
    <li>Charge off-peak or with renewables if available.</li>
    <li>Advocate campus/municipal charging, clean-power procurement, and car-share fleets.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

def show_ev_statistics():
    st.markdown('<h2 class="sub-header">📊 EV Market Statistics</h2>', unsafe_allow_html=True)
    years = list(range(2015, 2024))
    global_ev_sales = [0.4, 0.7, 1.2, 2.0, 2.2, 3.1, 6.6, 10.5, 14.1]  # millions

    fig = px.line(
        x=years, y=global_ev_sales, markers=True,
        title="Global Electric Vehicle Sales (2015–2023)",
        labels={"x": "Year", "y": "Sales (Millions)"}
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff", height=430)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Main App wrapper
# ---------------------------------------------------------
def show_main_app():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 CO₂ Predictor", "🚗 EV Benefits", "🌱 Environmental Impact", "📊 EV Statistics"])

    with tab1: show_co2_predictor()
    with tab2: show_ev_benefits()
    with tab3: show_environmental_impact()
    with tab4: show_ev_statistics()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Back to Landing Page", key="back", use_container_width=True):
        st.session_state["page"] = "landing"

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    show_landing_page()
else:
    show_main_app()
