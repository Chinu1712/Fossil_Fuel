import warnings
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Page + Dark theme CSS
# ---------------------------------------------------------
st.set_page_config(page_title="CO2 Emission Predictor & EV Awareness", page_icon="🌍", layout="wide")
st.markdown("""
<style>
  .stApp, body { background:#0e1117; color:#fff; }
  .main-header { font-size:2.6rem; color:#37c77f; text-align:center; margin:0 0 1rem 0; }
  .sub-header  { font-size:1.6rem; color:#8ab4f8; margin:1rem 0 .6rem 0; }
  .card { background:#1e2130; border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:1rem 1.2rem; }
  .accent { color:#37c77f; }
  .pill { background:linear-gradient(90deg,#2563eb,#16a34a); height:12px; border-radius:999px; margin:.6rem 0; }
  .btn-wide>button { width:100% !important; }
</style>
""", unsafe_allow_html=True)

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
        st.error(f"⚠️ Failed to load model assets: {e}")
        return None, None, None

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv(BASE_DIR / "owid-co2-data.csv")
    except Exception:
        return None

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def prepare_input_data(
    country, year, population, gdp, energy_per_capita,
    primary_energy_consumption, cement_co2, coal_co2, oil_co2,
    gas_co2, flaring_co2, methane, nitrous_oxide, features
):
    x = {f: 0 for f in features}
    x["year"] = year
    x["population"] = population
    x["gdp"] = gdp * 1e9
    x["energy_per_capita"] = energy_per_capita
    x["primary_energy_consumption"] = primary_energy_consumption
    x["cement_co2"] = cement_co2
    x["coal_co2"] = coal_co2
    x["oil_co2"] = oil_co2
    x["gas_co2"] = gas_co2
    x["flaring_co2"] = flaring_co2
    x["methane"] = methane
    x["nitrous_oxide"] = nitrous_oxide

    # engineered
    x["year_sq"] = year**2
    x["year_cub"] = year**3
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

    # country one-hot if the feature exists
    cfeat = f"country_{country}"
    if cfeat in x:
        x[cfeat] = 1

    return pd.DataFrame([x])

def show_source_breakdown_charts(coal, oil, gas, cement, flaring):
    df = pd.DataFrame({
        "Source": ["Coal", "Oil", "Gas", "Cement", "Flaring"],
        "Emissions (Mt)": [coal, oil, gas, cement, flaring],
    })
    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(df, names="Source", values="Emissions (Mt)", hole=0.35,
                     title="CO₂ Emissions by Source (input mix)")
        pie.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        bar = px.bar(df, x="Source", y="Emissions (Mt)", text_auto=True,
                     title="Source Contribution (bar view)")
        bar.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(bar, use_container_width=True)

def show_continent_stats(df):
    st.markdown('<h3 class="sub-header">🌍 Continent-level CO₂ Statistics</h3>', unsafe_allow_html=True)
    if df is None or "continent" not in df.columns or "co2" not in df.columns:
        st.info("Continental stats unavailable.")
        return
    recent = df[(df["year"] >= 2020) & df["continent"].notna()]
    cont = (recent.groupby("continent", as_index=False)["co2"]
            .sum().sort_values("co2", ascending=False))
    fig = px.bar(cont, x="continent", y="co2", text_auto=True,
                 title="Total CO₂ Emissions by Continent (2020+)",
                 labels={"continent": "Continent", "co2": "CO₂ Emissions (Mt)"})
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
    st.plotly_chart(fig, use_container_width=True)

    needed = {"coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"}
    if needed.issubset(recent.columns):
        bysrc = (recent.groupby("continent")[list(needed)]
                 .sum().reset_index().melt("continent", var_name="Source", value_name="Mt"))
        bysrc["Source"] = bysrc["Source"].str.replace("_co2", "", regex=False).str.title()
        fig2 = px.bar(bysrc, x="continent", y="Mt", color="Source", barmode="stack",
                      title="CO₂ by Source and Continent (2020+)",
                      labels={"continent": "Continent", "Mt": "CO₂ Emissions (Mt)"})
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Chatbot (local keyword-based; no external APIs)
# ---------------------------------------------------------
FAQ = [
    ("ev benefits", "EVs have lower operating costs, zero tailpipe emissions, and fewer moving parts, which reduces maintenance."),
    ("what does prediction mean", "The model estimates total annual CO₂ emissions (in million tonnes) given your inputs. Per-capita is shown for context."),
    ("why year to 2070", "Values beyond recent history are extrapolations—use them as scenarios, not certainties."),
    ("which source contributes most", "Check the Source Breakdown pie/bar—coal and oil are typically dominant where fossil electricity and transport are high."),
    ("how to reduce emissions", "Electrify transport, add renewables, improve efficiency, and reduce high-carbon fuels like coal and oil."),
    ("ev vs gas savings", "We compare gasoline cost vs electricity cost using your inputs; typical EV efficiency ~3.5 mi/kWh."),
    ("data source", "We use OWID CO₂ datasets for charts and trained the model on engineered features derived from them."),
]

def kb_answer(msg: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", msg.lower())
    tokens = set(text.split())
    best, score = None, 0
    for k, a in FAQ:
        kt = set(k.split())
        sc = len(tokens & kt)
        if sc > score:
            best, score = a, sc
    if score == 0:
        return ("I'm a lightweight helper. Ask me about: EV benefits, prediction meaning, "
                "sources, savings, or reducing emissions. I don't access the internet.")
    return best

def chat_widget():
    st.markdown('<h3 class="sub-header">🤖 Ask the Assistant</h3>', unsafe_allow_html=True)
    if "chat" not in st.session_state: st.session_state.chat = []
    for who, msg in st.session_state.chat:
        st.chat_message(who).markdown(msg)
    user = st.chat_input("Ask something about EVs, emissions, or this app…")
    if user:
        st.session_state.chat.append(("user", user))
        st.chat_message("user").markdown(user)
        reply = kb_answer(user)
        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").markdown(reply)

# ---------------------------------------------------------
# Pages
# ---------------------------------------------------------
def show_landing_page():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card" style="text-align:center;">'
        '<p style="font-size:1.1rem;margin:.2rem 0;">'
        'Understand emissions, explore EV savings, and see continent-level patterns with clean, dark-themed visuals.'
        '</p></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">💡 What you can do here</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
  <ul>
    <li><b>Predict</b> annual CO₂ using macro + energy inputs (with engineered features).</li>
    <li><b>Compare</b> running costs: EV vs gasoline—see your yearly & 5-year savings.</li>
    <li><b>Visualize</b> source mix (coal/oil/gas/cement/flaring) with pie + bar charts.</li>
    <li><b>Explore</b> continent rollups and stacked source bars for global context.</li>
  </ul>
  <div class="pill"></div>
  <p style="opacity:.9"><b>Note:</b> Years far in the future are scenario extrapolations.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Enter App", key="enter", use_container_width=True):
        st.session_state.page = "main"

def show_co2_predictor():
    model, scaler, features = load_model()
    sample = load_sample_data()
    if model is None or scaler is None or features is None:
        st.error("Model assets not available.")
        return

    st.markdown('<h2 class="sub-header">🔮 CO₂ Predictor</h2>', unsafe_allow_html=True)

    # --- Sidebar inputs (added Country) ---
    st.sidebar.markdown('<h3 class="sub-header">🔧 Input Parameters</h3>', unsafe_allow_html=True)
    # country selector: from file or text
    if sample is not None and "country" in sample.columns:
        countries = sorted(sample["country"].dropna().unique().tolist())
        default_idx = countries.index("United States") if "United States" in countries else 0
        country = st.sidebar.selectbox("Country", countries, index=default_idx)
    else:
        country = st.sidebar.text_input("Country", value="United States")

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
        st.info("ℹ️ You selected a far-future year. Treat results as scenario extrapolation.")

    if st.sidebar.button("🔮 Predict CO₂ Emissions", type="primary"):
        try:
            X = prepare_input_data(
                country, year, population, gdp, energy_per_capita,
                primary_energy_consumption, cement_co2, coal_co2, oil_co2,
                gas_co2, flaring_co2, methane, nitrous_oxide, features
            )
            Xs = scaler.transform(X)
            pred = float(model.predict(Xs)[0])  # Mt

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted CO₂ Emissions: **{pred:,.2f} Mt**")
            per_capita_tonnes = (pred * 1e6) / max(population, 1) / 1e3
            vs_avg_t = 4.8  # rough global per-capita tCO2
            delta = per_capita_tonnes - vs_avg_t
            comp = "above" if delta > 0 else "below"
            st.markdown(f"**Per Capita:** {per_capita_tonnes:,.2f} t/person "
                        f"({abs(delta):.1f} t {comp} global avg ~{vs_avg_t} t)")
            st.markdown(f"**Country used for features:** {country}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<h3 class="sub-header">📈 Source Breakdown</h3>', unsafe_allow_html=True)
            show_source_breakdown_charts(coal_co2, oil_co2, gas_co2, cement_co2, flaring_co2)

            show_continent_stats(sample)

            if sample is not None and {"year","country","co2"}.issubset(sample.columns):
                recent = (sample[sample["year"] >= 2020]
                          .groupby("country", as_index=True)["co2"]
                          .sum().sort_values(ascending=False).head(10))
                st.markdown('<h3 class="sub-header">🌐 Top Regions by CO₂ (2020+)</h3>', unsafe_allow_html=True)
                fig = px.bar(
                    x=recent.values, y=recent.index, orientation="h",
                    title="Top 10 Regions/Groups by Total CO₂ (2020+)",
                    labels={"x": "CO₂ Emissions (Mt)", "y": "Region/Country"},
                    color=recent.values, color_continuous_scale="Reds"
                )
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff", height=460)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    chat_widget()

def show_ev_benefits():
    st.markdown('<h2 class="sub-header">🚗 EV Benefits & Savings</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 30000, 12000, 1000)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 10.0, 3.50, 0.10)
        car_mpg = st.number_input("Your car's MPG", 10, 80, 25, 1)
        electricity_rate = st.number_input("Electricity rate (¢/kWh)", 4, 60, 12, 1)
    with c2:
        annual_gas_cost = (annual_miles / car_mpg) * gas_price
        annual_electricity_cost = (annual_miles / 3.5) * (electricity_rate / 100.0)
        annual_savings = annual_gas_cost - annual_electricity_cost
        st.markdown(
            f"""
<div class="card">
  <h4 class="accent">💰 Your Annual Savings</h4>
  <p style="font-size:1.2rem;"><b>${annual_savings:,.0f}</b> /year &nbsp;|&nbsp; <b>${annual_savings*5:,.0f}</b> over 5 years</p>
  <p>Gas: ${annual_gas_cost:,.0f} &nbsp;|&nbsp; Electric: ${annual_electricity_cost:,.0f}</p>
</div>""",
            unsafe_allow_html=True,
        )

def show_environmental_impact():
    st.markdown('<h2 class="sub-header">🌱 Environmental Impact</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
  <h4 class="accent">Why EVs help</h4>
  <ul>
    <li><b>Zero tailpipe emissions</b> ⇒ cleaner city air (NOx/PM).</li>
    <li><b>Grid synergy</b> ⇒ as grids add solar/wind, lifecycle CO₂ declines.</li>
    <li><b>High efficiency</b> ⇒ electric drivetrains convert ~3× the energy to motion vs ICE.</li>
    <li><b>Noise reduction</b> ⇒ quieter streets at low speeds.</li>
  </ul>
  <div class="pill"></div>
  <h4 class="accent">Battery considerations</h4>
  <ul>
    <li>Manufacturing CO₂ is front-loaded; typically offset during use.</li>
    <li>Recycling & second-life storage are scaling quickly.</li>
  </ul>
  <div class="pill"></div>
  <h4 class="accent">Practical actions</h4>
  <ul>
    <li>Prefer transit, cycling, or walking for short trips.</li>
    <li>Pick efficient vehicles; eco-drive to cut energy use.</li>
    <li>Charge off-peak or via renewables if available.</li>
    <li>Advocate for public charging and clean-power procurement.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

def show_ev_statistics():
    st.markdown('<h2 class="sub-header">📊 EV Market Statistics</h2>', unsafe_allow_html=True)
    years = list(range(2015, 2024))
    global_ev_sales = [0.4, 0.7, 1.2, 2.0, 2.2, 3.1, 6.6, 10.5, 14.1]  # millions
    fig = px.line(x=years, y=global_ev_sales, markers=True,
                  title="Global Electric Vehicle Sales (2015–2023)",
                  labels={"x": "Year", "y": "Sales (Millions)"})
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#fff", height=430)
    st.plotly_chart(fig, use_container_width=True)

def show_main_app():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    tabs = st.tabs(["🔮 CO₂ Predictor", "🚗 EV Benefits", "🌱 Environmental Impact", "📊 EV Statistics"])
    with tabs[0]: show_co2_predictor()
    with tabs[1]: show_ev_benefits()
    with tabs[2]: show_environmental_impact()
    with tabs[3]: show_ev_statistics()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Back to Landing Page", key="back", use_container_width=True):
        st.session_state.page = "landing"

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing_page()
else:
    show_main_app()
