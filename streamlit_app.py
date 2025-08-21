# app.py — Fossil Fuel COUNTDOWN (tabs + floating chatbot)
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent

# =========================
# PROJECT CONTEXT (always used by chatbot)
# =========================
PROJECT_CONTEXT = """
You are a chatbot designed to answer questions about the "Fossil Fuel Countdown: The Race to EV & Renewables" project.
This project is an interactive dashboard that shows how fast our petrol and diesel reserves are running out and how much we can extend the timeline by switching to EVs and renewable sources today.

Key Objectives:
- Create curiosity: Show how limited fossil fuel reserves are.
- Educate: Demonstrate the impact of daily fuel consumption on depletion timelines.
- Inspire: Visualize how EV adoption and renewable integration delay extinction of reserves.
- Engage: Let visitors simulate their own "switch scenarios" and see instant results.

What the Dashboard Shows:
- Large screen dashboard with a world map/country view showing known petrol/diesel reserves.
- Countdown timers for "Petrol runs out in X years" and "Diesel runs out in Y years" based on current consumption rates.
- A graph of reserves vs. time (business as usual vs. EV adoption).
- Interactive slider or buttons: "Adopt EVs faster" (10%, 25%, 50% per year) and "Add renewables" (% of grid energy from clean sources).
- Instant visual change in: extended reserve lifespan (years) and CO₂ reduction over time.
- Curiosity facts (e.g., "At current rates, the world uses enough oil in 1 day to fill 8,000 Olympic pools").
- Call to action: "If every 5th vehicle goes electric, reserves last X years longer."

Core Data Used (Simplified for Demo):
- Global oil reserves: ~1.65 trillion barrels.
- Annual global oil consumption: ~35 billion barrels/year.
- CO₂ emissions per litre: Petrol: ~2.31 kg CO₂/litre, Diesel: ~2.68 kg CO₂/litre.
- EV energy consumption & renewable adoption scenarios are simulated.

How the App Works (4 Steps):
1. Initial Calculation (Business As Usual): Years until depletion = Total reserves / Annual consumption. Shows timeline graph.
2. Scenario Simulation: Sliders adjust EV adoption rate/year and renewable share growth/year. Formula reduces fossil fuel consumption, new depletion dates appear instantly.
3. CO₂ Impact Visualization: CO₂ avoided per year from reduced fuel burning, shown as equivalents (e.g., "X million trees planted").
4. Comparison View: "If we do nothing" vs. "If we switch now" side-by-side bars.

Engagement Flow:
1. Visitor sees scary "Fossil Fuels Will Run Out In..." countdown.
2. You invite them to adjust EV adoption/renewable sliders.
3. They watch the depletion date move further into the future.
4. Show the CO₂ savings meter shoot up.
5. End with "Every small switch counts... what will you switch today?"

Visualization Ideas:
- Dual Countdown Timers (Petrol, Diesel: large, bold, red turning green).
- Curved depletion graphs (before vs. after).
- CO₂ Cloud Animation (shrinks as cleaner options are chosen).
- World map heatmap (top consumers in red, turning greener).
- Infographic-style callouts: "One EV saves ~X litres of fuel/year", "At 50% EV adoption, we delay petrol extinction by 15 years."

Identity:
If a user asks "who are you?" reply:
"I'm a chatbot here to assist you with the Fossil Fuel Countdown project — ask me anything about the experience, EVs, emissions, or what the charts mean."
"""

# =========================
# PAGE CONFIG + THEME
# =========================
st.set_page_config(page_title="Fossil Fuel COUNTDOWN", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#0a1422; --fg:#e9f4ff; --muted:#93b0c8; --card:#0f2133; --card2:#132941;
  --border:rgba(255,255,255,.10); --accent:#12d7ff; --accent2:#ff2e7e; --lime:#00ffb7;
  --shadow:0 18px 38px rgba(0,0,0,.35), 0 8px 18px rgba(0,0,0,.28);
  --zchat: 2147483647;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f7fbff; --fg:#0b1b2b; --muted:#3c4e65; --card:#ffffff; --card2:#ffffff;
    --border:rgba(11,27,43,.12); --accent:#0ea5e9; --accent2:#d946ef; --lime:#16a34a;
    --shadow:0 16px 30px rgba(11,27,43,.15), 0 8px 16px rgba(11,27,43,.08);
  }
}
html, body, .stApp {
  background:
   radial-gradient(1200px 800px at 75% 10%, rgba(18,215,255,.15), transparent 35%),
   radial-gradient(900px 600px at 20% 90%, rgba(255,46,126,.10), transparent 40%),
   var(--bg) !important;
  color: var(--fg) !important;
}
.block-container { padding-top: .6rem; }

.header-wrap{ text-align:center; margin: 0 0 .6rem; }
.header-wrap .title{
  font-size: 2.4rem; font-weight: 900; letter-spacing:.04em;
  color: var(--accent); text-shadow:0 0 14px rgba(18,215,255,.45);
}
.header-wrap .subtitle{ font-size: .95rem; color: var(--muted); max-width: 980px; margin: .25rem auto .6rem; }

.section-title { font-weight: 800; font-size: 1.2rem; margin: 1.0rem 0 .4rem; color:#bfe7ff; text-shadow:0 0 10px rgba(18,215,255,.35); }

.card {
  background: linear-gradient(180deg, rgba(18,215,255,.05), rgba(18,215,255,.02)), var(--card);
  border:1px solid var(--border); border-radius:16px; padding:1rem 1.1rem; box-shadow: var(--shadow);
}
.kpi{
  background: linear-gradient(180deg, rgba(255,46,126,.06), rgba(18,215,255,.05)), var(--card2);
  border:1px solid var(--border); border-radius:18px; padding:1rem; text-align:center; box-shadow: var(--shadow);
}
.kpi .big { font-size: 1.8rem; font-weight: 900; letter-spacing:.03em; color:#ff2e7e; text-shadow:0 0 12px rgba(255,46,126,.45); }
.kpi small { color: var(--muted); }

/* Plotly background */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Floating chat FAB and panel - Highest z-index possible */
.chat-fab {
  position: fixed !important;
  right: 22px !important;
  bottom: 22px !important;
  z-index: var(--zchat) !important;
  width: 60px;
  height: 60px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(145deg, #16c5ff, #ff2e7e);
  color: #fff;
  border: none;
  box-shadow: 0 16px 34px rgba(0,0,0,.35), 0 8px 18px rgba(0,0,0,.2);
  cursor: pointer;
  font-size: 26px;
  transition: all 0.3s ease;
}

.chat-fab:hover {
  transform: scale(1.05);
  box-shadow: 0 20px 40px rgba(0,0,0,.4), 0 10px 20px rgba(0,0,0,.3);
}

.chat-panel {
  position: fixed !important;
  right: 22px !important;
  bottom: 92px !important;
  z-index: calc(var(--zchat) - 1) !important;
  width: min(560px, 92vw);
  max-height: 70vh;
  background: linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.08));
  -webkit-backdrop-filter: blur(12px);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid rgba(255,255,255,.16);
  background: rgba(255,255,255,.05);
}

.chat-title {
  font-weight: 900;
  color: #f3f9ff;
  font-size: 1.1rem;
}

.chat-close {
  background: transparent;
  border: none;
  color: #f3f9ff;
  font-size: 22px;
  cursor: pointer;
  opacity: 0.9;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  transition: all 0.2s ease;
}

.chat-close:hover {
  background: rgba(255,255,255,.1);
  opacity: 1;
}

.chat-body {
  padding: 16px;
  overflow-y: auto;
  flex: 1;
  min-height: 200px;
  max-height: 400px;
}

.chat-input-area {
  padding: 12px 16px;
  border-top: 1px solid rgba(255,255,255,.16);
  background: rgba(255,255,255,.03);
}

.bubble {
  padding: 10px 14px;
  border-radius: 16px;
  margin: 8px 0;
  width: fit-content;
  max-width: 85%;
  box-shadow: 0 5px 16px rgba(0,0,0,.2);
  word-wrap: break-word;
}

.bubble.user {
  background: linear-gradient(145deg, rgba(22,197,255,.9), rgba(22,197,255,.7));
  color: #052033;
  margin-left: auto;
  font-weight: 500;
}

.bubble.bot {
  background: linear-gradient(145deg, rgba(255,255,255,.25), rgba(255,255,255,.18));
  color: #f7fbff;
  border: 1px solid rgba(255,255,255,.15);
}

/* Hide default streamlit elements that might interfere */
.stApp > header {
  background: transparent;
}

/* Ensure chat is always on top */
.chat-fab, .chat-panel {
  pointer-events: all !important;
}

/* Override any conflicting styles */
div[data-testid="stSidebar"] {
  z-index: 1000;
}

.main .block-container {
  z-index: 1;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOADERS
# =========================
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

# =========================
# HELPERS
# =========================
def prepare_input_data(country, year, population, gdp, energy_per_capita,
                       primary_energy_consumption, cement_co2, coal_co2, oil_co2,
                       gas_co2, flaring_co2, methane, nitrous_oxide, features):
    x = {f:0 for f in features}
    x["year"]=year
    x["population"]=population
    x["gdp"]=gdp*1e9
    x["energy_per_capita"]=energy_per_capita
    x["primary_energy_consumption"]=primary_energy_consumption
    x["cement_co2"]=cement_co2; x["coal_co2"]=coal_co2; x["oil_co2"]=oil_co2
    x["gas_co2"]=gas_co2; x["flaring_co2"]=flaring_co2
    x["methane"]=methane; x["nitrous_oxide"]=nitrous_oxide
    # engineered
    x["year_sq"]=year**2; x["year_cub"]=year**3
    x["gdp_per_capita"]=x["gdp"]/max(population,1)
    x["energy_per_capita_log"]=np.log1p(energy_per_capita)
    x["population_log"]=np.log1p(population); x["gdp_log"]=np.log1p(x["gdp"])
    for k in ["cement_co2","coal_co2","flaring_co2","gas_co2","oil_co2"]:
        x[f"{k}_log"]=np.log1p(x[k])
    x["gdp_x_energy"]=x["gdp"]*energy_per_capita
    x["population_x_gdp"]=population*x["gdp"]
    cfeat=f"country_{country}"
    if cfeat in x: x[cfeat]=1
    return pd.DataFrame([x])

def source_breakdown_charts(coal, oil, gas, cement, flaring):
    df = pd.DataFrame({"Source":["Coal","Oil","Gas","Cement","Flaring"],
                       "Emissions (Mt)":[coal,oil,gas,cement,flaring]})
    c1,c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="Source", values="Emissions (Mt)", hole=.35,
                     title="CO₂ by Source (input mix)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bar = px.bar(df, x="Source", y="Emissions (Mt)", text_auto=True,
                     title="Source Contribution")
        bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)

# =========================
# GEMINI (REST) — use Secrets env
# =========================
def _get_gemini_key()->str:
    key = st.secrets.get("GEMINI_API_KEY","")
    if not key: key = os.getenv("GEMINI_API_KEY","")
    return key

def _gemini_reply(user_message:str, history:list)->str:
    key=_get_gemini_key()
    if not key:
        return "❗ Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets."
    # Build contents with context + turns
    preface = {"role":"user","parts":[{"text":PROJECT_CONTEXT}]}
    contents=[preface]
    for m in history:
        role = "user" if m["role"]=="user" else "model"
        contents.append({"role":role,"parts":[{"text":m["content"]}]})
    contents.append({"role":"user","parts":[{"text":user_message}]})

    url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    try:
        r=requests.post(url, params={"key":key}, headers={"Content-Type":"application/json"},
                        json={"contents":contents}, timeout=60)
        r.raise_for_status()
        data=r.json()
        return data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","") or "No response."
    except requests.HTTPError as e:
        return f"HTTP error: {e} – {getattr(e,'response',None) and e.response.text}"
    except Exception as e:
        return f"Error: {e}"

# =========================
# FLOATING CHAT - FIXED VERSION
# =========================
def render_floating_chat():
    """
    Fixed floating chatbot with proper state management and event handling
    """
    # Initialize session state
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot", "content": "Hi! I'm here to help with the Fossil Fuel Countdown project. Ask me anything about EVs, emissions, or renewable energy!"}
        ]

    # Chat toggle buttons
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("💬", key="chat_toggle", help="Open Chat"):
            st.session_state.chat_open = True

    # Floating FAB (Always visible)
    st.markdown(f"""
    <div class="chat-fab" onclick="document.getElementById('chat_toggle').click();" title="Open Fossil Fuel Chat">
        💬
    </div>
    """, unsafe_allow_html=True)

    # Chat Panel
    if st.session_state.chat_open:
        # Create chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown("""
            <div class="chat-panel">
                <div class="chat-header">
                    <div class="chat-title">🛢️ Fossil Fuel Chat</div>
                </div>
                <div class="chat-body">
            """, unsafe_allow_html=True)
            
            # Display chat messages
            for msg in st.session_state.chat_history:
                role_class = "user" if msg["role"] == "user" else "bot"
                # Escape HTML to prevent XSS
                content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f'<div class="bubble {role_class}">{content}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Chat input area
            st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
            
            # Create columns for input and buttons
            input_col, send_col, close_col = st.columns([6, 1, 1])
            
            with input_col:
                user_input = st.text_input("Ask about fossil fuels, EVs, renewables...", 
                                         key="chat_input", 
                                         placeholder="Type your question here...",
                                         label_visibility="collapsed")
            
            with send_col:
                if st.button("Send", key="chat_send", use_container_width=True, type="primary"):
                    if user_input.strip():
                        # Add user message
                        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
                        
                        # Get bot response
                        try:
                            bot_response = _gemini_reply(user_input.strip(), st.session_state.chat_history[:-1])
                            st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                        except Exception as e:
                            error_msg = f"Sorry, I'm having trouble responding right now. Error: {str(e)}"
                            st.session_state.chat_history.append({"role": "bot", "content": error_msg})
                        
                        # Clear input and rerun to show new messages
                        st.rerun()
            
            with close_col:
                if st.button("✕", key="chat_close", use_container_width=True, help="Close Chat"):
                    st.session_state.chat_open = False
                    st.rerun()
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# =========================
# MAIN APP CONTENT
# =========================
# Header
st.markdown("""
<div class="header-wrap">
  <div class="title">Fossil Fuel COUNTDOWN</div>
  <div class="subtitle">Understand reserves and emissions — compare sources — and see how EVs change the trajectory.</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🔮 CO₂ Predictor", "🚗 EV Benefits", "🌱 Environmental Impact", "📊 EV Statistics"])

# ============ CO₂ Predictor ============
with tabs[0]:
    data=load_sample_data()
    model, scaler, features = load_model()

    if data is not None and "country" in data.columns:
        countries=sorted(data["country"].dropna().unique().tolist())
        default_idx=countries.index("United States") if "United States" in countries else 0
        country=st.sidebar.selectbox("Country", countries, index=default_idx)
    else:
        country=st.sidebar.text_input("Country","United States")

    # Up to 2070
    year=st.sidebar.slider("Year", 1990, 2070, 2023, 1)

    st.sidebar.markdown("**Population & GDP**")
    population=st.sidebar.number_input("Population", min_value=1, value=330_000_000, step=1_000_000)
    gdp=st.sidebar.number_input("GDP (billion USD)", 0.0, 40_000.0, 25_000.0, 100.0)

    st.sidebar.markdown("**Energy & Sources**")
    energy_per_capita=st.sidebar.number_input("Energy per Capita (kWh)", 0.0, 100_000.0, 12_000.0, 100.0)
    primary_energy=st.sidebar.number_input("Primary Energy (TWh)", 0.0, 20_000.0, 2_500.0, 10.0)
    cement=st.sidebar.number_input("Cement CO₂ (Mt)", 0.0, 2_000.0, 50.0, 1.0)
    coal=st.sidebar.number_input("Coal CO₂ (Mt)", 0.0, 10_000.0, 1200.0, 10.0)
    oil=st.sidebar.number_input("Oil CO₂ (Mt)", 0.0, 10_000.0, 800.0, 10.0)
    gas=st.sidebar.number_input("Gas CO₂ (Mt)", 0.0, 10_000.0, 600.0, 10.0)
    flaring=st.sidebar.number_input("Flaring CO₂ (Mt)", 0.0, 1_000.0, 10.0, 1.0)
    methane=st.sidebar.number_input("Methane (Mt CO₂e)", 0.0, 5_000.0, 300.0, 10.0)
    nitrous=st.sidebar.number_input("Nitrous Oxide (Mt CO₂e)", 0.0, 1_000.0, 100.0, 5.0)

    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    if model is None or scaler is None or features is None:
        st.warning("Prediction model files not found; charts below still work.")
    else:
        if st.button("🔮 Predict CO₂ Emissions", type="primary"):
            try:
                X=prepare_input_data(country, year, population, gdp, energy_per_capita,
                                     primary_energy, cement, coal, oil, gas, flaring, methane, nitrous, features)
                Xs=scaler.transform(X)
                pred=float(model.predict(Xs)[0]) # Mt
                per_capita_t=(pred*1e6)/max(population,1)/1e3
                vs=4.8; delta=per_capita_t-vs; comp="above" if delta>0 else "below"
                st.markdown(f"""
<div class="card">
  <b>🎯 Predicted CO₂</b><br>
  <span style="font-size:1.4rem">{pred:,.2f} Mt</span><br>
  <small>Per capita: {per_capita_t:,.2f} t — {abs(delta):.1f} t {comp} global avg ~{vs} t</small><br>
  <small>{country} • {year}</small>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown('<div class="section-title">Source Mix</div>', unsafe_allow_html=True)
    source_breakdown_charts(coal, oil, gas, cement, flaring)

# ============ EV Benefits ============
with tabs[1]:
    st.markdown('<div class="section-title">Cost & Savings</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        annual_miles=st.number_input("Annual miles driven", 5000, 40000, 12000, 500)
        gas_price=st.number_input("Gas price per gallon ($)", 2.0, 12.0, 3.50, 0.10)
        mpg=st.number_input("Your car's MPG (gasoline)", 10, 100, 30, 1)
        elec_rate=st.number_input("Electricity rate (¢/kWh)", 4, 80, 12, 1)
        ev_eff=st.slider("EV efficiency (mi/kWh)", 2.5, 6.0, 3.5, 0.1)
        maint_gas=st.slider("Yearly maintenance (Gasoline) $", 200, 2000, 900, 50)
        maint_ev=st.slider("Yearly maintenance (EV) $", 50, 1500, 400, 50)
        years=st.slider("Ownership years", 1, 10, 5, 1)
    with c2:
        annual_gas_cost=(annual_miles/mpg)*gas_price
        annual_elec_cost=(annual_miles/ev_eff)*(elec_rate/100.0)
        total_gas=years*(annual_gas_cost+maint_gas)
        total_ev =years*(annual_elec_cost+maint_ev)
        st.markdown(f"""
<div class="kpi">
  <div class="big">${(annual_gas_cost-annual_elec_cost):,.0f}</div>
  <small>Annual savings</small><br>
  <small>Gas fuel: ${annual_gas_cost:,.0f} • Electric: ${annual_elec_cost:,.0f}</small><br>
  <small>Total {years}-yr savings: ${(total_gas-total_ev):,.0f}</small>
</div>
""", unsafe_allow_html=True)

    d1,d2 = st.columns(2)
    with d1:
        pie = px.pie(names=["Gasoline (fuel+maint)","EV (energy+maint)"],
                     values=[annual_gas_cost+maint_gas, annual_elec_cost+maint_ev], hole=.35,
                     title="Annual running cost split")
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(pie, use_container_width=True)
    with d2:
        yrs=np.arange(1,years+1)
        df=pd.DataFrame({"Year":yrs,
                         "Gasoline Total ($)": yrs*(annual_gas_cost+maint_gas),
                         "EV Total ($)":       yrs*(annual_elec_cost+maint_ev)})
        line=px.line(df, x="Year", y=["Gasoline Total ($)","EV Total ($)"], markers=True,
                     title=f"Cumulative cost over {years} years")
        line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(line, use_container_width=True)

# ============ Environmental Impact ============
with tabs[2]:
    st.markdown("""
<div class="card">
  <b>Why this matters</b>
  <ul>
    <li>EVs remove tailpipe pollution (NOx, PM) and cut lifecycle CO₂ as the grid decarbonizes.</li>
    <li>Electric drivetrains are ~3× more energy-efficient than ICE vehicles.</li>
    <li>Battery emissions are front-loaded; typically offset in operation. Recycling & second-life are scaling.</li>
  </ul>
  <b>Actions</b>
  <ul>
    <li>Choose efficient transport; eco-drive; consolidate trips.</li>
    <li>Charge off-peak or with renewables when possible.</li>
    <li>Back public charging and clean power policies locally.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ============ EV Statistics ============
with tabs[3]:
    years=list(range(2015,2024))
    global_ev_sales=[0.4,0.7,1.2,2.0,2.2,3.1,6.6,10.5,14.1]
    fig=px.line(x=years, y=global_ev_sales, markers=True,
                title="Global Electric Vehicle Sales (2015–2023)",
                labels={"x":"Year","y":"Sales (Millions)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig, use_container_width=True)

# === Render floating chatbot on all pages ===
render_floating_chat()
