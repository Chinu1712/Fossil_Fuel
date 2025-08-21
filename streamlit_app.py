# app.py — Fossil Fuel COUNTDOWN (no rounded pill, no continents) — HEADER FIXED
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
BASE_DIR = Path(os.path.abspath(__file__)).parent if "__file__" in globals() else Path.cwd()


PROJECT_CONTEXT = """
You are a chatbot designed to answer questions about the "Fossil Fuel Countdown: The Race to EV & Renewables" project.
Identity (for “who are you?”): "I'm a chatbot here to assist you with the Fossil Fuel Countdown project — ask me anything about the experience, EVs, emissions, or what the charts mean."
"""

st.set_page_config(page_title="Fossil Fuel COUNTDOWN", page_icon="🛢️", layout="wide")

# ---------- THEME & CHATBOT UI (flatten inputs) ----------
st.markdown("""
<style>
:root{
  --bg:#0a1422; --fg:#e9f4ff; --muted:#93b0c8; --card:#0f2133; --card2:#132941;
  --border:rgba(255,255,255,.10); --accent:#12d7ff; --accent2:#ff2e7e;
  --shadow:0 18px 38px rgba(0,0,0,.35), 0 8px 18px rgba(0,0,0,.28);
}

/* Light mode palette */
@media (prefers-color-scheme: light){
  :root{ --bg:#f7fbff; --fg:#0b1b2b; --muted:#3c4e65; --card:#ffffff; --card2:#ffffff;
         --border:rgba(11,27,43,.12); --accent:#0ea5e9; --accent2:#d946ef; }
}

/* Background & base text */
html, body, .stApp {
  background:
   radial-gradient(1200px 800px at 75% 10%, rgba(18,215,255,.15), transparent 35%),
   radial-gradient(900px 600px at 20% 90%, rgba(255,46,126,.10), transparent 40%),
   var(--bg) !important;
  color: var(--fg) !important;
}

/* === STREAMLIT HEADER / TOP SPACING FIX ===
   - Make the Streamlit header transparent
   - Ensure content starts lower so it never gets clipped */
.stApp > header {
  background: transparent !important;
}

/* Give the main container generous top padding, including safe-area */
.main .block-container {
  padding-top: calc(2.6rem + env(safe-area-inset-top)) !important;
}

/* Extra safety on very small screens */
@media (max-width: 768px){
  .main .block-container { padding-top: calc(3.2rem + env(safe-area-inset-top)) !important; }
}

/* Header */
.header-wrap{ text-align:center; margin: .25rem 0 .6rem; }  /* small top margin to avoid clipping */
.header-wrap .title{
  font-size: 2.4rem; font-weight: 900; letter-spacing:.04em;
  color: var(--accent); text-shadow:0 0 14px rgba(18,215,255,.45);
}
@media (max-width: 680px){
  .header-wrap .title{ font-size: 1.9rem; }
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
.kpi .big { font-size: 1.2rem; font-weight: 900; letter-spacing:.02em; color:#ffb703; text-shadow:0 0 10px rgba(255,183,3,.35); }
.kpi small { color: var(--muted); }

/* Plotly */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Chatbot (flatten inputs/buttons) */
.chatwrap { max-width: 900px; margin: 0 auto; }
.chatcard {
  background: radial-gradient(200px 140px at 15% 0%, rgba(255,255,255,.08), transparent 42%),
              linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 18px; box-shadow: var(--shadow);
  padding: 18px 18px 12px;
}
.chatstream { max-height: 58vh; overflow-y: auto; padding: 6px 6px 2px; }
.bubble {
  padding: 12px 16px; border-radius: 14px; margin: 10px 6px; width: fit-content; max-width: 85%;
  box-shadow: 0 10px 25px rgba(0,0,0,.25); word-wrap: break-word;
}
.bubble.user { margin-left: auto; background: linear-gradient(145deg, #1db2ff, #0ea5e9); color: #052033; font-weight: 600; }
.bubble.bot  { background: linear-gradient(145deg, rgba(255,255,255,.30), rgba(255,255,255,.18));
               border: 1px solid rgba(255,255,255,.16); color: #f6fbff; }

/* == Kill the “pill” look == */
.chatwrap .stTextInput > div > div { border-radius: 6px !important; }
.chatwrap .stTextInput input {
  border-radius: 6px !important;
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.25) !important;
  box-shadow: none !important;
}
.chatwrap .sendbtn button,
.chatwrap .sendbtn [data-testid="baseButton-primary"],
.chatwrap .sendbtn [data-testid="baseButton-secondary"] {
  background: linear-gradient(145deg, #ff7a1a, #ff5252) !important;
  border: none !important; color: white !important; font-weight: 800 !important;
  border-radius: 6px !important; height: 44px !important;
  box-shadow: 0 10px 20px rgba(0,0,0,.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOADERS ----------
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

# ---------- HELPERS ----------
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

# ---------- GEMINI ----------
def _get_gemini_key()->str:
    key = st.secrets.get("GEMINI_API_KEY","")
    if not key: key = os.getenv("GEMINI_API_KEY","")
    return key

def _gemini_reply(user_message:str, history:list)->str:
    key=_get_gemini_key()
    if not key:
        return "❗ Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets."
    if user_message.strip().lower() in {"who are you","who are you?","who r u","who r u?"}:
        return "I'm a chatbot here to assist you with the Fossil Fuel Countdown project — ask me anything about the experience, EVs, emissions, or what the charts mean."
    preface = {"role":"user","parts":[{"text":PROJECT_CONTEXT}]}
    contents=[preface]
    for m in history:
        role = "user" if m["role"]=="user" else "model"
        contents.append({"role":role,"parts":[{"text":m["content"]}]} )
    contents.append({"role":"user","parts":[{"text":user_message}]} )
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

# ---------- HEADER ----------
st.markdown("""
<div class="header-wrap">
  <div class="title">Fossil Fuel COUNTDOWN</div>
  <div class="subtitle">Understand reserves and emissions — compare sources — and see how EVs change the trajectory.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔮 CO₂ Predictor",
    "🚗 EV Benefits",
    "🌱 Environmental Impact",
    "📊 EV Statistics",
    "💬 Chatbot"
])

# ---- CO₂ Predictor ----
with tabs[0]:
    data=load_sample_data()
    model, scaler, features = load_model()

    if data is not None and "country" in data.columns:
        countries=sorted(data["country"].dropna().unique().tolist())
        default_idx=countries.index("United States") if "United States" in countries else 0
        country=st.sidebar.selectbox("Country", countries, index=default_idx)
    else:
        country=st.sidebar.text_input("Country","United States")

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

    predicted = None
    per_capita_t = None
    if model is None or scaler is None or features is None:
        st.warning("Prediction model files not found; charts below still work.")
    else:
        if st.button("🔮 Predict CO₂ Emissions", type="primary"):
            try:
                X=prepare_input_data(country, year, population, gdp, energy_per_capita,
                                     primary_energy, cement, coal, oil, gas, flaring, methane, nitrous, features)
                Xs=scaler.transform(X)
                predicted=float(model.predict(Xs)[0]) # Mt
                per_capita_t=(predicted*1e6)/max(population,1)/1e3
                vs=4.8; delta=per_capita_t-vs; comp="above" if delta>0 else "below"

                kc1,kc2,kc3 = st.columns(3)
                with kc1:
                    st.markdown(f'<div class="kpi"><div class="big">{predicted:,.2f} Mt</div><small>Predicted total CO₂</small></div>', unsafe_allow_html=True)
                with kc2:
                    st.markdown(f'<div class="kpi"><div class="big">{per_capita_t:,.2f} t</div><small>Per-capita emissions</small></div>', unsafe_allow_html=True)
                with kc3:
                    st.markdown(f'<div class="kpi"><div class="big">{abs(delta):.1f} t</div><small>{comp} world avg (~4.8 t)</small></div>', unsafe_allow_html=True)

                # Historical vs Predicted chart
                data_local = data  # alias
                if data_local is not None and {'country','year','co2'}.issubset(set(data_local.columns)):
                    dctry = data_local[data_local['country']==country].dropna(subset=['year','co2'])
                    if not dctry.empty:
                        hist_df = dctry[['year','co2']].sort_values('year')
                        fig = px.line(hist_df, x='year', y='co2',
                                      title=f"Historical CO₂ for {country} with {year} prediction",
                                      labels={'year':'Year','co2':'CO₂ (Mt)'}, markers=True)
                        fig.add_scatter(x=[year], y=[predicted], mode='markers+text',
                                        name='Predicted', text=[f"{predicted:,.0f}"],
                                        textposition='top center')
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown('<div class="section-title">Source Mix</div>', unsafe_allow_html=True)
    source_breakdown_charts(coal, oil, gas, cement, flaring)

# ---- EV Benefits ----
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
  <small>Gas: ${annual_gas_cost:,.0f} • Electric: ${annual_elec_cost:,.0f}</small><br>
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

# ---- Environmental Impact ----
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

# ---- EV Statistics ----
with tabs[3]:
    years=list(range(2015,2024))
    global_ev_sales=[0.4,0.7,1.2,2.0,2.2,3.1,6.6,10.5,14.1]
    fig=px.line(x=years, y=global_ev_sales, markers=True,
                title="Global Electric Vehicle Sales (2015–2023)",
                labels={"x":"Year","y":"Sales (Millions)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ---- Chatbot ----
with tabs[4]:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role":"model",
             "content":"Hi there! I'm ready to answer any questions you have about the "
                       '"Fossil Fuel Countdown: The Race to EV & Renewables" project. '
                       "What would you like to know?"}
        ]

    st.markdown('<div class="chatwrap"><div class="chatcard">', unsafe_allow_html=True)
    st.markdown('<div class="chatstream">', unsafe_allow_html=True)

    def esc(s:str)->str:
        return s.replace("&","&amp;").replace("<","&lt;").replace(">", "&gt;")
    for m in st.session_state.chat_messages:
        cls = "user" if m["role"]=="user" else "bot"
        st.markdown(f'<div class="bubble {cls}">{esc(m["content"])}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end chatstream

    with st.form("chat_send_form", clear_on_submit=True):
        c1, c2 = st.columns([6,1])
        with c1:
            user_text = st.text_input(
                "Ask about fossil fuels, EVs, or CO₂…",
                label_visibility="collapsed",
                placeholder="Ask me about EVs, emissions, charts, or how to use this app…"
            )
        with c2:
            st.markdown('<div class="sendbtn">', unsafe_allow_html=True)
            sent = st.form_submit_button("Send", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if sent and user_text.strip():
            st.session_state.chat_messages.append({"role":"user","content":user_text.strip()})
            try:
                reply = _gemini_reply(user_text.strip(), st.session_state.chat_messages[:-1])
            except Exception as e:
                reply = f"Sorry, I hit an error: {e}"
            st.session_state.chat_messages.append({"role":"model","content":reply})
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
