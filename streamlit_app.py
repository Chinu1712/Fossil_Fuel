# streamlit_app.py
import warnings
from pathlib import Path
import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent

# =========================
# Page & Global Styles (Neon Landing Look)
# =========================
st.set_page_config(page_title="Fossil Fuel COUNTDOWN", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
/* ======= COLOR SYSTEM (prefers-dark first) ======= */
:root{
  --bg:#061a28; --fg:#cfe9ff; --muted:#8fb0c9; --card:#0b2a3f; --border:rgba(255,255,255,.06);
  --accent:#12d7ff; --accent2:#ff2e7e; --lime:#00ffb7; --warning:#ffd166; --danger:#ff6b6b;
  --shadow:0 10px 30px rgba(0,0,0,.35);
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f5fbff; --fg:#0b1b2b; --muted:#374b63; --card:#ffffff; --border:rgba(11,27,43,.12);
    --accent:#0ea5e9; --accent2:#d946ef; --lime:#16a34a; --warning:#b45309; --danger:#dc2626;
    --shadow:0 8px 24px rgba(11,27,43,.15);
  }
}

/* ======= Base ======= */
html, body, .stApp { background: var(--bg) !important; color: var(--fg) !important; }
.block-container { padding-top: 1rem; }

/* Star field background (subtle) */
.stApp:before {
  content:""; position:fixed; inset:0; pointer-events:none;
  background:
    radial-gradient(circle at 20% 10%, rgba(18,215,255,.12), transparent 35%),
    radial-gradient(circle at 80% 30%, rgba(255,46,126,.11), transparent 40%),
    radial-gradient(circle at 60% 80%, rgba(0,255,183,.12), transparent 40%);
  mix-blend-mode:screen;
}

/* Headings */
.hero   { text-align:center; margin: 0 0 1.6rem; }
.hero h1{
  font-size: 3.2rem; letter-spacing: .05em; font-weight:800; margin:.2rem 0 0;
  color: var(--accent); text-shadow: 0 0 10px rgba(18,215,255,.65), 0 0 25px rgba(18,215,255,.35);
}
.hero h2{
  font-size: 3.0rem; letter-spacing: .25em; font-weight:900; margin:0 0 .4rem;
  color: #78c7ff; text-shadow: 0 0 16px rgba(120,199,255,.45);
}
.hero p { color: var(--muted); max-width: 860px; margin: .6rem auto 0; }

/* Buttons & cards */
.card {
  background: var(--card); border:1px solid var(--border); border-radius:16px; padding:1rem 1.2rem; 
  box-shadow: var(--shadow);
}
.kpi {
  background: linear-gradient(180deg, rgba(18,215,255,.04), rgba(18,215,255,.02));
  border:1px solid var(--border); border-radius:18px; padding:1.0rem 1.2rem; text-align:center; 
  box-shadow: var(--shadow);
}
.kpi h3 { margin:.25rem 0; font-size: 2.1rem; color: var(--fg);}
.kpi small { color: var(--muted); }
.kpi .big {
  font-size: 2.4rem; font-weight:900; color:#ff2e7e; letter-spacing:.06em;
  text-shadow:0 0 14px rgba(255,46,126,.45);
}
.neon-btn>button{
  width:100%; font-weight:700; border-radius:12px;
  background:linear-gradient(90deg, var(--accent), var(--accent2));
  box-shadow:0 6px 20px rgba(18,215,255,.25), 0 6px 20px rgba(255,46,126,.15);
  border:none;
}

/* Section headings */
.section-title {
  font-size:1.45rem; margin: 1.4rem 0 .6rem; font-weight:800; letter-spacing:.02em;
  color:#bfe7ff; text-shadow:0 0 10px rgba(18,215,255,.35);
}

/* Plotly background match */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ====== Floating Chat (moved up to avoid Streamlit icon) ====== */
.fab{
  position:fixed; right:22px; bottom:110px; z-index:999999;
  width:54px; height:54px; border-radius:999px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg, var(--accent2), var(--accent)); color:#fff; border:none;
  box-shadow:0 10px 30px rgba(0,0,0,.35); cursor:pointer; font-size:26px;
}
.fab:hover{ filter:brightness(.96) }

.chat-window{
  position:fixed; right:22px; bottom:180px; z-index:999998;
  width:min(380px, 92vw); max-height:min(70vh, 640px);
  background:var(--card); color:var(--fg); border:1px solid var(--border); border-radius:16px;
  overflow:hidden; box-shadow: var(--shadow); display:none;
}
.chat-header{ display:flex; align-items:center; justify-content:space-between; padding:.75rem 1rem; border-bottom:1px solid var(--border);}
.chat-title{ font-weight:800; color:#bfe7ff; }
.chat-close{ background:transparent; color:var(--muted); border:none; font-size:20px; cursor:pointer; }
.chat-body{ padding:10px 12px; overflow:auto; max-height:50vh; }
.bubble{ padding:.6rem .8rem; border-radius:12px; margin:.35rem 0; width:fit-content; max-width:92%; }
.me{ background: rgba(18,215,255,.15); }
.bot{ background: rgba(255,46,126,.15); }
.chat-input{ display:flex; gap:.5rem; padding:.6rem; border-top:1px solid var(--border); background:var(--card); }
.chat-input textarea{
  flex:1; border:1px solid var(--border); border-radius:12px; padding:.55rem .7rem; background:transparent; color:var(--fg);
  height:66px; resize:vertical;
}
.chat-send{ border:none; border-radius:10px; padding:.55rem .9rem; 
  background:linear-gradient(90deg, var(--accent), var(--accent2)); color:#fff; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# =========================
# Data + Model loaders
# =========================
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

# =========================
# Helpers for predictor + charts
# =========================
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

    cfeat = f"country_{country}"
    if cfeat in x:
        x[cfeat] = 1
    return pd.DataFrame([x])

def source_breakdown_charts(coal, oil, gas, cement, flaring):
    df = pd.DataFrame({
        "Source": ["Coal", "Oil", "Gas", "Cement", "Flaring"],
        "Emissions (Mt)": [coal, oil, gas, cement, flaring],
    })
    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(df, names="Source", values="Emissions (Mt)", hole=.35,
                     title="CO₂ Emissions by Source (input mix)")
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        bar = px.bar(df, x="Source", y="Emissions (Mt)", text_auto=True, title="Source Contribution (bar)")
        bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(bar, use_container_width=True)

def continent_stats(df):
    if df is None or "continent" not in df.columns or "co2" not in df.columns:
        st.info("Continental stats unavailable.")
        return
    st.markdown('<div class="section-title">🌍 Continent-level CO₂ Statistics</div>', unsafe_allow_html=True)
    recent = df[(df["year"] >= 2020) & df["continent"].notna()]
    cont = (recent.groupby("continent", as_index=False)["co2"]
            .sum().sort_values("co2", ascending=False))
    fig = px.bar(cont, x="continent", y="co2", text_auto=True,
                 title="Total CO₂ Emissions by Continent (2020+)",
                 labels={"continent":"Continent", "co2":"CO₂ Emissions (Mt)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
    st.plotly_chart(fig, use_container_width=True)

    needed = {"coal_co2","oil_co2","gas_co2","cement_co2","flaring_co2"}
    if needed.issubset(recent.columns):
        bysrc = (recent.groupby("continent")[list(needed)]
                 .sum().reset_index().melt("continent", var_name="Source", value_name="Mt"))
        bysrc["Source"] = bysrc["Source"].str.replace("_co2","",regex=False).str.title()
        fig2 = px.bar(bysrc, x="continent", y="Mt", color="Source", barmode="stack",
                      title="CO₂ by Source & Continent (2020+)",
                      labels={"continent":"Continent","Mt":"CO₂ Emissions (Mt)"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# Gemini Chat (floating) – uses st.query_params (no deprecated API)
# =========================
def _get_gemini_key() -> str:
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")
    return key

def _gemini_reply(user_message: str, history: list) -> str:
    api_key = _get_gemini_key()
    if not api_key:
        return "❗ Gemini API key is missing. Set it in Streamlit Secrets."
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts":[{"text": m["content"]}]})
    contents.append({"role":"user","parts":[{"text": user_message}]})
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    r = requests.post(url, params={"key":api_key}, headers={"Content-Type":"application/json"},
                      json={"contents":contents}, timeout=60)
    try:
        r.raise_for_status()
        data = r.json()
        return (data.get("candidates",[{}])[0]
                .get("content",{}).get("parts",[{}])[0].get("text","")) or "No response."
    except requests.HTTPError as e:
        return f"HTTP error: {e} — {getattr(e,'response',None) and e.response.text}"
    except Exception as e:
        return f"Error: {e}"

def render_floating_chat():
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role":"model","content":"Hi! Ask about EV savings, emissions, charts, or this app."}]

    qp = dict(st.query_params)
    msg = qp.get("chatq")
    if msg:
        msg = msg.strip()
        if msg:
            st.session_state.chat.append({"role":"user","content":msg})
            reply = _gemini_reply(msg, st.session_state.chat)
            st.session_state.chat.append({"role":"model","content":reply})
        st.query_params["open"] = "1"
        if "chatq" in st.query_params: del st.query_params["chatq"]

    # bubbles HTML
    def esc(t:str)->str:
        return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    bubbles = "".join([f'<div class="bubble {"me" if m["role"]=="user" else "bot"}">{esc(m["content"])}</div>'
                       for m in st.session_state.chat])

    # HTML (no f-strings around JS)
    st.markdown("""
<button class="fab" id="fabBtn" title="Ask the assistant">💬</button>
<div class="chat-window" id="chatWin">
  <div class="chat-header">
    <div class="chat-title">AI Assistant</div>
    <button class="chat-close" id="chatClose">✕</button>
  </div>
  <div class="chat-body" id="chatBody">
""", unsafe_allow_html=True)
    st.markdown(bubbles, unsafe_allow_html=True)
    st.markdown("""
  </div>
  <div class="chat-input">
    <textarea id="chatInput" placeholder="Type your question..."></textarea>
    <button class="chat-send" id="chatSend">Send</button>
  </div>
</div>

<script>
(function(){
  const fab = document.getElementById("fabBtn");
  const box = document.getElementById("chatWin");
  const closeBtn = document.getElementById("chatClose");
  const sendBtn = document.getElementById("chatSend");
  const input = document.getElementById("chatInput");
  if(!fab || !box) return;

  function openBox(){ box.style.display = "block"; }
  function closeBox(){ box.style.display = "none"; }

  try{
    const sp = new URLSearchParams(window.location.search);
    if(sp.get("open")==="1"){ openBox(); }
  }catch(e){}

  fab.onclick = function(){
    if(box.style.display==="block"){
      closeBox();
      try{
        const u = new URL(window.location.href);
        u.searchParams.delete("open");
        window.history.replaceState({}, "", u.toString());
      }catch(e){}
    }else{
      openBox();
      try{
        const u = new URL(window.location.href);
        u.searchParams.set("open","1");
        window.history.replaceState({}, "", u.toString());
      }catch(e){}
    }
  };
  if(closeBtn) closeBtn.onclick = function(){
    closeBox();
    try{
      const u = new URL(window.location.href);
      u.searchParams.delete("open");
      window.history.replaceState({}, "", u.toString());
    }catch(e){}
  };

  function send(){
    const val = (input && input.value || "").trim();
    if(!val) return;
    try{
      const u = new URL(window.location.href);
      u.searchParams.set("chatq", val);
      u.searchParams.set("open","1");
      window.location.href = u.toString();
    }catch(e){}
  }
  if(sendBtn) sendBtn.onclick = send;
  if(input) input.addEventListener("keydown", function(ev){ if(ev.key==="Enter" && !ev.shiftKey){ ev.preventDefault(); send(); }});
})();
</script>
""", unsafe_allow_html=True)

# =========================
# UI Sections
# =========================
def hero_and_stats():
    st.markdown("""
<div class="hero">
  <div class="chip"></div>
  <h1>Fossil Fuel</h1>
  <h2>COUNTDOWN</h2>
  <p>Discover how fast our <b>petrol and diesel reserves</b> are running out — and how much we can extend the timeline
     by switching to <b>EVs and renewable sources</b> today.</p>
</div>
""", unsafe_allow_html=True)

    # Controls for countdown (like your design numbers)
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        reserves_tbbl = st.number_input("Global liquid reserves (Trillion barrels eq.)", 0.1, 5.0, 1.65, 0.01, key="res")
    with colB:
        annual_bby = st.number_input("Annual consumption (Billion barrels / year)", 10.0, 800.0, 358.0, 1.0, key="cons")
    with colC:
        ev_extension = st.slider("EV adoption years added (scenario)", 0, 50, 25, 1)

    years_remaining = (reserves_tbbl * 1000.0) / max(annual_bby, 1e-6)
    petrol_years = max(int(round(years_remaining)), 0)
    diesel_years = max(int(round(years_remaining * 1.06)), 0)  # small offset to vary visuals

    k1, k2, k3 = st.columns([1,1,1])
    with k1:
        st.markdown(f"""
<div class="kpi">
  <div class="big">{petrol_years} YEARS</div>
  <small>Until Petrol Depletion</small>
</div>
""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
<div class="kpi">
  <div class="big">{diesel_years} YEARS</div>
  <small>Until Diesel Depletion</small>
</div>
""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
<div class="kpi">
  <div class="big">+{ev_extension}</div>
  <small>EV Impact Potential (years possible)</small>
</div>
""", unsafe_allow_html=True)

    # Live project statistics grid
    st.markdown('<div class="section-title">📡 Live Project Statistics</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""
<div class="card"><b>Global Reserves</b><br><span style="font-size:1.6rem">{reserves_tbbl:.2f}T</span><br>
<small>Trillion barrels (BP Statistical Review)</small></div>
""", unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
<div class="card"><b>Annual Consumption</b><br><span style="font-size:1.6rem">{annual_bby:.0f}B</span><br>
<small>Billion barrels per year</small></div>
""", unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
<div class="card"><b>Time Remaining</b><br><span style="font-size:1.6rem">{years_remaining:,.1f}</span><br>
<small>Years until depletion (simple ratio)</small></div>
""", unsafe_allow_html=True)

def predictor_section():
    st.markdown('<div class="section-title">🔮 CO₂ Predictor</div>', unsafe_allow_html=True)
    model, scaler, features = load_model()
    data = load_sample_data()

    # Sidebar Inputs
    if data is not None and "country" in data.columns:
        countries = sorted(data["country"].dropna().unique().tolist())
        default_idx = countries.index("United States") if "United States" in countries else 0
        country = st.sidebar.selectbox("Country", countries, index=default_idx)
    else:
        country = st.sidebar.text_input("Country", "United States")
    year = st.sidebar.slider("Year", 1990, 2070, 2023, 1)
    st.sidebar.markdown("**Population & GDP**")
    population = st.sidebar.number_input("Population", min_value=1, value=330_000_000, step=1_000_000)
    gdp = st.sidebar.number_input("GDP (billion USD)", min_value=0.0, value=25_000.0, step=100.0)

    st.sidebar.markdown("**Energy & Sources**")
    energy_per_capita = st.sidebar.number_input("Energy per Capita (kWh)", 0.0, 100_000.0, 12_000.0, 100.0)
    primary_energy_consumption = st.sidebar.number_input("Primary Energy (TWh)", 0.0, 20_000.0, 2_500.0, 10.0)
    cement_co2 = st.sidebar.number_input("Cement CO₂ (Mt)", 0.0, 2_000.0, 50.0, 1.0)
    coal_co2   = st.sidebar.number_input("Coal CO₂ (Mt)",   0.0, 10_000.0, 1200.0, 10.0)
    oil_co2    = st.sidebar.number_input("Oil CO₂ (Mt)",    0.0, 10_000.0, 800.0, 10.0)
    gas_co2    = st.sidebar.number_input("Gas CO₂ (Mt)",    0.0, 10_000.0, 600.0, 10.0)
    flaring_co2= st.sidebar.number_input("Flaring CO₂ (Mt)",0.0, 1_000.0, 10.0, 1.0)
    methane = st.sidebar.number_input("Methane (Mt CO₂e)", 0.0, 5_000.0, 300.0, 10.0)
    nitrous_oxide = st.sidebar.number_input("Nitrous Oxide (Mt CO₂e)", 0.0, 1_000.0, 100.0, 5.0)

    if model is None or scaler is None or features is None:
        st.warning("Model assets not available — charts below still interactive.")
    else:
        if st.button("🔮 Predict CO₂ Emissions", type="primary"):
            try:
                X = prepare_input_data(
                    country, year, population, gdp, energy_per_capita,
                    primary_energy_consumption, cement_co2, coal_co2, oil_co2,
                    gas_co2, flaring_co2, methane, nitrous_oxide, features
                )
                Xs = scaler.transform(X)
                pred = float(model.predict(Xs)[0])  # Mt
                per_capita_t = (pred * 1e6) / max(population,1) / 1e3
                vs_avg = 4.8
                delta = per_capita_t - vs_avg
                comp = "above" if delta>0 else "below"

                st.markdown(f"""
<div class="card">
  <b>🎯 Predicted CO₂ Emissions</b><br>
  <span style="font-size:1.6rem">{pred:,.2f} Mt</span><br>
  <small>Per capita: {per_capita_t:,.2f} t/person ({abs(delta):.1f} t {comp} global avg ~{vs_avg} t)</small><br>
  <small>Country used for features: {country} | Year: {year}</small>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    source_breakdown_charts(coal_co2, oil_co2, gas_co2, cement_co2, flaring_co2)
    continent_stats(data)

def ev_benefits_section():
    st.markdown('<div class="section-title">🚗 EV Benefits & Savings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 40000, 12000, 500)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 12.0, 3.50, 0.10)
        mpg = st.number_input("Your car's MPG (gasoline)", 10, 100, 30, 1)
        elec_rate = st.number_input("Electricity rate (¢/kWh)", 4, 80, 12, 1)
        ev_eff = st.slider("EV efficiency (mi/kWh)", 2.5, 6.0, 3.5, 0.1)
        maint_gas = st.slider("Yearly maintenance (Gasoline) $", 200, 2000, 900, 50)
        maint_ev  = st.slider("Yearly maintenance (EV) $", 50, 1500, 400, 50)
        years = st.slider("Ownership years", 1, 10, 5, 1)
    with c2:
        annual_gas_cost = (annual_miles / mpg) * gas_price
        annual_elec_cost = (annual_miles / ev_eff) * (elec_rate / 100.0)
        annual_savings = annual_gas_cost - annual_elec_cost
        st.markdown(f"""
<div class="kpi">
  <h3 class="big">${annual_savings:,.0f}</h3>
  <small>Annual savings</small><br>
  <small>Gas fuel: ${annual_gas_cost:,.0f} | Electric: ${annual_elec_cost:,.0f}</small><br>
  <small>Maint (Gas): ${maint_gas:,.0f} | Maint (EV): ${maint_ev:,.0f}</small><br>
  <small>Total {years}-yr: ${years*(annual_gas_cost+maint_gas) - years*(annual_elec_cost+maint_ev):,.0f}</small>
</div>
""", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        pie = px.pie(names=["Gasoline (fuel+maint)", "EV (energy+maint)"],
                     values=[annual_gas_cost+maint_gas, annual_elec_cost+maint_ev],
                     hole=.35, title="Annual running cost split")
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(pie, use_container_width=True)
    with d2:
        yrs = np.arange(1, years+1)
        gas_total = yrs * (annual_gas_cost + maint_gas)
        ev_total  = yrs * (annual_elec_cost + maint_ev)
        df = pd.DataFrame({"Year": yrs, "Gasoline Total ($)": gas_total, "EV Total ($)": ev_total})
        line = px.line(df, x="Year", y=["Gasoline Total ($)", "EV Total ($)"], markers=True,
                       title=f"Cumulative cost over {years} years")
        line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(line, use_container_width=True)

def env_impact_section():
    st.markdown('<div class="section-title">🌱 Environmental Impact</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
  <b>Why this matters</b>
  <ul>
    <li>EVs remove tailpipe emissions (NOx, PM) and cut lifecycle CO₂ as grids get cleaner.</li>
    <li>Electric drivetrains are ~3× more energy-efficient than ICE vehicles.</li>
    <li>Battery impacts are front-loaded; typically offset during use. Recycling & second-life are scaling fast.</li>
  </ul>
  <b>Actions you can take</b>
  <ul>
    <li>Prefer transit, cycling, or walking for short trips; eco-drive to save energy.</li>
    <li>Choose efficient vehicles and charge off-peak or via renewables when possible.</li>
    <li>Support public charging & clean-power procurement in your community.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

def ev_stats_section():
    st.markdown('<div class="section-title">📊 EV Market Statistics</div>', unsafe_allow_html=True)
    years = list(range(2015, 2024))
    global_ev_sales = [0.4, 0.7, 1.2, 2.0, 2.2, 3.1, 6.6, 10.5, 14.1]  # in millions
    fig = px.line(x=years, y=global_ev_sales, markers=True,
                  title="Global Electric Vehicle Sales (2015–2023)",
                  labels={"x":"Year","y":"Sales (Millions)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)", height=430)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# App Body (single-page, no separate landing)
# =========================
hero_and_stats()
predictor_section()
ev_benefits_section()
env_impact_section()
ev_stats_section()

# Always render floating chat
render_floating_chat()
