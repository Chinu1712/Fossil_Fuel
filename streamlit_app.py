# streamlit_app.py  — Fossil Fuel COUNTDOWN
# Full app: predictor, EV benefits, impact, stats + floating Gemini chat widget.

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

st.set_page_config(page_title="Fossil Fuel COUNTDOWN", page_icon="🛢️", layout="wide")

# =========================
# THEME & STYLES
# =========================
st.markdown("""
<style>
:root{
  --bg:#0a1422; --fg:#e9f4ff; --muted:#93b0c8; --card:#0f2133; --card2:#132941;
  --border:rgba(255,255,255,.10); --accent:#12d7ff; --accent2:#ff2e7e; --lime:#00ffb7;
  --shadow:0 18px 38px rgba(0,0,0,.35), 0 8px 18px rgba(0,0,0,.28);
  --zchat: 2147483000;
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

/* Floating chat */
#ff_fab{
  position:fixed; right:22px; bottom:22px; z-index:var(--zchat);
  width:60px; height:60px; border-radius:16px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(145deg, #16c5ff, #ff2e7e); color:#fff; border:none;
  box-shadow: 0 16px 34px rgba(0,0,0,.35), 0 8px 18px rgba(0,0,0,.2); cursor:pointer; font-size:26px;
}
#ff_fab:hover{ filter:brightness(.96) }

#ff_chat{
  position:fixed; right:22px; bottom:92px; z-index:calc(var(--zchat) - 1);
  width:min(560px, 92vw);
  background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.05));
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border:1px solid rgba(255,255,255,.16); border-radius:20px; overflow:hidden; display:none;
  box-shadow: var(--shadow);
}
.ff_header{ display:flex; align-items:center; justify-content:space-between; padding:12px 14px; border-bottom:1px solid rgba(255,255,255,.14); }
.ff_title{ font-weight:900; color:#f3f9ff; }
.ff_close{ background:transparent; color:#f3f9ff; opacity:.8; border:none; font-size:20px; cursor:pointer; }
.ff_body{ padding:14px; overflow:auto; max-height:56vh; }

.bubble{ padding:.7rem .9rem; border-radius:16px; margin:.45rem 0; width:fit-content; max-width:86%; box-shadow:0 5px 16px rgba(0,0,0,.2); }
.me{ background: linear-gradient(145deg, rgba(22,197,255,.85), rgba(22,197,255,.55)); color:#052033; margin-left:auto; }
.bot{ background: linear-gradient(145deg, rgba(255,255,255,.22), rgba(255,255,255,.16)); color:#f7fbff; border:1px solid rgba(255,255,255,.15); }

.ff_input{ display:flex; gap:.6rem; padding:12px; border-top:1px solid rgba(255,255,255,.14); background:rgba(0,0,0,.06); }
.ff_input textarea{
  flex:1; border:1px solid rgba(255,255,255,.22); border-radius:14px; padding:.6rem .75rem;
  background: rgba(255,255,255,.08); color:var(--fg); height:64px; resize:vertical;
}
.ff_send{
  border:none; border-radius:14px; padding:.6rem 1rem; font-weight:700; color:#0b1b2b;
  background:linear-gradient(90deg, #00ffd0, #12d7ff); cursor:pointer; box-shadow:0 10px 20px rgba(0,0,0,.25);
}
.ff_send:hover{ filter:brightness(.95) }
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

def continent_stats(df):
    if df is None or not {"continent","co2"}.issubset(df.columns):
        st.info("Continental data unavailable.")
        return
    recent = df[(df["year"]>=2020) & df["continent"].notna()]
    cont = recent.groupby("continent", as_index=False)["co2"].sum().sort_values("co2", ascending=False)
    fig = px.bar(cont, x="continent", y="co2", text_auto=True,
                 title="Total CO₂ by Continent (2020+)",
                 labels={"continent":"Continent","co2":"CO₂ Emissions (Mt)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    needed={"coal_co2","oil_co2","gas_co2","cement_co2","flaring_co2"}
    if needed.issubset(recent.columns):
        bysrc=(recent.groupby("continent")[list(needed)]
               .sum().reset_index().melt("continent", var_name="Source", value_name="Mt"))
        bysrc["Source"]=bysrc["Source"].str.replace("_co2","",regex=False).str.title()
        fig2=px.bar(bysrc, x="continent", y="Mt", color="Source", barmode="stack",
                    title="CO₂ by Source & Continent (2020+)", labels={"Mt":"CO₂ Emissions (Mt)"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# GEMINI CHAT BACKEND
# =========================
def _get_gemini_key()->str:
    key = st.secrets.get("GEMINI_API_KEY","")
    if not key: key = os.getenv("GEMINI_API_KEY","")
    return key

def _gemini_reply(user_message:str, history:list)->str:
    key=_get_gemini_key()
    if not key:
        return "❗ Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets."
    contents=[]
    for m in history:
        role = "user" if m["role"]=="user" else "model"
        contents.append({"role":role,"parts":[{"text":m["content"]}]})
    contents.append({"role":"user","parts":[{"text":user_message}]})
    url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    r=requests.post(url, params={"key":key}, headers={"Content-Type":"application/json"},
                    json={"contents":contents}, timeout=60)
    try:
        r.raise_for_status()
        data=r.json()
        return data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","") or "No response."
    except requests.HTTPError as e:
        return f"HTTP error: {e} – {getattr(e,'response',None) and e.response.text}"
    except Exception as e:
        return f"Error: {e}"

def render_floating_chat():
    """Floating button + modal chat. No f-strings inside HTML/JS to avoid brace issues."""
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "model", "content": "Hi! Ask me about fossil fuels, EVs, or CO₂."}
        ]

    # Handle message from query param (?chatq=...) so Send works
    qp = dict(st.query_params)
    msg = qp.get("chatq")
    if msg:
        msg = msg.strip()
        if msg:
            st.session_state.chat.append({"role": "user", "content": msg})
            reply = _gemini_reply(msg, st.session_state.chat)
            st.session_state.chat.append({"role": "model", "content": reply})
        # Keep the panel open and clear chatq
        st.query_params["open"] = "1"
        if "chatq" in st.query_params:
            del st.query_params["chatq"]

    # Build bubbles safely
    def esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    bubbles = "".join(
        [
            '<div class="bubble {cls}">{txt}</div>'.format(
                cls=("me" if m["role"] == "user" else "bot"), txt=esc(m["content"])
            )
            for m in st.session_state.chat
        ]
    )

    # Plain triple-quoted string (NOT an f-string). We inject bubbles via placeholder.
    html = """
<button id="ff_fab" title="Ask">💬</button>

<div id="ff_chat">
  <div class="ff_header">
    <div class="ff_title">Fossil Fuel Chat</div>
    <button class="ff_close" id="ff_close">✕</button>
  </div>
  <div class="ff_body" id="ff_body">%%BUBBLES%%</div>
  <div class="ff_input">
    <textarea id="ff_input" placeholder="Ask me about fossil fuels, EVs, or CO₂!"></textarea>
    <button class="ff_send" id="ff_send">Send</button>
  </div>
</div>

<script>
(function(){
  const fab = document.getElementById("ff_fab");
  const box = document.getElementById("ff_chat");
  const closeBtn = document.getElementById("ff_close");
  const sendBtn = document.getElementById("ff_send");
  const input = document.getElementById("ff_input");
  const body = document.getElementById("ff_body");

  function openBox(){
    if(!box) return;
    box.style.display = "block";
    setTimeout(function(){
      if(body) body.scrollTop = body.scrollHeight;
    }, 50);
  }
  function closeBox(){
    if(!box) return;
    box.style.display = "none";
  }

  // Restore open state if ?open=1
  try {
    const sp = new URLSearchParams(window.location.search);
    if (sp.get("open") === "1") openBox();
  } catch(e) {}

  if (fab) fab.onclick = function(){
    if (box.style.display === "block"){
      closeBox();
      try {
        const u = new URL(window.location.href);
        u.searchParams.delete("open");
        window.history.replaceState({}, "", u.toString());
      } catch(e) {}
    } else {
      openBox();
      try {
        const u = new URL(window.location.href);
        u.searchParams.set("open", "1");
        window.history.replaceState({}, "", u.toString());
      } catch(e) {}
    }
  };

  if (closeBtn) closeBtn.onclick = function(){
    closeBox();
    try {
      const u = new URL(window.location.href);
      u.searchParams.delete("open");
      window.history.replaceState({}, "", u.toString());
    } catch(e) {}
  };

  function send(){
    const val = (input && input.value || "").trim();
    if (!val) return;
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("chatq", val);
      u.searchParams.set("open", "1");   // keep panel open after reload
      window.location.href = u.toString();
    } catch(e) {}
  }
  if (sendBtn) sendBtn.onclick = send;
  if (input) input.addEventListener("keydown", function(ev){
    if (ev.key === "Enter" && !ev.shiftKey){
      ev.preventDefault();
      send();
    }
  });
})();
</script>
"""
    st.markdown(html.replace("%%BUBBLES%%", bubbles), unsafe_allow_html=True)

# =========================
# HEADER + TABS
# =========================
st.markdown("""
<div class="header-wrap">
  <div class="title">Fossil Fuel COUNTDOWN</div>
  <div class="subtitle">Understand reserves and emissions — compare sources — and see how EVs change the trajectory.</div>
</div>
""", unsafe_allow_html=True)

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

    # Up to 2070 as requested
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
        st.warning("Model files not found; charts below still work.")
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

    st.markdown('<div class="section-title">Continents</div>', unsafe_allow_html=True)
    continent_stats(data)

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

# === Always render floating chat on top ===
render_floating_chat()
