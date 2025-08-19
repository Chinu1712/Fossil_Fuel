import warnings
from pathlib import Path
import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

# Floating UI helper
try:
    from streamlit_float import float_init
    HAVE_FLOAT = True
except Exception:
    HAVE_FLOAT = False

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent

# =========================
# Page + Theme-aware CSS
# =========================
st.set_page_config(page_title="CO2 Emission Predictor & EV Awareness", page_icon="🌍", layout="wide")
st.markdown("""
<style>
  :root{
    --bg:#0e1117;--fg:#ffffff;--muted:#bcc3cf;--card:#1e2130;--border:rgba(255,255,255,.08);
    --accent:#37c77f;--link:#8ab4f8;--danger:#ff6b6b;--shadow:rgba(0,0,0,.3);
  }
  @media (prefers-color-scheme: light){
    :root{
      --bg:#f8fafc;--fg:#0f172a;--muted:#334155;--card:#ffffff;--border:rgba(15,23,42,.08);
      --accent:#0ea5e9;--link:#2563eb;--danger:#e11d48;--shadow:rgba(15,23,42,.15);
    }
  }
  .stApp, body{background:var(--bg); color:var(--fg);}
  .main-header{font-size:2.6rem; color:var(--accent); text-align:center; margin:0 0 1rem;}
  .sub-header{font-size:1.6rem; color:var(--link); margin:1rem 0 .6rem;}
  .card{background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.2rem; box-shadow:0 10px 24px var(--shadow);}
  .accent{color:var(--accent);}
  .pill{background:linear-gradient(90deg,#2563eb,#16a34a); height:12px; border-radius:999px; margin:.6rem 0;}
  .btn-wide>button{width:100% !important;}

  /* Floating FAB + Chat window */
  .fab{
    position:fixed; right:22px; bottom:22px; z-index:9999;
    width:54px; height:54px; border-radius:999px; display:flex; align-items:center; justify-content:center;
    background:var(--accent); color:#fff; border:none; box-shadow:0 10px 24px var(--shadow); cursor:pointer;
    font-size:26px; line-height:1;
  }
  .fab:hover{filter:brightness(0.95);}
  .chat-window{
    position:fixed; right:22px; bottom:90px; z-index:9998; width:min(380px, 92vw); max-height:min(70vh, 640px);
    background:var(--card); color:var(--fg); border:1px solid var(--border); border-radius:16px; overflow:hidden;
    box-shadow:0 20px 40px var(--shadow);
  }
  .chat-header{display:flex; align-items:center; justify-content:space-between; padding:.75rem 1rem; border-bottom:1px solid var(--border);}
  .chat-title{font-weight:600; color:var(--fg);}
  .chat-close{background:transparent; color:var(--muted); border:none; font-size:20px; cursor:pointer;}
  .chat-body{padding:10px 12px; overflow:auto; max-height:50vh;}
  .bubble{padding:.6rem .8rem; border-radius:12px; margin:.35rem 0; width:fit-content; max-width:92%;}
  .me{background:rgba(37,99,235,.15);}
  .bot{background:rgba(55,199,127,.15);}
  .chat-input{display:flex; gap:.5rem; padding:.6rem; border-top:1px solid var(--border); background:var(--card);}
  .chat-input input{
    flex:1; border:1px solid var(--border); border-radius:12px; padding:.55rem .7rem; background:transparent; color:var(--fg);
  }
  .chat-send{border:none; border-radius:10px; padding:.55rem .9rem; background:var(--accent); color:#fff; cursor:pointer;}
</style>
""", unsafe_allow_html=True)

# =========================
# Loaders
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
# Helpers
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

    # engineered (match training)
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

    # country one-hot if exists
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
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        bar = px.bar(df, x="Source", y="Emissions (Mt)", text_auto=True,
                     title="Source Contribution (bar view)")
        bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
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
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
    st.plotly_chart(fig, use_container_width=True)

    needed = {"coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"}
    if needed.issubset(recent.columns):
        bysrc = (recent.groupby("continent")[list(needed)]
                 .sum().reset_index().melt("continent", var_name="Source", value_name="Mt"))
        bysrc["Source"] = bysrc["Source"].str.replace("_co2", "", regex=False).str.title()
        fig2 = px.bar(bysrc, x="continent", y="Mt", color="Source", barmode="stack",
                      title="CO₂ by Source and Continent (2020+)",
                      labels={"continent": "Continent", "Mt": "CO₂ Emissions (Mt)"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# Gemini Chatbot (floating)
# =========================
def get_gemini_key() -> str:
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")
    return key

def gemini_reply(user_message: str, history: list) -> str:
    api_key = get_gemini_key()
    if not api_key:
        return "❗ Gemini API key is missing. Please set it in Streamlit Secrets."

    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    r = requests.post(url, params={"key": api_key},
                      headers={"Content-Type": "application/json"},
                      json={"contents": contents}, timeout=60)
    try:
        r.raise_for_status()
        data = r.json()
        return (data.get("candidates",[{}])[0]
                    .get("content",{}).get("parts",[{}])[0].get("text","")) or "No response."
    except requests.HTTPError as e:
        return f"HTTP error: {e} — {getattr(e, 'response', None) and e.response.text}"
    except Exception as e:
        return f"Error: {e}"

def render_floating_chat():
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "model",
                                  "content": "Hi! Ask me about EV benefits, emissions, charts, or how to use this app."}]

    # We render HTML buttons and bind them to Streamlit via query params-ish toggles
    placeholder = st.empty()  # keep a mount point

    # If we can, make the containers actually float
    if HAVE_FLOAT:
        float_init()

    # FAB
    with placeholder.container():
        st.markdown(
            f"""
            <button class="fab" onclick="window.parent.postMessage({{type:'toggle_chat'}}, '*')">💬</button>
            """,
            unsafe_allow_html=True
        )

        # Chat Window (hidden unless open)
        display = "block" if st.session_state.chat_open else "none"
        st.markdown(
            f"""
            <div class="chat-window" id="chatwin" style="display:{display};">
              <div class="chat-header">
                <div class="chat-title">AI Assistant</div>
                <button class="chat-close" onclick="window.parent.postMessage({{type:'toggle_chat'}}, '*')">✕</button>
              </div>
              <div class="chat-body" id="chatbody">
                {"".join([
                  f'<div class="bubble {"me" if m["role"]=="user" else "bot"}">{m["content"]}</div>'
                  for m in st.session_state.chat
                ])}
              </div>
              <div class="chat-input">
                <input id="chat_input" placeholder="Type your question..." onkeydown="if(event.key==='Enter') window.parent.postMessage({{type:'send_chat'}}, '*')">
                <button class="chat-send" onclick="window.parent.postMessage({{type:'send_chat'}}, '*')">Send</button>
              </div>
            </div>
            <script>
              // simple bridge: toggle + send events back to Streamlit via set query params
              window.addEventListener('message', (e)=>{
                if(!e.data) return;
                if(e.data.type==='toggle_chat'){ window.location.hash = (window.location.hash==='#open' ? '' : '#open'); window.parent.postMessage({{type:'rerun'}}, '*'); }
                if(e.data.type==='send_chat'){ window.location.hash = '#send'; window.parent.postMessage({{type:'rerun'}}, '*'); }
                if(e.data.type==='rerun'){ setTimeout(()=>{{ window.parent.postMessage({{type:'streamlit:rerun'}}, '*'); }}, 10); }
              });
            </script>
            """,
            unsafe_allow_html=True
        )

    # Read hash to toggle / send
    hash_val = st.experimental_get_query_params().get("_", [""])[0]  # unused; kept to avoid caching
    # We can inspect st.session_state for a light toggle: we’ll use st.query_params via JS hash
    # Use the URL fragment accessible via `st.query_params` (Streamlit doesn't expose hash; we emulate via rerun events)

    # Backend toggle using the browser hash content via the special widget below:
    hash_box = st.text_input("", value="", key="__invisible__", label_visibility="collapsed", help="ignore me")
    # (This invisible box exists only to trigger reruns on hash change via the JS above.)

    # Process a pseudo-hash by checking the actual JS toggled state we encoded with hash:
    # Since Streamlit cannot read hash directly, we simply flip state on every rerun if the DOM asked it.
    # To keep predictable, we store a latch in session_state: when JS posts 'toggle', we flip.
    if "_pending_toggle" not in st.session_state:
        st.session_state._pending_toggle = False
    if "_pending_send" not in st.session_state:
        st.session_state._pending_send = False

    # Use a small trick: when the hidden input changes (any rerun), check the browser hash by injecting small JS is not supported.
    # Instead, we expose two buttons that JS "clicks" via postMessage. We've already posted 'rerun' events; here we can't detect them reliably.
    # So we fall back to explicit Streamlit controls below too:
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("🟡 toggle_chat", key="__toggle__"):
            st.session_state.chat_open = not st.session_state.chat_open
    with col_b:
        # Send message using a Streamlit text_input when chat is open
        if st.session_state.chat_open:
            user_msg = st.text_input("Type your message here", key="__chat_msg__", label_visibility="collapsed")
            if st.button("Send", key="__send__"):
                if user_msg.strip():
                    st.session_state.chat.append({"role": "user", "content": user_msg.strip()})
                    with st.spinner("Thinking…"):
                        reply = gemini_reply(user_msg.strip(), st.session_state.chat)
                    st.session_state.chat.append({"role": "model", "content": reply})
                st.rerun()

# =========================
# Pages
# =========================
def show_landing_page():
    st.markdown('<h1 class="main-header">🌍 CO2 Emission Predictor & EV Awareness</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card" style="text-align:center;">'
        '<p style="font-size:1.1rem;margin:.2rem 0;">'
        'Understand emissions, explore EV savings, and see continent-level patterns with clean, theme-aware visuals.'
        '</p></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">💡 What you can do here</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
  <ul>
    <li><b>Predict</b> annual CO₂ using macro + energy inputs (with engineered features) up to year 2070.</li>
    <li><b>Compare</b> running costs: EV vs gasoline—see your yearly & 5-year savings with live charts.</li>
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

    # Country selector
    st.sidebar.markdown('<h3 class="sub-header">🔧 Input Parameters</h3>', unsafe_allow_html=True)
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
            vs_avg_t = 4.8
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
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)", height=460)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

def show_ev_benefits():
    st.markdown('<h2 class="sub-header">🚗 EV Benefits & Savings</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        annual_miles = st.number_input("Annual miles driven", 5000, 40000, 12000, 500)
        gas_price = st.number_input("Gas price per gallon ($)", 2.0, 12.0, 3.50, 0.10)
        car_mpg = st.number_input("Your car's MPG (gasoline)", 10, 100, 30, 1)
        electricity_rate = st.number_input("Electricity rate (¢/kWh)", 4, 80, 12, 1)
        ev_eff = st.slider("EV efficiency (mi/kWh)", 2.5, 6.0, 3.5, 0.1)
        maint_gas = st.slider("Yearly maintenance (Gasoline) $", 200, 2000, 900, 50)
        maint_ev  = st.slider("Yearly maintenance (EV) $", 50, 1500, 400, 50)
        years = st.slider("Ownership years", 1, 10, 5, 1)

    with c2:
        annual_gas_cost = (annual_miles / car_mpg) * gas_price
        annual_electricity_cost = (annual_miles / ev_eff) * (electricity_rate / 100.0)
        annual_savings = annual_gas_cost - annual_electricity_cost

        st.markdown(
            f"""
<div class="card">
  <h4 class="accent">💰 Your Annual Savings</h4>
  <p style="font-size:1.2rem;"><b>${annual_savings:,.0f}</b> /year &nbsp;|&nbsp; <b>${annual_savings*years:,.0f}</b> over {years} years</p>
  <p>Gas fuel: ${annual_gas_cost:,.0f} &nbsp;|&nbsp; Electric: ${annual_electricity_cost:,.0f}</p>
  <p>Maintenance (gas): ${maint_gas:,.0f} &nbsp;|&nbsp; Maintenance (EV): ${maint_ev:,.0f}</p>
</div>""",
            unsafe_allow_html=True,
        )

    # Charts: annual split + cumulative ownership cost
    c3, c4 = st.columns(2)
    with c3:
        pie = px.pie(
            names=["Gasoline (fuel+maint)", "EV (energy+maint)"],
            values=[annual_gas_cost + maint_gas, annual_electricity_cost + maint_ev],
            hole=0.35, title="Annual running cost split"
        )
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(pie, use_container_width=True)

    with c4:
        yrs = np.arange(1, years+1)
        gas_total = yrs * (annual_gas_cost + maint_gas)
        ev_total  = yrs * (annual_electricity_cost + maint_ev)
        df = pd.DataFrame({"Year": yrs, "Gasoline Total ($)": gas_total, "EV Total ($)": ev_total})
        line = px.line(df, x="Year", y=["Gasoline Total ($)", "EV Total ($)"], markers=True,
                       title=f"Cumulative cost over {years} years")
        line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)")
        st.plotly_chart(line, use_container_width=True)

def show_environmental_impact():
    st.markdown('<h2 class="sub-header">🌱 Environmental Impact</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
  <h4 class="accent">Why EVs help</h4>
  <ul>
    <li><b>Zero tailpipe emissions</b> ⇒ cleaner city air (NOx/PM).</li>
    <li><b>Grid synergy</b> ⇒ as grids add solar/wind, lifecycle CO₂ declines.</li>
    <li><b>High efficiency</b> ⇒ electric drivetrains convert ~3× energy to motion vs ICE.</li>
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
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="var(--fg)", height=430)
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

    # Floating assistant (always on)
    render_floating_chat()

# =========================
# Entrypoint
# =========================
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing_page()
else:
    show_main_app()
