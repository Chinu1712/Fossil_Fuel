import streamlit as st
import google.generativeai as genai

# --- Context ---
PROJECT_CONTEXT = """
You are a chatbot designed to answer questions about the "Fossil Fuel Countdown: The Race to EV & Renewables" project.
This project is an interactive dashboard that shows how fast our petrol and diesel reserves are running out and how much we can extend the timeline by switching to EVs and renewable sources today.
... (keep full context you pasted above) ...
"""

# --- Setup Gemini ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def _gemini_reply(prompt, history):
    # include PROJECT_CONTEXT always
    conversation = PROJECT_CONTEXT + "\n\n"
    for turn in history:
        conversation += f"{turn['role'].upper()}: {turn['content']}\n"
    conversation += f"USER: {prompt}\nMODEL:"
    try:
        resp = model.generate_content(conversation)
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Floating Chat UI ---
def render_floating_chat():
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "model", "content": "Hi! I am a chatbot here to assist you about the Fossil Fuel Countdown project 🚗⚡."}
        ]

    # Floating button
    fab = st.container()
    with fab:
        st.markdown(
            """
            <style>
            #chat_fab {
                position: fixed; right: 22px; bottom: 22px; z-index: 9999;
                width: 60px; height: 60px; border-radius: 50%;
                background: linear-gradient(145deg, #16c5ff, #ff2e7e);
                color: white; font-size: 28px; border:none;
                display:flex; align-items:center; justify-content:center;
                cursor:pointer; box-shadow:0 8px 20px rgba(0,0,0,.3);
            }
            </style>
            <button id="chat_fab">💬</button>
            """,
            unsafe_allow_html=True
        )
        # Tiny JS to toggle session_state (safe because just triggers rerun)
        if st.button("Toggle Chat", key="fab_toggle"):
            st.session_state.chat_open = not st.session_state.chat_open

    # Chat window
    if st.session_state.chat_open:
        with st.container():
            st.markdown(
                """
                <div style="
                    position: fixed; right: 22px; bottom: 92px; width: 360px;
                    max-height: 70vh; background: rgba(30,30,40,.95);
                    border-radius: 16px; padding: 12px; overflow-y:auto;
                    z-index: 9998; box-shadow:0 12px 32px rgba(0,0,0,.45);
                ">
                """,
                unsafe_allow_html=True
            )

            # Messages
            for msg in st.session_state.chat:
                align = "flex-end" if msg["role"] == "user" else "flex-start"
                bubble_color = "#16c5ff" if msg["role"] == "user" else "#444"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:{align}; margin:4px 0;">
                        <div style="background:{bubble_color}; color:white; padding:8px 12px;
                                    border-radius:14px; max-width:75%;">{msg['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Input bar
            prompt = st.chat_input("Ask about fossil fuels, EVs, or CO₂...")
            if prompt:
                st.session_state.chat.append({"role": "user", "content": prompt})
                reply = _gemini_reply(prompt, st.session_state.chat)
                st.session_state.chat.append({"role": "model", "content": reply})

            st.markdown("</div>", unsafe_allow_html=True)


# --- Main Tabs ---
st.set_page_config(layout="wide", page_title="Fossil Fuel Countdown")

tabs = st.tabs(["Global", "Countries", "Predictor", "EV Impact"])  # ✅ removed "Continents"

# Example of usage
with tabs[0]:
    st.header("Global Fossil Fuel Countdown")
    st.write("Global stats go here...")

# Render chatbot on all pages
render_floating_chat()
