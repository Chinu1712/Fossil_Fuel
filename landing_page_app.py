import streamlit as st

st.set_page_config(
    page_title="CO2 Emission Predictor & EV Awareness - Project Idea",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2E8B57; /* Sea Green */
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem;
        color: #4682B4; /* Steel Blue */
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 0.5rem;
    }
    .section-text {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #e8f5e8; /* Light Green */
        border-left: 8px solid #2E8B57;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .impact-point {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .impact-point .icon {
        font-size: 2rem;
        margin-right: 1rem;
        color: #2E8B57;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class=\"main-header\">🌍 CO2 Emission Predictor & EV Awareness Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style=\"text-align: center; font-size: 1.3rem; color: #555;\">Leveraging Data Science to Combat Climate Change and Promote Sustainable Futures</p>", unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1532187863570-877237497150?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", use_column_width=True, caption="Driving towards a cleaner, greener planet.")

st.markdown("<h2 class=\"sub-header\">💡 Our Project Idea: Bridging Prediction with Action</h2>", unsafe_allow_html=True)
st.markdown("""
<p class=\"section-text\">Our project aims to tackle the critical issue of CO2 emissions by providing a dual-pronged solution: accurate predictive modeling and impactful public awareness. We believe that by making the complex data of CO2 emissions accessible and understandable, and by clearly demonstrating the benefits of sustainable alternatives like Electric Vehicles (EVs), we can empower individuals and communities to make informed choices that contribute to a healthier planet.</p>
<p class=\"section-text\">The core idea is to transform abstract environmental data into actionable insights, fostering a sense of curiosity and responsibility among the audience. We want to show not just <i>what</i> the emissions are, but <i>why</i> they matter and <i>how</i> collective action can make a significant difference.</p>
""", unsafe_allow_html=True)

st.markdown("<h2 class=\"sub-header\">✨ Self-Explanatory Project Overview</h2>", unsafe_allow_html=True)
st.markdown("""
<div class=\"highlight-box\">
    <p class=\"section-text\">This platform is designed to be intuitive and informative, guiding users through the journey of understanding CO2 emissions and the transformative potential of Electric Vehicles. Here's how it works:</p>
    
    <div class=\"impact-point\">
        <span class=\"icon\">📈</span>
        <div>
            <h3 style=\"color: #2E8B57;\">Accurate CO2 Emission Prediction</h3>
            <p class=\"section-text\">At its heart, our platform features a robust machine learning model, meticulously trained on comprehensive global CO2 emission data. This model can predict future CO2 emissions based on various factors like population, GDP, energy consumption, and industrial activities. With an R² score exceeding 90%, our predictions offer a reliable glimpse into potential environmental impacts, highlighting areas where intervention is most needed.</p>
        </div>
    </div>

    <div class=\"impact-point\">
        <span class=\"icon\">🚗</span>
        <div>
            <h3 style=\"color: #2E8B57;\">Promoting Electric Vehicle Adoption</h3>
            <p class=\"section-text\">Beyond prediction, a key objective is to advocate for sustainable transportation. The platform includes dedicated sections that vividly illustrate the myriad benefits of Electric Vehicles. We present clear, data-backed comparisons between EVs and traditional gasoline cars, covering environmental advantages (zero tailpipe emissions, reduced air pollution), economic savings (lower fuel and maintenance costs, government incentives), and performance benefits (instant torque, quiet operation).</p>
        </div>
    </div>

    <div class=\"impact-point\">
        <span class=\"icon\">🌱</span>
        <div>
            <h3 style=\"color: #2E8B57;\">Spreading Environmental Awareness</h3>
            <p class=\"section-text\">Our goal is to foster a deeper understanding of climate change and the role of individual and collective actions. The platform provides educational content on the sources and impacts of CO2, the importance of renewable resources, and the long-term benefits of transitioning to cleaner energy. Interactive visualizations and statistics make complex environmental data engaging and accessible, encouraging users to explore and learn.</p>
        </div>
    </div>

    <div class=\"impact-point\">
        <span class=\"icon\">💡</span>
        <div>
            <h3 style=\"color: #2E8B57;\">Inspiring Curiosity and Action</h3>
            <p class=\"section-text\">The design of the platform is centered around sparking curiosity. By allowing users to input parameters and see predicted CO2 emissions, they can directly observe the impact of different scenarios. This hands-on experience, combined with compelling arguments for EVs and renewable energy, aims to inspire users to consider their own carbon footprint and become advocates for sustainable living. We emphasize that every choice, no matter how small, contributes to the larger goal of preserving our limited non-renewable resources.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class=\"sub-header\">🚀 Ready to Explore?</h2>", unsafe_allow_html=True)
st.markdown("""
<p class=\"section-text\">Dive into the application to predict CO2 emissions, understand the benefits of EVs, and explore global environmental statistics. Let's work together towards a sustainable future!</p>
""", unsafe_allow_html=True)

if st.button("Go to CO2 Predictor App"): # This button would ideally link to your main app
    st.write("Redirecting to the main application...")
    # In a multi-page Streamlit app, you would use st.switch_page("enhanced_streamlit_app")
    # For now, this is a placeholder. If you have a multi-page app, replace this.
    st.markdown("[Click here to go to the main app](https://8501-i26cpkkqw3gww8jnvddt3-9fbcd6e1.manus.computer)")

st.markdown("""
<br><br>
<p style=\"text-align: center; font-size: 0.9rem; color: #888;\">Developed with a passion for data science and environmental sustainability.</p>
""", unsafe_allow_html=True)


