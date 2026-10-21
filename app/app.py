"""
VeriFact - Streamlit Web Dashboard with Explainable AI (LIME)
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from src.predict import NewsPredictor

# Page Configuration
st.set_page_config(
    page_title="VeriFact | Fake News & Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .result-suspicious {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.05));
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .result-reliable {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(21, 128, 61, 0.05));
        border-left: 6px solid #22C55E;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .highlight-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    .footer-text {
        text-align: center;
        color: #64748B;
        margin-top: 3rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    model_path = os.path.join(BASE_DIR, "models", "model_pipeline.pkl")
    return NewsPredictor(model_path=model_path)

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}. Please ensure `python -m src.train` has been executed.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About Project")
    st.markdown("""
    **VeriFact** is an NLP-driven misinformation verification system specifically optimized for viral **WhatsApp forwards**, SMS scams, and unverified digital news.
    
    - **Student Author:** [Khushi Yadav](https://github.com/khushiyadav35)
    - **Academic Level:** B.Tech Final Year Capstone
    - **Core Tech:** TF-IDF n-grams, Calibrated Classification, LIME (XAI), Streamlit
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ LIME XAI Settings")
    num_features = st.slider("Number of Explanatory Features", min_value=4, max_value=12, value=8, step=1)
    
    st.markdown("---")
    st.markdown("### 🛡️ WhatsApp Verification Checklist")
    st.markdown("""
    Before clicking **Forward** on messaging apps:
    - ⚠️ Look for urgency words (*'share immediately'*, *'before midnight'*)
    - 🔎 Check for official government PIB/WHO confirmation
    - 🔗 Avoid unverified short links (`bit.ly`, `tinyurl`)
    - ❌ Be wary of sensational claims without named sources
    """)
    
    st.markdown("---")
    st.markdown("[📁 View GitHub Repository](https://github.com/khushiyadav35/fake-news-detector)")

# Main Header
st.markdown('<div class="main-header">🔍 VeriFact: WhatsApp Misinformation & Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify news and message forwards as <b>Reliable</b> or <b>Suspicious</b>, powered by Explainable AI (LIME) word attribution.</div>', unsafe_allow_html=True)

# Preset Benchmark Examples
st.markdown("##### ⚡ Quick Test Benchmark Samples:")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

sample_text = ""

with col1:
    if st.button("🚨 WhatsApp Medical Hoax", use_container_width=True):
        st.session_state["input_text"] = "URGENT ALERT: WHO and UNESCO have declared that drinking hot water with lemon and baking soda kills 100% of coronavirus and all viral flu immediately! Doctor from AIIMS confirms! Share with every family group before midnight to save lives!!!"
with col2:
    if st.button("🚨 Phishing / Bank Scam", use_container_width=True):
        st.session_state["input_text"] = "Dear customer, your SBI / HDFC bank account will be blocked within 24 hours due to non-updation of PAN and KYC documents. Click here immediately to update your details and avoid Rs 10,000 fine: http://bit.ly/sbi-kyc-verify-urgent"
with col3:
    if st.button("🚨 Cultural Award Rumor", use_container_width=True):
        st.session_state["input_text"] = "PROUD MOMENT FOR INDIA: UNESCO has just officially declared Jana Gana Mana as the BEST NATIONAL ANTHEM in the entire world! No other country's anthem matched its scientific composition. Every true Indian must forward this to 10 groups right now!"
with col4:
    if st.button("📰 Verified ISRO Mission", use_container_width=True):
        st.session_state["input_text"] = "The Indian Space Research Organisation (ISRO) successfully launched the NVS-02 navigation satellite aboard the GSLV-F12 rocket from the Satish Dhawan Space Centre in Sriharikota on Wednesday morning, mission directors confirmed in an official statement."
with col5:
    if st.button("📰 WHO Health Guideline", use_container_width=True):
        st.session_state["input_text"] = "The World Health Organization (WHO) published an updated guideline on dietary sodium intake, recommending that adults consume less than 2,000 milligrams of sodium per day to lower the risk of cardiovascular diseases and stroke."
with col6:
    if st.button("📰 Official RBI Advisory", use_container_width=True):
        st.session_state["input_text"] = "The Reserve Bank of India (RBI) issued a notification advising bank account holders to remain vigilant against phishing links and cautioned that the central bank never solicits personal banking credentials, passwords, or OTPs through SMS or phone calls."

# Text Area Input
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

user_input = st.text_area(
    "Paste news headline, WhatsApp forward, or message to analyze:",
    value=st.session_state["input_text"],
    height=140,
    placeholder="Type or paste message here (e.g., 'URGENT: Drinking hot water with lemon cures all diseases...')"
)

analyze_btn = st.button("🔎 Analyze Message & Explain", type="primary", use_container_width=True)

if analyze_btn or (user_input and len(user_input.strip()) > 10):
    text_to_analyze = user_input.strip()
    
    if len(text_to_analyze) < 10:
        st.warning("Please enter a longer message (at least 10 characters) for accurate linguistic analysis.")
    else:
        with st.spinner("Analyzing message patterns and generating LIME explanations..."):
            result = predictor.predict(text_to_analyze, explain=True)
            
        prediction = result["prediction"]
        confidence = result["confidence"]
        signals = result["linguistic_signals"]
        explanation = result["explanation"]
        
        st.markdown("---")
        
        # Classification Banner
        if prediction == "Suspicious":
            st.markdown(f"""
            <div class="result-suspicious">
                <h3 style="margin: 0; color: #F87171;">⚠️ POTENTIAL MISINFORMATION / SUSPICIOUS FORWARD DETECTED</h3>
                <p style="margin: 0.5rem 0 0 0; color: #FCA5A5; font-size: 1.05rem;">
                    The model flagged this message as <b>Suspicious</b> with <b>{confidence:.1f}% confidence</b>. It exhibits typical characteristics of viral misinformation, unverified claims, or urgency triggers.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-reliable">
                <h3 style="margin: 0; color: #4ADE80;">✅ VERIFIED / LIKELY RELIABLE INFORMATION</h3>
                <p style="margin: 0.5rem 0 0 0; color: #86EFAC; font-size: 1.05rem;">
                    The model classified this message as <b>Reliable</b> with <b>{confidence:.1f}% confidence</b>. It matches factual reporting tone and formal press communication structures.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Model Confidence", f"{confidence:.1f}%")
        with mcol2:
            st.metric("Linguistic Risk Index", f"{signals['risk_index']}/100")
        with mcol3:
            st.metric("Urgency Triggers", f"{signals['urgency_score']}")
        with mcol4:
            st.metric("Capitalization Ratio", f"{signals['uppercase_ratio'] * 100:.1f}%")
            
        st.markdown("---")
        
        # Explainable AI (LIME) Section
        st.markdown("### 🧠 Explainable AI (LIME): Feature Attribution")
        st.markdown("""
        **LIME** highlights words that pushed the prediction toward **Suspicious** (red) or toward **Reliable** (green).
        This guarantees transparency rather than an unexplainable black-box prediction.
        """)
        
        # In-line Highlighting
        st.markdown("##### 📝 In-line Highlighted Text View:")
        st.markdown(f'<div class="highlight-box">{explanation["html_highlighted"]}</div>', unsafe_allow_html=True)
        st.caption("Hover over highlighted words to inspect their attribution score. Red indicates words that made the message seem suspicious; Green indicates factual/neutral words.")
        
        # Feature Importance Chart
        st.markdown("##### 📊 Top Word Contribution Weights (LIME):")
        
        chart_data = []
        for word, weight in explanation["all_features"]:
            chart_data.append({
                "Word": str(word),
                "Weight": float(weight),
                "Direction": "Pushes to Suspicious" if weight > 0 else "Pushes to Reliable"
            })
            
        chart_df = pd.DataFrame(chart_data)
        chart_df = chart_df.sort_values(by="Weight", ascending=True)
        
        # Plot with Altair
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Weight:Q", title="LIME Weight (Negative = Reliable | Positive = Suspicious)"),
            y=alt.Y("Word:N", sort=None, title="Influential Word"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Pushes to Reliable", "Pushes to Suspicious"],
                    range=["#22C55E", "#EF4444"]
                ),
                legend=alt.Legend(title="Influence Direction")
            ),
            tooltip=["Word", "Weight", "Direction"]
        ).properties(height=320)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Actionable Advice
        st.markdown("### 📋 Verification Summary & Advice")
        if prediction == "Suspicious":
            st.error("""
            **Recommendation: DO NOT FORWARD THIS MESSAGE.**
            - The message contains sensationalized claims or artificial urgency designed to bypass critical thinking.
            - Verify with fact-checking portals like **PIB Fact Check**, **AltNews**, or official government websites before sharing.
            """)
        else:
            st.success("""
            **Recommendation: Information appears factual and credible.**
            - The phrasing aligns with verified institutional or news agency reporting standards.
            - Always double check the direct source or official domain when making sensitive financial or medical decisions.
            """)

# Footer
st.markdown("""
<div class="footer-text">
    VeriFact &bull; B.Tech Computer Science & AI-ML Capstone Project &bull; Built with Python, Scikit-learn, LIME, and Streamlit &bull; 
    <a href="https://github.com/khushiyadav35/fake-news-detector" target="_blank" style="color: #38BDF8;">GitHub: khushiyadav35</a>
</div>
""", unsafe_allow_html=True)
