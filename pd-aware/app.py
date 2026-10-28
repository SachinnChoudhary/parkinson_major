"""
PD-Aware — AI-Assisted Parkinson's Disease Screening

Main application entry point.
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import APP_NAME, MEDICAL_DISCLAIMER
from database.engine import init_db
from components.auth_manager import login_form, init_session_state

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — Parkinson's Screening Support",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"**{APP_NAME}** — AI-Assisted Parkinson's Disease Screening Tool.\n\n"
            "This application provides screening information only and does not "
            "diagnose Parkinson's disease."
        ),
    },
)

# ─── Initialize Database ─────────────────────────────────────────────────────
init_db()
init_session_state()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2962A0 50%, #3B82F6 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(41, 98, 160, 0.2);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 300;
    }

    /* Card styling */
    .info-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .info-card h3 {
        color: #1E3A5F;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }

    .info-card p {
        color: #475569;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Step Badge */
    .step-badge {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: #EFF6FF;
        color: #2962A0;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid #BFDBFE;
        margin-bottom: 0.75rem;
    }

    /* Disclaimer bar */
    .disclaimer-bar {
        background: #FEF9C3;
        border: 1px solid #FCD34D;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.85rem;
        color: #92400E;
        line-height: 1.5;
    }

    .disclaimer-bar strong {
        color: #78350F;
    }

    /* Hide Streamlit hamburger and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    /* Progress bar colors */
    .stProgress > div > div {
        background-color: #2962A0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PD-Aware")
    st.caption("AI-Assisted Screening System")
    st.divider()
    login_form()
    st.divider()
    st.markdown(
        '<p style="font-size: 0.75rem; color: #94A3B8;">Version 1.0.0 • Clinical Screening Support</p>',
        unsafe_allow_html=True,
    )

# ─── Home Page Content ───────────────────────────────────────────────────────

# Hero Section
st.markdown("""
<div class="main-header">
    <h1>PD-Aware</h1>
    <p>AI-Assisted Parkinson's Disease Screening & Decision-Support System</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown(f"""
<div class="disclaimer-bar">
    <strong>Important Notice:</strong> {MEDICAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### What is PD-Aware?")
    st.markdown("""
    PD-Aware is an AI-assisted screening tool that helps identify potential indicators
    associated with Parkinson's disease using validated machine learning models. It is designed to support
    — not replace — clinical evaluation by a qualified healthcare professional.

    **What this tool can do:**
    - Analyze voice biomedical acoustic features using a verified machine learning model
    - Help users document and organize motor and non-motor symptoms
    - Generate a structured screening report for medical consultations
    - Provide evidence-based clinical information about Parkinson's disease

    **What this tool cannot do:**
    - Diagnose Parkinson's disease or any other medical condition
    - Replace a comprehensive neurological examination
    - Prescribe medication or treatment plans
    - Guarantee individual diagnostic certainty
    """)

with col2:
    st.markdown("### Quick Actions")

    if st.button("Start Screening", icon=":material/search:", use_container_width=True, type="primary"):
        st.switch_page("pages/02_Start_Screening.py")

    if st.button("Clinician Portal", icon=":material/medical_services:", use_container_width=True):
        st.switch_page("pages/05_Doctor_Dashboard.py")

    if st.button("How It Works", icon=":material/menu_book:", use_container_width=True):
        st.switch_page("pages/03_How_It_Works.py")

    if st.button("About Parkinson's", icon=":material/health_and_safety:", use_container_width=True):
        st.switch_page("pages/04_Understanding_Parkinsons.py")

    if st.button("Model Engineering Lab", icon=":material/science:", use_container_width=True):
        st.switch_page("pages/06_Model_Lab.py")

st.divider()

# How It Works Section
st.markdown("### Screening Methodology Overview")

cols = st.columns(4)

steps = [
    ("01", "Consent & Context", "Review screening objectives, understand limitations, and provide informed consent."),
    ("02", "Symptom Reporting", "Document observations in plain language or complete the structured questionnaire."),
    ("03", "Acoustic Analysis", "Optional analysis of vocal acoustic features using validated classifiers."),
    ("04", "Clinical Summary", "Receive an explained assessment with a downloadable PDF report for your physician."),
]

for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; min-height: 190px;">
            <div class="step-badge">{num}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Understanding the Model
st.markdown("### Architecture & Principles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>Dataset Provenance</h3>
        <p>Trained on the UCI Oxford Parkinson's Disease Detection Dataset — 195 voice
        recordings from 31 subjects analyzed for 22 micro-acoustic features.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>Leakage Prevention</h3>
        <p>Subject-aware GroupShuffleSplit prevents identity memorization leakage. Normalization
        scalers are fit strictly on training splits.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>Data Security & Privacy</h3>
        <p>Inference occurs locally without transmitting health information to third-party APIs.
        User data is stored strictly with local consent controls.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #94A3B8; font-size: 0.8rem;">
    PD-Aware is an AI-assisted screening and decision-support system. It is not a medical device
    and has not been clinically validated. Always consult a licensed physician or neurologist.
</div>
""", unsafe_allow_html=True)
