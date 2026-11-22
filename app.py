import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# EV BATTERY INTELLIGENCE | Premium enterprise Streamlit app
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_soc_prediction_model.joblib"
METADATA_PATH = BASE_DIR / "project_metadata.json"
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize persistent prediction state before any page uses it.
if "prediction_output" not in st.session_state:
    st.session_state["prediction_output"] = None
if "prediction_source" not in st.session_state:
    st.session_state["prediction_source"] = None

# --------------------------- DESIGN SYSTEM ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --bg: #F4F7FB;
      --surface: #FFFFFF;
      --ink: #142238;
      --muted: #718096;
      --line: #E5EAF1;
      --navy: #12233D;
      --blue: #2864DC;
      --teal: #0F9D8A;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: var(--bg); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,251,0.88); }
    .block-container { max-width: 1500px; padding-top: 1.45rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
      background: #10213A;
      border-right: 1px solid #203754;
    }
    section[data-testid="stSidebar"] * { color: #EAF0F8; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #B7C5D8; }
    section[data-testid="stSidebar"] .stCaption { color: #91A5BF; }
    section[data-testid="stSidebar"] hr { border-color: #2A405E; }

    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom: 22px; }
    .brand-mark {
      width:42px; height:42px; border-radius:13px; display:flex; align-items:center;
      justify-content:center; background:linear-gradient(135deg,#2D6CDF,#18A999);
      color:white; font-size:22px; box-shadow:0 8px 18px rgba(38,100,220,.22);
    }
    .brand-name { font-family:Manrope,sans-serif; font-size:17px; font-weight:800; letter-spacing:.08em; color:#F7FAFF; }
    .brand-sub { color:#91A5BF; font-size:10px; letter-spacing:.16em; margin-top:2px; }

    .hero {
      background: linear-gradient(115deg, #122844 0%, #1C3D60 62%, #176C70 120%);
      border-radius:22px; padding:27px 30px; color:white; margin-bottom:22px;
      box-shadow:0 14px 35px rgba(19,42,70,.12);
      position:relative; overflow:hidden;
    }
    .hero:after {
      content:""; position:absolute; width:280px; height:280px; border-radius:50%;
      right:-75px; top:-145px; border:1px solid rgba(255,255,255,.14);
      box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025);
    }
    .eyebrow { color:#A9D6DC; font-size:10px; font-weight:700; letter-spacing:.19em; }
    .hero-title { font-family:Manrope,sans-serif; font-size:30px; font-weight:800; line-height:1.2; margin:8px 0; color:#fff; }
    .hero-copy { color:#D1DFEA; font-size:13px; max-width:720px; margin:0; }
    .hero-chip { display:inline-block; margin-top:17px; border:1px solid rgba(255,255,255,.2);
      background:rgba(255,255,255,.08); padding:6px 10px; border-radius:20px;
      color:#E8F5F6; font-size:11px; }

    .section-head { font-family:Manrope,sans-serif; font-size:18px; font-weight:800; color:#172A43; margin:8px 0 13px; }
    .section-sub { color:#78879A; font-size:12px; margin-top:-8px; margin-bottom:15px; }
    .kpi {
      background:#fff; border:1px solid var(--line); border-radius:16px; padding:17px 18px;
      box-shadow:0 4px 14px rgba(27,50,79,.035); min-height:112px;
