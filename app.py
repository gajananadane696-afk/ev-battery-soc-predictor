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
    }
    .kpi-label { color:#718096; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .kpi-value { color:#172A43; font-family:Manrope,sans-serif; font-size:25px; font-weight:800; margin-top:9px; }
    .kpi-note { color:#94A0AF; font-size:10px; margin-top:3px; }
    .status-pill { display:inline-block; border-radius:20px; padding:5px 9px; font-size:10px; font-weight:700; letter-spacing:.05em; }
    .status-ok { background:#E6F6F1; color:#087B68; }
    .status-warn { background:#FFF4DE; color:#9A6500; }
    .panel {
      background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 19px;
      box-shadow:0 4px 14px rgba(27,50,79,.03);
    }
    .panel-title { color:#20344F; font-weight:700; font-size:13px; margin-bottom:4px; }
    .panel-caption { color:#8794A5; font-size:11px; margin-bottom:10px; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    div[data-testid="stFileUploader"] {
      background:#F8FAFD; border:1px dashed #B8C7D9; border-radius:14px; padding:12px;
    }
    .stButton button[kind="primary"], .stDownloadButton button {
      border-radius:10px; font-weight:700; min-height:42px;
    }
    .stButton button[kind="primary"] {
      background:#245FD2; border:1px solid #245FD2; color:#fff;
      box-shadow:0 5px 12px rgba(36,95,210,.15);
    }
    .stButton button[kind="primary"]:hover { background:#194EBA; border-color:#194EBA; }
    .stTabs [data-baseweb="tab-list"] { gap:7px; background:#EAF0F7; padding:5px; border-radius:12px; }
    .stTabs [data-baseweb="tab"] { border-radius:9px; color:#65758B; font-weight:600; }
    .stTabs [aria-selected="true"] { background:white !important; color:#1E4FA8 !important; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:13px 15px; border-radius:13px; }
    div[data-testid="stMetricLabel"] { color:#718096; }
    div[data-testid="stMetricValue"] { color:#172A43; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- HELPERS ---------------------------
def card(label, value, note="", accent="#245FD2"):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{accent}">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.info("Place the trained .joblib model in the same folder as app.py.")
        st.stop()
    if not METADATA_PATH.exists():
        st.error(f"Metadata file not found: {METADATA_PATH.name}")
        st.info("Place project_metadata.json in the same folder as app.py.")
        st.stop()
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        st.stop()

    cols = loaded_metadata.get("feature_columns", [])
    if not isinstance(cols, list) or not cols:
        st.error("project_metadata.json must contain a non-empty feature_columns list.")
        st.stop()
    return loaded_model, loaded_metadata, cols


def find_target_column(df, metadata):
    candidates = [
        metadata.get("target_column"),
        "SoC", "SOC", "SOC_C_3300", "\ufeffSOC_C_3300",
        "Actual_SoC", "Actual_SOC", "SoC (%)", "SOC (%)"
    ]
    normalized = {str(c).replace("\ufeff", "").strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate is not None:
            key = str(candidate).replace("\ufeff", "").strip().lower()
            if key in normalized:
                return normalized[key]
    return None


def numeric_frame(df, columns):
    out = df[columns].copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError("Missing or non-numeric values in: " + ", ".join(map(str, bad)))
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite or invalid numeric values.")
    return out


def plotly_layout(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#68788D", size=11),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E7ECF2"),
        yaxis=dict(showgrid=True, gridcolor="#EEF2F6", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">ϟ</div>
          <div>
            <div class="brand-name">VOLTEX</div>
            <div class="brand-sub">BATTERY INTELLIGENCE</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Workspace")
    page = st.radio(
        "Navigation",
        ["Overview", "AI Predictions", "Model Performance", "Dataset Explorer", "Project Details"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Model status")
    model, metadata, features = load_artifacts()
    model_name = metadata.get("best_model_by_test_RMSE") or metadata.get("best_model") or "Saved ML model"
    st.markdown('<span class="status-pill status-ok">● MODEL LOADED</span>', unsafe_allow_html=True)
    st.caption(f"Active estimator: {model_name}")
    st.caption(f"Required inputs: {len(features)} features")
    st.divider()
    st.caption("Academic research prototype · Offline CSV inference")

# --------------------------- TOP HEADER ---------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">ELECTRIC MOBILITY · COMPUTATIONAL INTELLIGENCE</div>
      <div class="hero-title">EV Battery Intelligence</div>
      <p class="hero-copy">A machine-learning workspace for battery State-of-Charge estimation, dataset exploration and model evaluation.</p>
      <span class="hero-chip">● Prediction engine ready</span>
      <span class="hero-chip">CSV-based inference</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- OVERVIEW ---------------------------
if page == "Overview":
    st.markdown('<div class="section-head">Workspace overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">System configuration and latest prediction activity.</div>', unsafe_allow_html=True)

    latest = st.session_state.prediction_output
    count = len(latest) if latest is not None else 0
    mean_soc = float(latest["Predicted_SoC"].mean()) if count else None
    k1, k2, k3, k4 = st.columns(4)
    with k1: card("Active algorithm", "Random Forest" if "random" in str(model_name).lower() else str(model_name)[:20], "Loaded trained estimator")
    with k2: card("Model inputs", str(len(features)), "Features expected by model", "#168A7A")
    with k3: card("Latest records", f"{count:,}", "Rows predicted in this session", "#6C63C8")
    with k4: card("Mean predicted SoC", f"{mean_soc:.2f}%" if mean_soc is not None else "—", "Available after prediction", "#168A7A")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Platform workflow</div><div class="panel-caption">From operating measurements to model inference</div><b>01 · Prepare</b><br><span style="color:#718096;font-size:12px">Use a CSV containing every required model feature.</span><br><br><b>02 · Validate</b><br><span style="color:#718096;font-size:12px">The application checks columns, numeric values and missing data.</span><br><br><b>03 · Predict</b><br><span style="color:#718096;font-size:12px">The saved preprocessing/model pipeline estimates SoC.</span><br><br><b>04 · Review</b><br><span style="color:#718096;font-size:12px">Explore results, charts and downloadable predictions.</span></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model configuration</div><div class="panel-caption">Loaded from project_metadata.json</div></div>', unsafe_allow_html=True)
        st.write("**Estimator**", model_name)
        st.write("**Target column**", metadata.get("target_column", "Not specified"))
        st.write("**Input feature count**", len(features))
        st.write("**Inference mode**", "Manual + CSV" if len(features) == 5 and set(features) == {"Voltage", "Current", "Power", "Temperature", "Time"} else "CSV feature matrix")
        st.info("This application runs predictions using your saved model. It does not generate live sensor readings.")

    if latest is not None:
        st.markdown('<div class="section-head">Latest prediction snapshot</div>', unsafe_allow_html=True)
        st.dataframe(latest.head(10), use_container_width=True, hide_index=True)

# --------------------------- AI PREDICTIONS ---------------------------
elif page == "AI Predictions":
    st.markdown('<div class="section-head">AI prediction workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Choose manual measurements when using a five-feature model, or upload a dataset matching the trained model schema.</div>', unsafe_allow_html=True)

    manual_allowed = set(features) == {"Voltage", "Current", "Power", "Temperature", "Time"} and len(features) == 5
    tab_names = ["CSV prediction"]
    if manual_allowed:
        tab_names.insert(0, "Single measurement")
    tabs = st.tabs(tab_names)

    if manual_allowed:
        with tabs[0]:
            st.markdown("#### Enter battery operating measurements")
            st.caption("These values are passed directly to the trained model in the exact metadata feature order.")
            with st.form("manual_prediction_form"):
                a, b, c = st.columns(3)
                with a: voltage = st.number_input("Voltage (V)", min_value=0.0, value=3.85, step=0.01, format="%.3f")
                with b: current = st.number_input("Current (A)", value=2.50, step=0.05, format="%.3f")
                with c: power = st.number_input("Power (W)", value=9.625, step=0.05, format="%.3f")
                d, e = st.columns(2)
                with d: temperature = st.number_input("Temperature (°C)", value=30.0, step=0.5)
                with e: time_value = st.number_input("Time (s)", min_value=0.0, value=1200.0, step=10.0)
                submitted = st.form_submit_button("Estimate battery SoC", type="primary", use_container_width=True)
            if submitted:
                manual_values = {"Voltage": voltage, "Current": current, "Power": power, "Temperature": temperature, "Time": time_value}
                row = pd.DataFrame([[manual_values[f] for f in features]], columns=features)
                try:
                    pred = np.asarray(model.predict(row)).reshape(-1)
                    if len(pred) != 1 or not np.isfinite(pred[0]):
                        raise ValueError("The model returned an invalid prediction.")
                    result = row.copy()
                    result["Predicted_SoC"] = pred
                    st.session_state.prediction_output = result
                    st.session_state.prediction_source = "Manual measurement"
                    st.success("Prediction completed.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            if st.session_state.prediction_source == "Manual measurement" and st.session_state.prediction_output is not None:
                val = float(st.session_state.prediction_output["Predicted_SoC"].iloc[0])
                st.metric("Predicted battery SoC", f"{val:.2f}%")
                if val < 0 or val > 100:
                    st.warning("Prediction is outside the expected 0–100% range. Review model training and input units.")
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=val, number={"suffix":"%", "font":{"size":32,"color":"#1C3554"}},
                    gauge={"axis":{"range":[0,100],"tickcolor":"#718096"},
                           "bar":{"color":"#168A7A"}, "bgcolor":"#F2F5F9",
                           "steps":[{"range":[0,20],"color":"#FCE8E6"},{"range":[20,80],"color":"#E8F2FC"},{"range":[80,100],"color":"#E5F5EF"}],
                           "threshold":{"line":{"color":"#245FD2","width":3},"thickness":0.75,"value":val}}
                ))
                gauge.update_layout(height=260, margin=dict(l=15,r=15,t=25,b=5), paper_bgcolor="rgba(0,0,0,0)", font={"color":"#20344F"})
                st.plotly_chart(gauge, use_container_width=True)

    csv_tab = tabs[-1]
    with csv_tab:
        st.markdown("#### Upload battery dataset")
        st.write("Your CSV must contain every feature listed in the model metadata. Extra columns are allowed and will be preserved in the results.")
        template = pd.DataFrame(columns=features)
        st.download_button(
            "Download model input template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="ev_battery_model_input_template.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="prediction_csv")
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                missing = [col for col in features if col not in input_df.columns]
                if missing:
                    st.error("Missing required feature columns: " + ", ".join(map(str, missing)))
                elif input_df.empty:
                    st.warning("The uploaded CSV contains no records.")
                else:
                    X = numeric_frame(input_df, features)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Records", f"{len(input_df):,}")
                    m2.metric("Required features", len(features))
                    m3.metric("Missing feature values", "0")
                    with st.expander("Preview model input", expanded=False):
                        st.dataframe(X.head(20), use_container_width=True, hide_index=True)
                    if st.button("Run SoC prediction", type="primary", use_container_width=True):
                        with st.spinner("Running the saved model..."):
                            predictions = np.asarray(model.predict(X)).reshape(-1)
                        if len(predictions) != len(input_df) or not np.isfinite(predictions).all():
                            st.error("The model returned an unexpected or invalid prediction array.")
                        else:
                            output = input_df.copy()
                            output["Predicted_SoC"] = predictions
                            target_col = find_target_column(input_df, metadata)
                            if target_col:
                                output["Actual_SoC"] = pd.to_numeric(input_df[target_col], errors="coerce")
                            st.session_state.prediction_output = output
                            st.session_state.prediction_source = uploaded.name
                            st.success("Prediction completed successfully.")
            except Exception as exc:
                st.error(f"Could not process the uploaded CSV: {exc}")

    # Show current output regardless of which tab produced it
    if st.session_state.prediction_output is not None:
        out = st.session_state.prediction_output
        pred = pd.to_numeric(out["Predicted_SoC"], errors="coerce").dropna()
        st.markdown("---")
        st.markdown("#### Prediction summary")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Predicted records", f"{len(pred):,}")
        q2.metric("Mean SoC", f"{pred.mean():.2f}%")
        q3.metric("Minimum SoC", f"{pred.min():.2f}%")
        q4.metric("Maximum SoC", f"{pred.max():.2f}%")
        if (pred.lt(0) | pred.gt(100)).any():
            st.warning("One or more predicted values fall outside 0–100%. Do not treat these as validated battery measurements.")
        trend = px.line(pd.DataFrame({"Record": np.arange(1, len(pred)+1), "Predicted SoC (%)": pred.to_numpy()}),
                        x="Record", y="Predicted SoC (%)", title="Predicted SoC across uploaded records")
        trend = plotly_layout(trend, 330)
        st.plotly_chart(trend, use_container_width=True)
        actual_col = "Actual_SoC" if "Actual_SoC" in out.columns else None
        if actual_col and out[actual_col].notna().any():
            valid = out[[actual_col, "Predicted_SoC"]].dropna()
            compare = pd.DataFrame({"Record": np.arange(1, len(valid)+1),
                                    "Actual SoC": valid[actual_col].to_numpy(),
                                    "Predicted SoC": valid["Predicted_SoC"].to_numpy()})
            fig = px.line(compare, x="Record", y=["Actual SoC", "Predicted SoC"], title="Actual vs predicted SoC")
            st.plotly_chart(plotly_layout(fig, 330), use_container_width=True)
            mae = float(np.mean(np.abs(valid[actual_col] - valid["Predicted_SoC"])))
            rmse = float(np.sqrt(np.mean((valid[actual_col] - valid["Predicted_SoC"]) ** 2)))
            e1, e2 = st.columns(2)
            e1.metric("MAE", f"{mae:.4f}")
            e2.metric("RMSE", f"{rmse:.4f}")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction results",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="ev_battery_soc_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

# --------------------------- MODEL PERFORMANCE ---------------------------
elif page == "Model Performance":
    st.markdown('<div class="section-head">Model performance & comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Review saved evaluation results from your training workflow.</div>', unsafe_allow_html=True)
    st.markdown(f"**Selected estimator:** {model_name}")
    if RESULTS_PATH.exists():
        try:
            results = pd.read_csv(RESULTS_PATH)
            st.dataframe(results, use_container_width=True, hide_index=True)
            numeric_cols = results.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                metric = st.selectbox("Choose metric to visualize", numeric_cols)
                non_numeric = [c for c in results.columns if c not in numeric_cols]
                if non_numeric:
                    model_col = non_numeric[0]
                    plot_df = results[[model_col, metric]].copy()
                    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
                    plot_df = plot_df.dropna()
                    fig = px.bar(plot_df, x=model_col, y=metric, text_auto=".4g",
                                 title=f"Model comparison · {metric}")
                    st.plotly_chart(plotly_layout(fig, 380), use_container_width=True)
                else:
                    st.info("No text model-name column was found, so the metric table is shown without a model comparison chart.")
        except Exception as exc:
            st.error(f"Could not read model comparison results: {exc}")
    else:
        st.info("model_comparison_results.csv was not found. Add it beside app.py to enable model comparison analytics.")
    st.caption("Only metrics present in your saved comparison CSV are displayed. The app does not invent evaluation scores.")

# --------------------------- DATASET EXPLORER ---------------------------
elif page == "Dataset Explorer":
    st.markdown('<div class="section-head">Dataset explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Inspect a source CSV, understand its columns and review descriptive statistics.</div>', unsafe_allow_html=True)
    data_file = st.file_uploader("Upload a dataset to explore", type=["csv"], key="explorer_csv")
    if data_file is not None:
        try:
            df = pd.read_csv(data_file)
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(df):,}")
            b.metric("Columns", f"{len(df.columns):,}")
            c.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
            st.markdown("#### Data preview")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            st.markdown("#### Column profile")
            profile = pd.DataFrame({
                "Column": df.columns.astype(str),
                "Data type": df.dtypes.astype(str).values,
                "Missing values": df.isna().sum().values,
                "Unique values": df.nunique(dropna=True).values,
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                st.markdown("#### Descriptive statistics")
                st.dataframe(numeric.describe().T, use_container_width=True)
                chosen = st.selectbox("Select numeric column", numeric.columns.tolist())
                hist = px.histogram(df, x=chosen, nbins=35, title=f"Distribution · {chosen}")
                st.plotly_chart(plotly_layout(hist, 330), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not inspect dataset: {exc}")
    else:
        st.info("Upload a CSV to inspect its structure. This explorer does not modify your training dataset.")

# --------------------------- PROJECT DETAILS ---------------------------
elif page == "Project Details":
    st.markdown('<div class="section-head">Project documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">EV Battery State-of-Charge Prediction using Computational Intelligence.</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown('<div class="panel"><div class="panel-title">Project objective</div><p style="color:#718096;font-size:13px">Estimate the State of Charge (SoC) of an electric-vehicle battery from operating measurements using a trained machine-learning model.</p><div class="panel-title">Current model configuration</div></div>', unsafe_allow_html=True)
        st.write("**Model:**", model_name)
        st.write("**Target:**", metadata.get("target_column", "Not specified"))
        st.write("**Input features:**", len(features))
    with p2:
        st.markdown('<div class="panel"><div class="panel-title">Technology stack</div><p style="color:#718096;font-size:13px">Python · Pandas · NumPy · Scikit-learn · Joblib · Streamlit · Plotly</p><div class="panel-title">Input schema</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Required feature": features}), use_container_width=True, hide_index=True)
    st.warning("Research and educational demonstration only. This application is not a certified battery-management system or medical/safety device. Do not use predictions to control real battery charging, discharging or protection systems.")

# --------------------------- FOOTER ---------------------------
st.markdown(
    """
    <div style="border-top:1px solid #E3E9F0;margin-top:35px;padding-top:16px;text-align:center;color:#91A0B2;font-size:10px;letter-spacing:.05em">
      VOLTEX · EV BATTERY INTELLIGENCE &nbsp; | &nbsp; COMPUTATIONAL INTELLIGENCE ACADEMIC PROJECT<br>
      Research prototype · Predictions depend on the uploaded data and the saved trained model
    </div>
    """,
    unsafe_allow_html=True,
)
