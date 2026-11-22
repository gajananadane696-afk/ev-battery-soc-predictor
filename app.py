
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_soc_prediction_model.joblib"
METADATA_PATH = BASE_DIR / "project_metadata.json"
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PREMIUM DARK UI
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080D18, #101B2B);
    color: #F1F5F9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #0B1422;
    border-right: 1px solid #263449;
}

.hero {
    background: linear-gradient(110deg, #102C35, #14263D);
    border: 1px solid #28515A;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.hero p {
    color: #A9C4CD;
    font-size: 14px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #E5EDF7;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #17263A, #111D2E);
    border: 1px solid #2A3B52;
    border-radius: 15px;
    padding: 19px;
    min-height: 125px;
}

.metric-label {
    color: #9AAFC5;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #5EEAD4;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: #8399AF;
    font-size: 11px;
    margin-top: 5px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0D9488, #14B8A6);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    height: 48px;
}

div.stButton > button[kind="primary"]:hover {
    background: #0F766E;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: #132137;
    border: 1px dashed #3B6078;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293D55;
    border-radius: 10px;
}

hr {
    border-color: #263449;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>⚡ EV BATTERY INTELLIGENCE</h1>
    <p>Advanced Machine Learning | Battery State-of-Charge Prediction</p>
    <p>Computational Intelligence Project · Electric Vehicle Analytics</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files missing. Please add the trained model and "
        "project_metadata.json to the application folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()

features = metadata.get("feature_columns", [])

if not features:
    st.error("Feature columns are missing from metadata.")
    st.stop()

model_name = metadata.get(
    "best_model_by_test_RMSE",
    "Saved Machine Learning Model"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🔋 EV INTELLIGENCE")

    st.caption("BATTERY ANALYTICS PLATFORM")

    st.divider()

    st.markdown("### Model Information")

    st.success("Model loaded successfully")

    st.write(f"**Selected Model:** {model_name}")
    st.write(f"**Input Features:** {len(features)}")

    st.divider()

    st.markdown("### Input Parameters")

    for feature in features:
        st.caption(f"• {feature}")

    st.divider()

    st.markdown("### Project Information")

    st.write("**Domain:** Computational Intelligence")
    st.write("**Application:** EV Battery SoC Prediction")
    st.write("**Framework:** Streamlit")
    st.write("**Language:** Python")

    st.divider()

    st.caption(
        "Academic research prototype. "
        "Predictions are not certified battery measurements."
    )

# --------------------------------------------------
# METADATA / MODEL COMPARISON
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Battery Dashboard",
    "🧠 Model Analytics",
    "ℹ️ About Project"
])

# --------------------------------------------------
# TAB 1 - BATTERY DASHBOARD
# --------------------------------------------------

with tab1:

    st.markdown(
        '<div class="section-title">Battery Monitoring Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">AI MODEL</div>
            <div class="metric-value" style="font-size:20px;">
                ACTIVE
            </div>
            <div class="metric-note">Prediction engine ready</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">INPUT FEATURES</div>
            <div class="metric-value">{len(features)}</div>
            <div class="metric-note">Required model parameters</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PREDICTION MODE</div>
            <div class="metric-value" style="font-size:20px;">
                CSV BASED
            </div>
            <div class="metric-note">Offline dataset prediction</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Upload Battery Measurements</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload your battery dataset containing the required "
        "input features to estimate battery State of Charge."
    )

    template = pd.DataFrame(columns=features)

    st.download_button(
        "⬇ Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="EV_Battery_Input_Template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload Battery Dataset",
        type=["csv"],
        help="Upload a CSV containing all required model features."
    )

    if uploaded is not None:

        try:
            input_df = pd.read_csv(uploaded)

            missing = [
                col for col in features
                if col not in input_df.columns
            ]

            if missing:
                st.error(
                    "Missing required features: "
                    + ", ".join(missing)
                )
                st.stop()

            if input_df.empty:
                st.warning("Uploaded dataset is empty.")
                st.stop()

            prediction_input = input_df[features].copy()

            # Validate numeric inputs
            for col in features:
                prediction_input[col] = pd.to_numeric(
                    prediction_input[col],
                    errors="coerce"
                )

            if prediction_input.isnull().any().any():
                st.error(
                    "Some input features contain missing or "
                    "non-numeric values. Please correct your CSV."
                )
                st.stop()

            if not np.isfinite(
                prediction_input.to_numpy(dtype=float)
            ).all():
                st.error("Dataset contains infinite values.")
                st.stop()

            st.markdown("### Dataset Preview")

            m1, m2, m3 = st.columns(3)

            m1.metric("Total Records", f"{len(input_df):,}")
            m2.metric("Input Features", len(features))
            m3.metric("Dataset Size", f"{input_df.memory_usage(deep=True).sum()/1024:.1f} KB")

            with st.expander("View uploaded measurements"):
                st.dataframe(
                    prediction_input.head(100),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")

            if st.button(
                "⚡ PREDICT BATTERY SOC",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("AI model is analyzing battery data..."):

                    predictions = np.asarray(
                        model.predict(prediction_input)
                    ).reshape(-1)

                if len(predictions) != len(input_df):
                    st.error("Model returned an unexpected number of predictions.")
                    st.stop()

                if not np.isfinite(predictions).all():
                    st.error("Model returned invalid prediction values.")
                    st.stop()

                output = input_df.copy()

                output["Predicted_SoC"] = predictions

                avg_soc = float(np.mean(predictions))
                min_soc = float(np.min(predictions))
                max_soc = float(np.max(predictions))

                st.session_state["soc_output"] = output
                st.session_state["soc_predictions"] = predictions

                st.success("Battery SoC prediction completed!")

            # Display saved prediction results
            if "soc_output" in st.session_state:

                output = st.session_state["soc_output"]
                predictions = st.session_state["soc_predictions"]

                avg_soc = float(np.mean(predictions))
                min_soc = float(np.min(predictions))
                max_soc = float(np.max(predictions))

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">Prediction Results</div>',
                    unsafe_allow_html=True
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "Average SoC",
                    f"{avg_soc:.2f}%"
                )

                b.metric(
                    "Minimum SoC",
                    f"{min_soc:.2f}%"
                )

                c.metric(
                    "Maximum SoC",
                    f"{max_soc:.2f}%"
                )

                d.metric(
                    "Predicted Records",
                    f"{len(output):,}"
                )

                if min_soc < 0 or max_soc > 100:
                    st.warning(
                        "Some predictions are outside the expected "
                        "0–100% SoC range. Check model performance "
                        "and training data."
                    )

                st.markdown("### Battery SoC Distribution")

                chart_data = pd.DataFrame({
                    "Predicted SoC (%)": predictions
                })

                st.bar_chart(
                    chart_data,
                    height=280
                )

                st.markdown("### SoC Prediction Trend")

                trend_data = pd.DataFrame({
                    "Predicted SoC (%)": predictions
                })

                trend_data.index.name = "Record"

                st.line_chart(
                    trend_data,
                    height=300
                )

                st.markdown("### Complete Prediction Table")

                st.dataframe(
                    output,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇ Download Prediction Results",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="EV_Battery_SoC_Predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as exc:
            st.error(f"Could not process dataset: {exc}")

    else:
        st.info(
            "Upload your CSV dataset to begin battery SoC prediction."
        )

# --------------------------------------------------
# TAB 2 - MODEL ANALYTICS
# --------------------------------------------------

with tab2:

    st.markdown(
        '<div class="section-title">Machine Learning Performance</div>',
        unsafe_allow_html=True
    )

    if RESULTS_PATH.exists():

        try:
            results = pd.read_csv(RESULTS_PATH)

            st.markdown("### Model Comparison")

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            # Identify common numeric performance metrics
            metric_columns = [
                col for col in results.select_dtypes(
                    include="number"
                ).columns
                if col.lower() in [
                    "mae", "mse", "rmse", "r2",
                    "r2_score", "accuracy"
                ]
            ]

            if metric_columns:

                st.markdown("### Performance Comparison Charts")

                selected_metric = st.selectbox(
                    "Select evaluation metric",
                    metric_columns
                )

                model_columns = [
                    col for col in results.columns
                    if col not in metric_columns
                    and not pd.api.types.is_numeric_dtype(
                        results[col]
                    )
                ]

                if model_columns:

                    chart_df = results[
                        [model_columns[0], selected_metric]
                    ].copy()

                    chart_df[selected_metric] = pd.to_numeric(
                        chart_df[selected_metric],
                        errors="coerce"
                    )

                    chart_df = chart_df.dropna().set_index(
                        model_columns[0]
                    )

                    st.bar_chart(
                        chart_df,
                        height=350
                    )

        except Exception as exc:
            st.error(f"Could not load model results: {exc}")

    else:
        st.warning(
            "Model comparison CSV not found. "
            "Add model_comparison_results.csv to view analytics."
        )

# --------------------------------------------------
# TAB 3 - PROJECT INFORMATION
# --------------------------------------------------

with tab3:

    st.markdown("## About This Project")

    st.write(
        "This project uses Computational Intelligence and "
        "Machine Learning techniques to estimate the State of "
        "Charge of Electric Vehicle batteries."
    )

    st.markdown("### Project Objectives")

    st.markdown("""
    - Predict battery State of Charge using machine learning.
    - Analyze battery operating parameters.
    - Compare machine learning model performance.
    - Visualize battery SoC predictions.
    - Provide an interactive prediction interface.
    """)

    st.markdown("### Input Features")

    st.dataframe(
        pd.DataFrame({
            "Feature Name": features
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Technology Stack")

    st.markdown("""
    | Technology | Purpose |
    |---|---|
    | Python | Programming |
    | Pandas | Data processing |
    | NumPy | Numerical computation |
    | Scikit-learn | Machine Learning |
    | Joblib | Model loading |
    | Streamlit | Dashboard |
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8094AA; padding:15px;">
        <b>EV BATTERY INTELLIGENCE SYSTEM</b><br>
        Computational Intelligence Academic Project<br>
        Research and Educational Demonstration Only
    </div>
    """,
    unsafe_allow_html=True
)
