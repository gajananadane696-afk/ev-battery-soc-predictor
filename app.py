
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_soc_prediction_model.joblib"
METADATA_PATH = BASE_DIR / "project_metadata.json"
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"

st.set_page_config(
    page_title="EV Battery SoC Predictor",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EV Battery State-of-Charge Predictor")
st.caption("Computational Intelligence Project | Machine-learning-based battery SoC estimation")

st.markdown(
    "Upload battery operating measurements in CSV format. "
    "The application applies the saved preprocessing-and-model "
    "pipeline and returns an estimated State of Charge (SoC)."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("The trained model files are missing.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
except Exception as exc:
    st.error(f"Could not load model or metadata: {exc}")
    st.stop()

features = metadata.get("feature_columns", [])

if not features:
    st.error("No feature_columns found in project_metadata.json.")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write(
        f"**Selected model:** "
        f"{metadata.get('best_model_by_test_RMSE', 'Saved model')}"
    )
    st.write(f"**Input features:** {len(features)}")
    st.caption(
        "Predictions are research estimates, not certified "
        "battery measurements or safety controls."
    )

    if RESULTS_PATH.exists():
        st.subheader("Model comparison")
        try:
            results = pd.read_csv(RESULTS_PATH)
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )
        except Exception:
            st.caption("Could not read model comparison results.")

st.subheader("1. Prepare input data")
st.write(
    "Your CSV must contain one or more rows and all "
    "the feature columns used during training."
)

template = pd.DataFrame(columns=features)

st.download_button(
    "Download CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="ev_battery_input_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader(
    "Upload battery measurements (CSV)",
    type=["csv"]
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
                "Missing required feature columns: "
                + ", ".join(map(str, missing))
            )
            st.stop()

        if input_df.empty:
            st.warning("The uploaded CSV has no rows.")
            st.stop()

        prediction_input = input_df[features].copy()

        st.subheader("2. Review uploaded measurements")
        st.dataframe(
            prediction_input.head(20),
            use_container_width=True
        )

        if st.button("Predict battery SoC", type="primary"):
            with st.spinner("Calculating SoC estimates..."):
                predictions = model.predict(prediction_input)

            output = input_df.copy()
            output["Predicted_SoC"] = predictions

            st.subheader("3. Prediction results")

            first_col, second_col = st.columns(2)

            first_col.metric(
                "Rows predicted",
                f"{len(output):,}"
            )

            second_col.metric(
                "Average predicted SoC",
                f"{float(pd.Series(predictions).mean()):.2f}"
            )

            st.dataframe(
                output,
                use_container_width=True
            )

            st.download_button(
                "Download predictions CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="ev_battery_soc_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")

st.divider()

st.caption(
    "Academic software prototype using an existing dataset. "
    "Do not use these estimates to control real battery charging, "
    "discharging, or protection systems."
)
