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
