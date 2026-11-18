# EV Battery SoC Predictor — Streamlit deployment

This is the Streamlit front end for the Computational Intelligence project. It loads the **actual fitted preprocessing + model pipeline** exported by the Colab notebook and predicts SoC for uploaded CSV rows.

## Important: add the trained artifacts first

The saved project library contains the notebook, report and presentation, but the fitted `.joblib` model artifact itself was not present. Do not replace it with made-up predictions or a dummy model.

1. Open `EV_Battery_SoC_Prediction_CI_Project.ipynb` in Google Colab.
2. Run all cells through **“Save trained model and project outputs”** (the final export cell).
3. Download these files from the generated `ev_soc_project_outputs` folder:
   - `best_soc_prediction_model.joblib`
   - `project_metadata.json`
   - `model_comparison_results.csv` (optional but used for the sidebar comparison)
4. Put the downloaded files beside `app.py` in this repository.

The model and metadata must come from the same notebook run. The model includes its preprocessing pipeline; do not separately scale or encode uploaded input data.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `ev-battery-soc-predictor`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the three exported artifacts listed above.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit will build the environment and provide the public app URL when deployment succeeds.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input format

Use the **Download CSV template** button in the app. It contains the exact feature column names saved in the model metadata. Fill one or more rows with measurements and upload the CSV. The app returns `Predicted_SoC` and allows downloading the results.

## Safety and scope

This is an academic dataset-based prediction prototype. It has no physical sensor, live battery connection, or charging-control function. Predictions are not certified battery safety measurements.
