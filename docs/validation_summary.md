# Validation and Software Reproducibility

The native JSON model and aggregate preprocessing metadata were exported from the frozen Hospital 5 pipeline. The original pipeline and development-workbook SHA-256 values were checked before export. Twenty deterministic, synthetic input profiles were compared with the private frozen pipeline. The maximum absolute prediction difference and native-model checksum are recorded in `tests/export_verification.json`. No patient row was exported.

Automated tests cover the 20 reference probabilities, predictor order, frozen categorical encoding, numeric bounds, invalid values, model checksum, and local SHAP additivity. GitHub Actions runs the test suite on every push and pull request. A passing local test does **not** substitute for a successful test on the deployed Streamlit app.

Analysis-environment component versions at model export: Python 3.10.14, XGBoost 3.2.0, scikit-learn 1.7.2, SHAP 0.49.1, NumPy 1.26.4, pandas 2.3.3, and joblib 1.5.2. Streamlit was not installed in the original analysis environment; the web app pins Streamlit 1.64.0 separately. The app should be deployed with Python 3.10, and its dependency installation and browser behavior must be checked on Community Cloud.
