# Deployment verification — 2026-09-25

- Repository: https://github.com/Hang0418/PELD-RLDH-ML-Calculator (public)
- App: https://peld-rldh-ml-calculator-2026.streamlit.app/
- Source: `main`, `streamlit_app.py`; Streamlit Community Cloud Python 3.10.21
- GitHub Actions: https://github.com/Hang0418/PELD-RLDH-ML-Calculator/actions/runs/36154069984 — completed successfully
- Cloud startup: the app loaded all seven inputs, its research-only warning, prediction panel, and explanation chart without a visible startup error.
- Browser smoke test: default inputs (Age 43, BMI 23.4, sROM 7°, multifidus CSA 10.18 cm², sacral slope 26.82°, Pfirrmann I–II, Modic absent) produced a displayed probability of 9.0%. The same values passed to the local frozen model produced 0.090273, matching the displayed value after rounding.

This is a software smoke test, not new clinical validation. The validation hospital cohort calibration limitations described in the model card remain unchanged.
