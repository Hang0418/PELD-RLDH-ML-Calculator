# PELD RLDH Research Calculator

This standalone Streamlit repository implements the **frozen seven-predictor XGBoost model** from the Hospital 5 development cohort. It estimates the probability of the study-defined recurrent lumbar disc herniation (RLDH) outcome within two years after index percutaneous endoscopic lumbar discectomy (PELD). It is for research and education, not diagnosis or treatment selection.

**Deployment status:** public research preview at [peld-rldh-ml-calculator-2026.streamlit.app](https://peld-rldh-ml-calculator-2026.streamlit.app/). The app was deployed from this repository's `main` branch on 2026-09-25 and its startup and example prediction were verified. It is not approved for clinical use.

## Model and evidence

The model accepts age, BMI, segmental range of motion, bilateral mean L4–5 multifidus cross-sectional area, sacral slope, Pfirrmann group, and Modic group. It uses the exact frozen numeric scaling and category ordering. Hospital 5 supplied 1,190 development patients and 117 events. Hospital 6 supplied 288 independent validation patients and 36 events. External AUROC was 0.847, AUPRC 0.640, Brier score 0.0772, calibration slope 0.709, and O/E 0.866. These findings do not establish deployment-ready probability calibration.

The stated outcome is MRI-confirmed herniation at the operated level, ipsilateral or contralateral, after at least one pain-free month and occurring within two years. This definition was supplied by the study team; patient-level follow-up completeness and adjudication fields are not present in the public package or supplied workbooks.

## Run locally

Use Python 3.10 and install the pinned dependencies in `requirements.txt` in an isolated environment. Then run:

```bash
streamlit run streamlit_app.py
```

The published native XGBoost JSON is checked against the SHA-256 in `model/model_metadata.json` at load time. The original serialized pipeline SHA-256 is also recorded there. The application does not refit the model or choose a clinical cutoff.

## Tests and provenance

Run `python -m pytest -q`. The 20 published reference cases and 30 SHAP-background cases are **synthetic**, not sampled patient records. Native export was checked against the private frozen pipeline with maximum absolute probability difference recorded in `tests/export_verification.json`; the test tolerance is `1e-6`. Local SHAP values are computed on the probability scale with the same permutation algorithm but a synthetic background, so the reference baseline and individual contributions need not numerically match the manuscript's patient-sample SHAP analysis.

The optional export script, `scripts/export_frozen_assets.py`, documents how the public model was derived from the private frozen pipeline. It requires the original private analysis directory and verifies both the source workbook and model hashes. Neither source workbook, patient row, patient identifier, nor serialized `joblib` pipeline is part of this repository.

## Intended use, privacy, and limitations

This is not a medical device. Do not use its output as a sole basis for diagnosis, surgery selection, or other clinical decisions. No risk categories or treatment recommendations are supplied. Inputs are limited to the observed development ranges, but being inside a range does not guarantee clinical applicability. Additional multicenter and prospective validation, calibration work, and independent software review are needed before clinical use.

The app code has no login, upload, patient database, or persistence for entered values. Inputs are processed in the current Streamlit session. The hosting provider may have operational logs; do not enter directly identifying information.

See `docs/model_card.md`, `docs/predictor_definitions.md`, and `docs/validation_summary.md` for more detail.

## Deployment

The public app runs `streamlit_app.py` from branch `main` on Streamlit Community Cloud with Python 3.10. On 2026-09-25, the cloud build installed the pinned dependencies, the app loaded without a startup error, and the default demonstration inputs returned 9.0%, agreeing with the frozen local model probability of 0.090273 after display rounding. GitHub Actions also passed the deterministic prediction and explanation tests. See `docs/deployment_verification.md` for the verification record. A version 1.0.0 release, DOI, manuscript screenshot, and citation should be made only after final wording and release readiness are confirmed.
