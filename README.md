# PELD RLDH Research Calculator

[Open the online calculator](https://peld-rldh-ml-calculator-2026.streamlit.app/) · [Predictor definitions](docs/predictor_definitions.md) · [Model card](docs/model_card.md)

This repository contains the frozen seven-predictor XGBoost model and Streamlit interface for estimating the study-defined probability of recurrent lumbar disc herniation (RLDH) within two years after index percutaneous endoscopic lumbar discectomy (PELD). The output is for research use, not clinical decision-making.

## Model and validation

The predictors are age, BMI, segmental range of motion, mean bilateral L4–5 multifidus cross-sectional area, sacral slope, Pfirrmann group, and Modic group. RLDH was defined as MRI-confirmed herniation at the operated level, ipsilateral or contralateral, after at least one pain-free month.

| Cohort | Participants | Events | AUROC | AUPRC | Brier score | Calibration slope | O/E |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development hospital cohort (outer-fold out-of-fold) | 1,190 | 117 | 0.904 | 0.755 | 0.0372 | 1.051 | 1.006 |
| Validation hospital cohort | 288 | 36 | 0.847 | 0.640 | 0.0772 | 0.709 | 0.866 |

The displayed probability has **not** been recalibrated for the validation hospital cohort. No clinical decision threshold was validated. Patient-level follow-up completeness and outcome adjudication cannot be verified from the public repository or the supplied workbooks.

## Code and reproducibility

- `streamlit_app.py`: seven-input interface and probability-scale local SHAP display.
- `src/prediction.py`: frozen feature coding, training-set scaling, model checksum verification, and probability prediction; no fitting occurs in the app.
- `src/explain.py`: permutation SHAP with 30 synthetic reference profiles. This background differs from the manuscript's patient-sample SHAP background.
- `model/`: native XGBoost JSON, preprocessing metadata, and synthetic SHAP background; no patient records.
- `tests/`: 20 synthetic reference predictions, export verification, and automated inference tests.
- `scripts/export_frozen_assets.py`: export procedure; it requires the private frozen pipeline and source workbook and is not needed to run the app.

The native model checksum is checked at load time. The 20 synthetic reference probabilities matched the private frozen pipeline within `1e-6` at export; see [`tests/export_verification.json`](tests/export_verification.json). Source workbooks and patient-level data are not published here. See the [validation and software reproducibility record](docs/validation_summary.md) for version details.

To run with Python 3.10 and the pinned dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run streamlit_app.py
```

The [deployment verification record](docs/deployment_verification.md) documents the original cloud smoke test. GitHub Actions runs the test suite on pushes and pull requests.

## Use boundary

This is not a medical device or a treatment recommendation. External validation included one hospital and 36 events; prospective clinical impact has not been evaluated. The app implements no patient database or input persistence. Do not enter identifying information; the hosting provider can retain operational logs. Code and model weights are publicly viewable but remain all rights reserved; see [`LICENSE`](LICENSE).
