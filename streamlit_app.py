"""Research-only Streamlit interface for the frozen PELD-RLDH model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import local_explanation
from src.prediction import canonicalize, metadata, predict_probability


st.set_page_config(page_title="PELD RLDH Research Calculator", layout="wide")
st.title("PELD RLDH Research Calculator")
st.caption("Unrecalibrated model estimate of recurrent lumbar disc herniation within 2 years after PELD")
st.warning(
    "Research use only. Validation hospital cohort calibration slope: 0.709. "
    "Do not use this estimate to make clinical decisions."
)

meta = metadata()
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Patient characteristics")
    with st.form("calculator"):
        values = {
            "Age": st.number_input("Age (years)", min_value=12.0, max_value=87.0,
                                   value=43.0, step=1.0),
            "BMI": st.number_input("BMI (kg/m²)", min_value=16.2, max_value=34.5,
                                   value=23.4, step=.1),
            "sROM/degrees": st.number_input("Segmental range of motion (°)",
                                             min_value=1.0, max_value=35.0,
                                             value=7.0, step=.1,
                                             help="Use the source study's sROM measurement protocol."),
            "Cross_sectional_area/cm^2": st.number_input(
                "Multifidus cross-sectional area at L4–5 (cm²)",
                min_value=2.37, max_value=23.032, value=10.18, step=.01,
                help="Bilateral multifidus CSA measured at L4–5 using ImageJ and averaged."),
            "Sacral_slope/degrees": st.number_input("Sacral slope (°)",
                                                    min_value=16.36, max_value=48.83,
                                                    value=26.82, step=.1),
            "Pfirrmann_group": st.selectbox("Pfirrmann group", ["I–II", "III–IV", "V"]),
            "Modic_group": st.selectbox("Modic group", ["Absent", "I", "II–III"]),
        }
        submitted = st.form_submit_button("Calculate probability", type="primary",
                                          use_container_width=True)

with right:
    if submitted:
        # Only the frozen model's source coding is passed to the backend.
        values["Modic_group"] = "无" if values["Modic_group"] == "Absent" else values["Modic_group"]
        try:
            record = canonicalize(values)
            probability = predict_probability(record)
            explanation = local_explanation(record)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.subheader("Model prediction")
            st.metric("Estimated 2-year RLDH probability", f"{probability:.1%}")
            st.progress(probability)
            st.info(
                "This probability was not recalibrated for the validation hospital cohort "
                "(O/E 0.866). No clinical decision threshold was validated."
            )
            st.subheader("Model contributions (SHAP)")
            names = {
                "Age": "Age", "BMI": "BMI", "sROM/degrees": "Segmental ROM",
                "Cross_sectional_area/cm^2": "Multifidus CSA",
                "Sacral_slope/degrees": "Sacral slope",
                "Pfirrmann_group": "Pfirrmann group", "Modic_group": "Modic group",
            }
            frame = pd.DataFrame(
                [{"Predictor": names[key], "Probability contribution": value}
                 for key, value in explanation["contributions"].items()]
            ).set_index("Predictor").sort_values("Probability contribution")
            st.bar_chart(frame, horizontal=True, color="#0072B2")
            st.caption(
                f"Synthetic-background reference: {explanation['base_probability']:.1%}. "
                "Contributions sum to the prediction; this reference differs from the "
                "manuscript's patient-sample SHAP baseline. Contributions are not causal effects."
            )
    else:
        st.subheader("Model prediction")
        st.write("Enter all seven inputs and select **Calculate probability**.")

with st.expander("About this model"):
    st.write(
        "Development hospital cohort: 1,190 participants, 117 events. "
        "Validation hospital cohort: 288 participants, 36 events. "
        "Validation AUROC 0.847, AUPRC 0.640, Brier score 0.0772, "
        "calibration slope 0.709, O/E 0.866."
    )
    st.write(f"Frozen source model SHA-256: `{meta['source_model_sha256']}`")
with st.expander("Outcome and predictor definitions"):
    st.write(
        "RLDH: MRI-confirmed herniation at the operated level, ipsilateral or "
        "contralateral, within 2 years after PELD and after ≥1 pain-free month. "
        "Segmental range of motion uses standing flexion–extension radiographs; "
        "multifidus CSA is the bilateral L4–5 ImageJ mean; sacral slope is the angle "
        "between the sacral endplate and horizontal."
    )
    st.markdown(
        "[Predictor coding](https://github.com/Hang0418/PELD-RLDH-ML-Calculator/"
        "blob/main/docs/predictor_definitions.md) · "
        "[Model card](https://github.com/Hang0418/PELD-RLDH-ML-Calculator/"
        "blob/main/docs/model_card.md)"
    )
st.divider()
st.caption(
    "No patient database or input persistence is implemented in this app. "
    "Do not enter identifying information. "
    "[Source code and methods](https://github.com/Hang0418/PELD-RLDH-ML-Calculator)."
)
