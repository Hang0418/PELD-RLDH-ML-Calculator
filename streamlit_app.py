"""Research-only Streamlit interface for the frozen PELD-RLDH model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import local_explanation
from src.prediction import canonicalize, metadata, predict_probability


st.set_page_config(page_title="PELD RLDH Research Calculator", layout="wide")
st.title("PELD RLDH Research Calculator")
st.caption("Research estimate of RLDH within 2 years after the index PELD procedure")
st.warning(
    "Research use only. External discrimination was encouraging, but probability "
    "calibration changed in the validation hospital cohort. This is not a clinical decision tool."
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
                "Interpret as a research estimate, not a treatment threshold. "
                "In the validation hospital cohort, the calibration slope was 0.709 and O/E was 0.866. "
                "No low-, medium-, or high-risk cutoff was prespecified."
            )
            st.subheader("Factors contributing to this prediction")
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
                f"Synthetic-background reference probability: "
                f"{explanation['base_probability']:.1%}. Contributions sum to the displayed "
                "probability; their baseline differs from the manuscript's patient-sample SHAP baseline. "
                "SHAP describes model behavior, not causal effects."
            )
    else:
        st.subheader("Model prediction")
        st.write("Enter all seven inputs and select **Calculate probability**.")

with st.expander("About this model"):
    st.write(
        "Development hospital cohort: 1,190 patients and 117 recorded recurrences. "
        "Validation hospital cohort: 288 patients and 36 recorded recurrences. "
        "External AUROC 0.847, AUPRC 0.640, Brier 0.0772, calibration slope 0.709."
    )
    st.write(f"Frozen source model SHA-256: `{meta['source_model_sha256']}`")
with st.expander("Predictor definitions and limitations"):
    st.write(
        "Inputs must follow the source study's measurement and coding methods. "
        "The study methods define CSA as bilateral multifidus area at L4–5, measured in ImageJ "
        "and averaged. The stated primary outcome is MRI-confirmed herniation at the operated "
        "level within two years, after at least one pain-free month. The supplied workbooks do "
        "not themselves document patient-level follow-up completeness. "
        "See the repository's predictor definitions and model card before interpreting results."
    )
st.divider()
st.caption(
    "Research and education only. Not a medical device; do not use as the sole basis for "
    "diagnosis, treatment selection, or clinical decisions. This app code has no patient database "
    "and does not write entered values to disk. The hosting platform may maintain operational logs."
)
