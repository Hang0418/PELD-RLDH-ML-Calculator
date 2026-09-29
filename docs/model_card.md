# Frozen Model Card

## Purpose

Research-only estimation of study-defined RLDH within two years after index PELD. No treatment recommendation or risk-category threshold is provided.

## Development and validation

Development hospital cohort: 1,190 participants, 117 recorded events. Validation hospital cohort: 288 participants, 36 events. The seven variables were selected from 16 candidates using a 100-bootstrap LASSO/Boruta consensus rule. XGBoost was selected using nested out-of-fold evidence from the development hospital cohort and frozen before independent evaluation in the validation hospital cohort.

## Inputs and output

The inputs are the seven source-model fields in `model/model_metadata.json`. The output is a model probability that has not been recalibrated against the validation hospital cohort, not a diagnostic classification or treatment threshold. Numeric preprocessing uses frozen training means and scales; categorical preprocessing uses frozen level order with the first level dropped.

## Performance

Development hospital cohort outer-fold out-of-fold AUROC 0.904, AUPRC 0.755, Brier 0.0372, calibration slope 1.051, O/E 1.006. Validation hospital cohort AUROC 0.847 (95% CI 0.758–0.928), AUPRC 0.640 (0.487–0.782), Brier 0.0772 (0.0532–0.1016), calibration-in-the-large −0.334, slope 0.709, and O/E 0.866. External probability calibration was weaker, so results are research estimates.

## Interpretation

The web app implements the same frozen prediction model and preprocessing as the externally validated model. Its probability-scale permutation SHAP uses a synthetic, aggregate-derived background to avoid publishing patient rows. It reconstructs each application prediction, but its reference baseline differs from the manuscript's patient-sample SHAP analysis. SHAP contributions describe model behavior, not causal effects.

## Known limitations

Patient-level source data are not included in the public software package; cohort eligibility, follow-up completeness, and outcome adjudication cannot be independently reconstructed from it. Independent validation involved one additional hospital and 36 events. The tool has not undergone prospective impact evaluation, software-as-medical-device assessment, or clinical deployment validation.
