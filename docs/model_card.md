# Frozen Model Card

## Purpose

Research-only estimation of study-defined RLDH within two years after index PELD. No treatment recommendation or risk-category threshold is provided.

## Development and validation

Hospital 5 development: 1,190 participants, 117 recorded events. Independent Hospital 6 evaluation: 288 participants, 36 events. The seven variables were selected from 16 candidates using a 100-bootstrap LASSO/Boruta consensus rule. XGBoost was selected using Hospital 5 nested out-of-fold evidence and frozen before Hospital 6 evaluation.

## Inputs and output

The inputs are the seven source-model fields in `model/model_metadata.json`. The output is an uncalibrated-against-Hospital-6 model probability, not a diagnostic classification or treatment threshold. Numeric preprocessing uses frozen training means and scales; categorical preprocessing uses frozen level order with the first level dropped.

## Performance

Hospital 5 outer-fold out-of-fold AUROC 0.904, AUPRC 0.755, Brier 0.0372, calibration slope 1.051, O/E 1.006. Hospital 6 AUROC 0.847 (95% CI 0.758–0.928), AUPRC 0.640 (0.487–0.782), Brier 0.0772 (0.0532–0.1016), calibration-in-the-large −0.334, slope 0.709, and O/E 0.866. External probability calibration was weaker, so results are research estimates.

## Interpretation

Local probability-scale SHAP uses a synthetic, aggregate-derived background to avoid publishing patient rows. It reconstructs each application prediction but does not share the manuscript's patient-sample SHAP baseline. SHAP contributions describe model behavior and are not causal effects.

## Known limitations

The public package does not contain source records or patient-level follow-up completeness fields. Independent validation involved one additional hospital and 36 events. The tool has not undergone prospective impact evaluation, software-as-medical-device assessment, or clinical deployment validation.
