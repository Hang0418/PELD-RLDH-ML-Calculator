"""Probability-scale local SHAP using a non-patient synthetic background."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import shap

from src.prediction import metadata, model, predict_probability, preprocess


BACKGROUND_PATH = Path(__file__).resolve().parents[1] / "model" / "synthetic_background.json"


@lru_cache(maxsize=1)
def _explainer():
    source = json.loads(BACKGROUND_PATH.read_text(encoding="utf-8"))
    background = np.vstack([preprocess(case)[0] for case in source])
    predictor = lambda x: model().predict_proba(np.asarray(x, dtype=np.float32))[:, 1]
    return shap.Explainer(
        predictor,
        background,
        algorithm="permutation",
        feature_names=metadata()["encoded_features"],
        seed=20260920,
    )


def local_explanation(values: Mapping[str, object]) -> dict:
    encoded = preprocess(values)
    result = _explainer()(encoded, max_evals=2 * encoded.shape[1] + 1)
    base = float(np.asarray(result.base_values).reshape(-1)[0])
    contributions = np.asarray(result.values).reshape(-1)
    probability = predict_probability(values)
    if abs(base + contributions.sum() - probability) > 1e-5:
        raise RuntimeError("SHAP contributions do not reconstruct prediction")
    grouped = {}
    for source, contribution in zip(metadata()["encoded_sources"], contributions):
        grouped[source] = grouped.get(source, 0.0) + float(contribution)
    return {"base_probability": base, "contributions": grouped,
            "predicted_probability": probability}
