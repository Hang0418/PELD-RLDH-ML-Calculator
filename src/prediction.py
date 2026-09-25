"""Exact numerical preprocessing and frozen XGBoost inference.

No training code or patient records are part of this module.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "xgboost_frozen_model.json"
METADATA_PATH = ROOT / "model" / "model_metadata.json"


@lru_cache(maxsize=1)
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def model() -> xgb.XGBClassifier:
    meta = metadata()
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != meta["native_model_sha256"]:
        raise RuntimeError("Frozen model checksum mismatch")
    classifier = xgb.XGBClassifier()
    classifier.load_model(MODEL_PATH)
    return classifier


def canonicalize(values: Mapping[str, object]) -> dict[str, object]:
    meta = metadata()
    expected = meta["features"]
    if set(values) != set(expected):
        raise ValueError(f"Expected exactly these seven inputs: {expected}")
    record: dict[str, object] = {}
    for name in meta["numeric_features"]:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        lower, upper = meta["development_ranges"][name]
        if not lower <= value <= upper:
            raise ValueError(f"{name} is outside the development range [{lower}, {upper}]")
        record[name] = value
    for name in meta["categorical_features"]:
        value = str(values[name])
        if value not in meta["category_levels"][name]:
            raise ValueError(f"Unknown {name} level: {value}")
        record[name] = value
    return record


def preprocess(values: Mapping[str, object]) -> np.ndarray:
    record = canonicalize(values)
    meta = metadata()
    numeric = [
        (record[name] - meta["numeric_mean"][name]) / meta["numeric_scale"][name]
        for name in meta["numeric_features"]
    ]
    categorical = []
    for name in meta["categorical_features"]:
        # The frozen sklearn OneHotEncoder used drop='first'. Its category
        # order, not the UI presentation order, is copied into metadata.
        categorical.extend(
            float(record[name] == level)
            for level in meta["category_levels"][name][1:]
        )
    array = np.asarray([numeric + categorical], dtype=np.float32)
    if array.shape[1] != len(meta["encoded_features"]):
        raise RuntimeError("Frozen feature count mismatch")
    return array


def predict_probability(values: Mapping[str, object]) -> float:
    probability = float(model().predict_proba(preprocess(values))[0, 1])
    if not 0 <= probability <= 1:
        raise RuntimeError("Model returned an invalid probability")
    return probability
