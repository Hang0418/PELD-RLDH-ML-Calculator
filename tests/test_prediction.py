"""Public, synthetic-reference tests; no patient rows are included."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction import canonicalize, metadata, predict_probability, preprocess


ROOT = Path(__file__).resolve().parents[1]


def cases() -> pd.DataFrame:
    return pd.read_csv(ROOT / "tests/reference_predictions.csv")


def test_exported_model_reproduces_frozen_pipeline_references() -> None:
    frame = cases()
    assert len(frame) == 20
    assert list(frame.columns) == metadata()["features"] + ["expected_probability"]
    errors = []
    for _, row in frame.iterrows():
        record = {name: row[name] for name in metadata()["features"]}
        actual = predict_probability(record)
        errors.append(abs(actual - row.expected_probability))
        assert 0 <= actual <= 1
    assert max(errors) < 1e-6


def test_categorical_encoding_uses_frozen_order() -> None:
    first = cases().iloc[0]
    record = {name: first[name] for name in metadata()["features"]}
    encoded = preprocess(record)[0]
    assert len(encoded) == 9
    meta = metadata()
    expected_tail = []
    for name in meta["categorical_features"]:
        expected_tail.extend(float(record[name] == level)
                             for level in meta["category_levels"][name][1:])
    np.testing.assert_array_equal(encoded[5:], expected_tail)


def test_invalid_input_rejected() -> None:
    first = cases().iloc[0]
    record = {name: first[name] for name in metadata()["features"]}
    for replacement in (float("nan"), -1, 1000):
        invalid = dict(record, Age=replacement)
        with pytest.raises(ValueError):
            canonicalize(invalid)
    with pytest.raises(ValueError):
        canonicalize(dict(record, Modic_group="unknown"))
    with pytest.raises(ValueError):
        canonicalize({"Age": 50})


def test_export_contains_no_patient_rows() -> None:
    report = json.loads((ROOT / "tests/export_verification.json").read_text())
    assert report["patient_rows_exported"] == 0
    assert report["synthetic_cases"] == 20
    assert report["maximum_absolute_difference"] < report["tolerance"]
    assert "id" not in cases().columns


def test_shap_additivity_on_synthetic_case() -> None:
    from src.explain import local_explanation

    first = cases().iloc[0]
    record = {name: first[name] for name in metadata()["features"]}
    explained = local_explanation(record)
    assert len(explained["contributions"]) == 7
    assert abs(explained["base_probability"]
               + sum(explained["contributions"].values())
               - explained["predicted_probability"]) < 1e-5
