"""Export only a frozen native XGBoost model and aggregate preprocessing.

Run locally with `--source-root` pointing to the private analysis directory.
No patient rows or identifiers are copied to this public-repository package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_records(ranges: dict, categories: dict, count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    names = list(ranges)
    for index in range(count):
        record = {}
        for name in names:
            low, high = ranges[name]
            record[name] = float(np.round(rng.uniform(low, high), 5))
        for offset, (name, levels) in enumerate(categories.items()):
            record[name] = levels[(index + offset) % len(levels)]
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    base = source / "outputs" / "trainval22_framework_20260922"
    lock = json.loads((base / "modeling/model_freeze.json").read_text())
    model_source = base / "modeling/h5_frozen_model.joblib"
    training_source = source / "训练集.xlsx"
    if sha256(model_source) != lock["model_sha256"]:
        raise RuntimeError("Frozen pipeline hash differs from model lock")
    if sha256(training_source) != lock["h5_source_sha256"]:
        raise RuntimeError("Development source hash differs from model lock")
    pipeline = joblib.load(model_source)
    prep, estimator = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    numeric = list(prep.transformers_[0][2])
    categorical = list(prep.transformers_[1][2])
    scaler, encoder = prep.named_transformers_["num"], prep.named_transformers_["cat"]
    if list(lock["features"]) != numeric + categorical or list(encoder.drop_idx_) != [0, 0]:
        raise RuntimeError("Frozen preprocessing contract changed")
    source_data = pd.read_excel(training_source, usecols=lock["features"])
    ranges = {name: [float(source_data[name].min()), float(source_data[name].max())]
              for name in numeric}
    categories = {name: [str(v) for v in levels]
                  for name, levels in zip(categorical, encoder.categories_)}
    encoded_names = list(map(str, prep.get_feature_names_out()))
    encoded_sources = numeric + [name for name, levels in categories.items()
                                 for _ in levels[1:]]
    model_target = REPO / "model/xgboost_frozen_model.json"
    model_target.parent.mkdir(parents=True, exist_ok=True)
    estimator.save_model(model_target)
    metadata = {
        "version": "0.1.0-research-preview",
        "frozen_training_seed": lock["seed"],
        "source_model_sha256": lock["model_sha256"],
        "development_source_sha256": lock["h5_source_sha256"],
        "native_model_sha256": sha256(model_target),
        "features": list(lock["features"]),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_mean": {name: float(value) for name, value in zip(numeric, scaler.mean_)},
        "numeric_scale": {name: float(value) for name, value in zip(numeric, scaler.scale_)},
        "development_ranges": ranges,
        "category_levels": categories,
        "encoded_features": encoded_names,
        "encoded_sources": encoded_sources,
        "stated_outcome_horizon_years": 2,
        "patient_level_follow_up_verified_from_workbooks": False,
        "synthetic_reference_cases_only": True,
    }
    (REPO / "model/model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = synthetic_records(ranges, categories, 20, 20260920)
    background = synthetic_records(ranges, categories, 30, 20260921)
    frame = pd.DataFrame(cases)[lock["features"]]
    expected = pipeline.predict_proba(frame)[:, 1]
    from src.prediction import preprocess, predict_probability
    original_encoded = np.asarray(prep.transform(frame), dtype=np.float32)
    exported_encoded = np.vstack([preprocess(case)[0] for case in cases])
    if not np.allclose(original_encoded, exported_encoded, atol=1e-6, rtol=0):
        raise RuntimeError("Native preprocessing differs from frozen pipeline")
    actual = np.array([predict_probability(case) for case in cases])
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-6:
        raise RuntimeError(f"Native export prediction difference {error} exceeds tolerance")
    frame["expected_probability"] = expected
    (REPO / "tests").mkdir(exist_ok=True)
    frame.to_csv(REPO / "tests/reference_predictions.csv", index=False)
    (REPO / "model/synthetic_background.json").write_text(
        json.dumps(background, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"synthetic_cases": len(cases), "maximum_absolute_difference": error,
              "tolerance": 1e-6, "source_model_sha256": lock["model_sha256"],
              "native_model_sha256": sha256(model_target), "patient_rows_exported": 0}
    (REPO / "tests/export_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
