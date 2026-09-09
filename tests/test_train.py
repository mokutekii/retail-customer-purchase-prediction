import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from retail_purchase.artifacts import PredictionBundle
from retail_purchase.train import (
    choose_threshold,
    feature_groups,
    load_dataset,
    parse_args,
)


def test_feature_groups_treat_integer_codes_as_categories():
    frame = pd.DataFrame(
        {
            "PageValues": [0.0, 2.5],
            "Region": [1, 2],
            "Month": ["May", "Jun"],
            "Weekend": [True, False],
        }
    )
    numeric, categorical = feature_groups(frame)
    assert numeric == ["PageValues"]
    assert categorical == ["Month", "Region", "Weekend"]


def test_threshold_comes_from_validation_scores_only():
    threshold, table = choose_threshold(pd.Series([0, 1, 0, 1]), pd.Series([0.1, 0.8, 0.3, 0.7]))
    assert 0.05 <= threshold <= 0.95
    assert {"threshold", "f1", "precision", "recall"}.issubset(table.columns)


def test_load_dataset_rejects_non_binary_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"Revenue": ["yes", "maybe"], "PageValues": [0.0, 1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_dataset(path)


def test_prediction_bundle_applies_its_saved_threshold_and_feature_contract(tmp_path):
    features = pd.DataFrame({"PageValues": [0.0, 2.0], "Month": ["May", "Jun"]})
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    bundle = PredictionBundle(model, threshold=0.75, feature_columns=("PageValues", "Month"))

    probability = bundle.predict_proba(features[["Month", "PageValues"]])[:, 1]
    assert (bundle.predict(features) == (probability >= 0.75)).all()
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(bundle, artifact_path)
    assert (joblib.load(artifact_path).predict(features) == bundle.predict(features)).all()
    with pytest.raises(ValueError, match="unexpected"):
        bundle.predict(features.assign(extra=1))


def test_parse_args_accepts_explicit_data_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = parse_args(["--data", "input.csv"])
    assert config.data_path == (tmp_path / "input.csv").resolve()
    assert config.output_dir == (tmp_path / "artifacts").resolve()
