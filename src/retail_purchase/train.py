"""Train, calibrate, and evaluate a retail-session purchase classifier.

The notebook remains useful for exploration. This module is the authoritative,
repeatable path for producing a model and its machine-readable evaluation files.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifacts import PredictionBundle

TARGET = "Revenue"
# These columns are coded as integers in the source data but represent labels, not quantities.
NOMINAL_COLUMNS = {"OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"}


@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    output_dir: Path
    seed: int = 42
    cv_folds: int = 5
    fast: bool = False


def repository_root(start: Path | None = None) -> Path:
    """Find the project root so the CLI works from any current directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find project root (expected pyproject.toml and data/).")


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI data and return validated features and a binary target."""
    frame = pd.read_csv(data_path)
    if TARGET not in frame:
        raise ValueError(f"Expected target column {TARGET!r}; found {sorted(frame.columns)}")
    if frame.empty:
        raise ValueError("The training data is empty.")

    raw_target = frame.pop(TARGET)
    if raw_target.dtype == bool:
        target = raw_target.astype("int8")
    else:
        normalized = raw_target.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        if not normalized.isin(mapping).all():
            raise ValueError(f"{TARGET} must be binary; received {sorted(normalized.unique())}")
        target = normalized.map(mapping).astype("int8")
    if target.nunique() != 2:
        raise ValueError(f"{TARGET} must contain both classes.")
    return frame, target


def feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate continuous values from categorical labels, including integer-coded labels."""
    categorical = sorted(
        column
        for column in features.columns
        if column in NOMINAL_COLUMNS
        or pd.api.types.is_bool_dtype(features[column])
        or pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    )
    numeric = sorted(column for column in features.columns if column not in categorical)
    if not numeric and not categorical:
        raise ValueError("No feature columns were found.")
    return numeric, categorical


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_groups(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_searches(features: pd.DataFrame, config: RunConfig) -> dict[str, GridSearchCV]:
    """Create leakage-safe hyperparameter searches; preprocessing is fit inside every fold."""
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=config.seed),
            {"model__C": [0.5, 1.0] if config.fast else [0.1, 0.5, 1.0, 2.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=150 if config.fast else 400,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.seed,
            ),
            {
                "model__max_depth": [12] if config.fast else [12, None],
                "model__min_samples_leaf": [2] if config.fast else [1, 2, 5],
            },
        ),
    }
    return {
        name: GridSearchCV(
            Pipeline([("preprocess", make_preprocessor(features)), ("model", estimator)]),
            param_grid=parameters,
            scoring="roc_auc",
            cv=folds,
            # The random forest parallelizes its trees itself; avoid nested process pools.
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        for name, (estimator, parameters) in candidates.items()
    }


def metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
    }


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose an operating point only on validation data, never on the test set."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [{"threshold": float(t), **metrics(y_true, probability, float(t))} for t in thresholds]
    table = pd.DataFrame(rows)
    # Prefer recall only as a deterministic tie-breaker when F1 is identical.
    best = table.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), table


def run(config: RunConfig) -> dict[str, Any]:
    features, target = load_dataset(config.data_path)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=0.30, stratify=target, random_state=config.seed
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=config.seed
    )

    searches = candidate_searches(x_train, config)
    selection_rows: list[dict[str, Any]] = []
    fitted: dict[str, GridSearchCV] = {}
    for name, search in searches.items():
        search.fit(x_train, y_train)
        fitted[name] = search
        selection_rows.append(
            {
                "model": name,
                "cv_roc_auc": float(search.best_score_),
                "best_parameters": json.dumps(search.best_params_),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values("cv_roc_auc", ascending=False)
    champion_name = str(selection.iloc[0]["model"])

    # Calibration uses folds from training data. Validation remains untouched until threshold selection.
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    champion = CalibratedClassifierCV(
        fitted[champion_name].best_estimator_, method="sigmoid", cv=folds
    )
    champion.fit(x_train, y_train)
    validation_probability = champion.predict_proba(x_validation)[:, 1]
    threshold, threshold_table = choose_threshold(y_validation, validation_probability)
    test_probability = champion.predict_proba(x_test)[:, 1]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(config.output_dir / "model_selection.csv", index=False)
    threshold_table.to_csv(config.output_dir / "validation_thresholds.csv", index=False)
    bundle = PredictionBundle(
        model=champion,
        threshold=threshold,
        feature_columns=tuple(features.columns),
    )
    joblib.dump(bundle, config.output_dir / "model.joblib")

    result = {
        "config": {
            **asdict(config),
            "data_path": str(config.data_path),
            "output_dir": str(config.output_dir),
        },
        "feature_groups": dict(zip(("numeric", "categorical"), feature_groups(features))),
        "selected_model": champion_name,
        "threshold": threshold,
        "validation_metrics": metrics(y_validation, validation_probability, threshold),
        "test_metrics": metrics(y_test, test_probability, threshold),
        "split_rows": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="CSV to train from; defaults to data/online_shoppers_intention.csv in a repository checkout.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for generated artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--fast", action="store_true", help="Use a reduced tuning grid for a smoke test."
    )
    args = parser.parse_args(argv)
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    root: Path | None = None
    if args.data is None:
        try:
            root = repository_root()
        except FileNotFoundError:
            parser.error("--data is required when running outside a repository checkout.")
        data_path = root / "data" / "online_shoppers_intention.csv"
    else:
        data_path = args.data
    output_dir = args.output_dir or ((root / "artifacts") if root else (Path.cwd() / "artifacts"))
    return RunConfig(
        data_path.expanduser().resolve(),
        output_dir.expanduser().resolve(),
        args.seed,
        args.cv_folds,
        args.fast,
    )


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {"selected_model": result["selected_model"], "test_metrics": result["test_metrics"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
