"""CLI training entrypoint (skeleton).

Usage examples:
    python src/train.py --model logistic
    python src/train.py --model rf --save outputs/rf.pkl
"""
import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data_loader import load_uci
from preprocessing import build_preprocess_pipeline, train_val_test_split, TARGET
from models import make_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="logistic", help="logistic|dt|rf|gb")
    ap.add_argument("--save", type=str, default=None, help="path to save fitted model (joblib)")
    args = ap.parse_args()

    df = load_uci()
    y = df[TARGET].astype(int) if df[TARGET].dtype != int else df[TARGET]
    X = df.drop(columns=[TARGET])

    pipe, _ = build_preprocess_pipeline(df)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # Fit transform on train only
    X_train_t = pipe.fit_transform(X_train)
    X_val_t = pipe.transform(X_val)

    model = make_model(args.model)
    model.fit(X_train_t, y_train)

    # Simple validation metrics
    val_proba = getattr(model, "predict_proba", None)
    if val_proba is not None:
        y_val_pred_proba = model.predict_proba(X_val_t)[:,1]
    else:
        # some models may not have predict_proba
        y_val_pred_proba = model.decision_function(X_val_t)

    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    roc = roc_auc_score(y_val, y_val_pred_proba)
    f1 = f1_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred)

    print(f"Validation ROC-AUC: {roc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocess": pipe, "model": model}, out)
        print(f"Saved model to {out}")

if __name__ == "__main__":
    main()
