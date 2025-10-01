"""Evaluate a saved model on the held-out test set (skeleton).

Usage:
    python src/evaluate.py --model outputs/rf.pkl
"""
import argparse, joblib
from data_loader import load_uci
from preprocessing import train_val_test_split, TARGET
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to saved joblib (dict with preprocess+model)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["preprocess"]
    model = bundle["model"]

    df = load_uci()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    # reconstruct split deterministically (same params as training)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    X_test_t = pipe.transform(X_test)
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        y_prob = model.predict_proba(X_test_t)[:,1]
    else:
        y_prob = model.decision_function(X_test_t)
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test ROC-AUC: {roc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print("Confusion matrix:\n", cm)

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve (Test)")
    plt.savefig("outputs/roc_curve_test.png", dpi=150, bbox_inches="tight")
    print("Saved ROC curve to outputs/roc_curve_test.png")

if __name__ == "__main__":
    main()
