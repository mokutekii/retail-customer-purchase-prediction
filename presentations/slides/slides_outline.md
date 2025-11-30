# Slide Deck Outline — Retail Customer Purchase Prediction
Last updated: 2025-11-30

Target length: 12–15 slides (8–10 minutes). Use numbered assets from `presentations/assets/` that the notebook exports.

## 1. Title
- Retail Customer Purchase Prediction
- Authors, course/semester, QR to repo
- One-line promise: “Predict purchase intent from session behavior; compare ML vs MLP; extract actionable levers.”

## 2–3. Motivation & Problem
- Business: many visits don’t convert → intent signals guide UX and remarketing.
- Academic: compare AI-4368 model families on tabular data; trade-offs between interpretability and accuracy.
- Problem: Binary classification per session: `Revenue ∈ {0,1}`.

## 4. Data
- UCI Online Shoppers Intention: N rows, target, key features, class imbalance.
- Optional: Retail II → RFM (if added later).
- Figure: `01_class_counts.png`.

## 5–6. Methods (Pipeline)
- Leakage-safe `ColumnTransformer`: scale numeric, OHE categoricals.
- Split: 70/15/15 train/val/test; `seed=42`.
- Models: Logistic Regression, Decision Tree, Random Forest (CV), (opt) Gradient Boosting, MLP.
- Threshold selection by validation sweep; calibration (reliability + Brier).

## 7–9. Results
- ROC (test): `02_roc_curve_test.png`.
- Confusion Matrix (test): `03_confusion_matrix_test.png` with chosen threshold.
- Precision–Recall (optional, if helps show imbalance).
- Calibration: `purchase_pred_calibration_test.png` or `calibration_curve_test.png`.

## 10. Interpretability
- Feature importance / coefficients: `04_feature_importance_top20.png` or `feature_importance_top25.png`.
- 2–3 bullets on the strongest positive/negative drivers.

## 11. Ablations
- Compare Behavioral vs Temporal/Tech feature groups.
- Figure: `05_ablation_delta_auc.png`.
- One-sentence takeaway: where most signal lives.

## 12. Business Takeaways
- 4–6 bullets: behaviors and segments to act on; recommended operating threshold and trade-offs.

## 13. Limitations & Future Work
- Session-only view; dataset bias; no causality.
- Future: cost-sensitive objectives, better calibration, RFM integration, deployment.

## 14. Acknowledgments & Repo
- Course staff, dataset maintainers.
- Link/QR to repo; assets path reminder.
