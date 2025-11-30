# Poster Outline — UH Template
Last updated: 2025-11-30

Format: 3 columns (Left / Middle / Right). Minimize text; emphasize figures. Use UH PPTX template and swap images.

## Header
- Title, authors, affiliations, course
- QR to repo

## Left Column
### Abstract (≤150 words)
Problem, dataset, method families, headline results (ROC-AUC + F1), business relevance.

### Background
Purchase prediction on tabular session data; why compare linear, trees/ensembles, MLP; importance of thresholding and calibration.

### Data
UCI Online Shoppers Intention: size, key features, class imbalance.
- Figure: `01_class_counts.png`.

## Middle Column
### Methods
Leakage-safe Pipeline (ColumnTransformer → model); 70/15/15 split; seed 42.
Models: LR, DT, RF (CV), (opt) GB, MLP.
Validation threshold sweep; calibration.

### Results
- ROC (test): `02_roc_curve_test.png`
- Confusion Matrix (test): `03_confusion_matrix_test.png`
- Calibration: `purchase_pred_calibration_test.png` or `calibration_curve_test.png`
- (Optional) Precision–Recall

## Right Column
### Interpretability
Feature importance / coefficients:
- Figure: `04_feature_importance_top20.png` (or `_top25.png`)
- 3 succinct bullets on key drivers.

### Ablations
Behavioral vs Temporal/Tech feature groups:
- Figure: `05_ablation_delta_auc.png`
- One-sentence conclusion.

### Business Takeaways & Future
Short action list (threshold, segments, behaviors).
Future: RFM enrichment, cost-sensitive metrics, deployment.

### Acknowledgments
Course staff; dataset sources.
