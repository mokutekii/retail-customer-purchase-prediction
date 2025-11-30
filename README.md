# Retail Customer Purchase Prediction — Notebook-First Repo

Pipeline to predict whether an online session results in a purchase. Aligned with course techniques: classical ML, ensembles, MLP, cross-validation, calibration, thresholding, interpretability, and ablations. Exports slide/poster-ready figures.

Central deliverable is a **Jupyter Notebook** that generates all plots/tables for: **slides, poster, and NeurIPS-style report**.


## Quick Start
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

pip install -r requirements.txt
jupyter lab  # open notebooks/retail_purchase_prediction.ipynb


## Data
- `data/online_shoppers_intention.csv` (UCI — primary)
- `data/online_retail_II.csv` (optional; RFM aggregation). Large files are git-ignored.

## Notebook Workflow
EDA → preprocessing → models (LR, DT/RF/(GB*), MLP) → metrics (ROC/F1/PR) → ablations → **save figures/tables** to `outputs/` → conclusions.

## Presentations
`presentations/slides/` and `/poster/` for exports; `/templates/` holds UH poster template and a reference deck.

## Report
NeurIPS LaTeX stub at `report/report.tex`. Add `neurips_2023.sty` to compile.

## Deliverables & Outlines

This repo is notebook-first and exports slide/poster-ready figures automatically.

- **Slides** → `presentations/slides/slides_outline.md` (use assets in `presentations/assets/`).
- **Poster** → `presentations/poster/poster_outline.md` (UH template; swap figures).
- **Report** → `report/report_outline.md` (NeurIPS-style headings; paste figures from `outputs/figures/<timestamp>/`).

### Recommended Figure Mapping
1. `01_class_counts.png` – Class balance
2. `02_roc_curve_test.png` – ROC (test)
3. `03_confusion_matrix_test.png` – Confusion Matrix (test)
4. `purchase_pred_calibration_test.png` or `calibration_curve_test.png` – Reliability
5. `04_feature_importance_top20.png` or `feature_importance_top25.png` – Feature importance
6. `05_ablation_delta_auc.png` – Ablations
7. `mlp_learning_curve.png` – Learning curve (optional)

### Build Order (fast track)
1. Run `notebooks/retail_purchase_prediction` top→bottom.
2. Collect exported figures from `presentations/assets/`.
3. Assemble slides, then poster, then report.  
4. Update `REFERENCES.md` (or `references.bib`) with final citations.

