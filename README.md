# Retail Customer Purchase Prediction — Notebook-First Repo

Pipeline to predict whether an online session results in a purchase. Aligned with course techniques: classical ML, ensembles, MLP, cross-validation, calibration, thresholding, interpretability, and ablations. Exports slide/poster-ready figures.

Central deliverable is a **Jupyter Notebook** that generates all plots/tables for: **slides, poster, and NeurIPS-style report**.

## Reproducible training pipeline

The notebooks are retained for exploration and presentation assets. Use the package entry point for a repeatable training run: it keeps the test split sealed, selects the model by cross-validation on training data, calibrates the selected model, and chooses the classification threshold on validation data only.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks]"
python -m retail_purchase.train --fast
pytest
```

Outputs are written to `artifacts/`:

- `model.joblib` — calibrated model, selected threshold, and input-feature contract
- `model_selection.csv` — cross-validation model selection evidence
- `validation_thresholds.csv` — validation-only operating-point sweep
- `metrics.json` — split sizes, configuration, and final test metrics

The preprocessing explicitly one-hot encodes integer-coded labels such as `Region` and `TrafficType`, which should not be treated as ordered quantities.

## Docker

The Docker image runs the same CLI pipeline and includes only the primary, versioned UCI dataset. Results are bind-mounted into the local `artifacts/` directory.

```bash
docker compose run --rm train
# Full search rather than the default smoke-test grid:
docker compose run --rm train --cv-folds 5
```

Use `docker build --tag retail-purchase-prediction:local .` followed by `docker run --rm -v "${PWD}/artifacts:/app/artifacts" retail-purchase-prediction:local` if you prefer not to use Compose.


## Quick Start
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

pip install -r requirements.txt
jupyter lab  # open notebook/retail_purchase_prediction.ipynb
```


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

