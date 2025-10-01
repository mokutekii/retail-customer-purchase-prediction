# Retail Customer Purchase Prediction
COSC 4368 final project (AI) — supervised classification of purchase intent from online retail session data.

> **Status:** (structure only). Add your data, code, and results as you proceed.

## 1) Overview & Motivation
Predict whether an online shopping **session** results in a purchase. This project lets us practice an end-to-end ML pipeline (preprocessing → models → evaluation → ablation) and build e‑commerce intuition (what behaviors signal conversion).

## 2) Datasets
- **Primary:** UCI *Online Shoppers Purchasing Intention* (~12k sessions; label `Revenue`). Place CSV under `data/`.
- **Optional:** *Online Retail II* (UCI/Kaggle). If used, aggregate to RFM features. Due to size, do **not** commit raw files.

See `docs/` for course instructions PDF and proposal (if included).

## 3) Project Structure
```
retail-customer-purchase-prediction/
├─ data/                 # raw/processed data (git-ignored, .gitkeep placeholder)
├─ notebooks/            # 1_data_exploration.ipynb, 2_baseline_models.ipynb, ...
├─ src/                  # reusable code (data loading, preprocessing, train, eval)
├─ outputs/              # models, plots, metrics (git-ignored, .gitkeep placeholder)
├─ report/               # NeurIPS-style LaTeX skeleton (report.tex + README)
├─ docs/                 # project instructions, proposal, notes
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 4) Environment Setup
```bash
git clone <your_repo_url>
cd retail-customer-purchase-prediction
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# Put online_shoppers_intention.csv into data/
jupyter lab
```

## 5) How to Run (planned flow)
1. **EDA & Split:** run `notebooks/1_data_exploration.ipynb` (inspect data, leakage audit, train/val/test split).
2. **Baselines:** `notebooks/2_baseline_models.ipynb` (Logistic Regression, Decision Tree).
3. **Advanced:** `notebooks/3_advanced_models.ipynb` (Random Forest, (optional) Gradient Boosting, MLP).
4. **Eval & Ablations:** `notebooks/4_evaluation_and_ablation.ipynb` (ROC/F1, confusion matrix, feature importance, ablations).

> Script mode (optional, to be implemented): `python src/train.py --model logistic` then `python src/evaluate.py --model outputs/logreg.pkl --data test`.

## 6) Methods (scope)
- Baseline: **Logistic Regression**.
- Ensembles: **Decision Tree → Random Forest → (if permitted) Gradient Boosting**.
- Neural net: **MLP** (ReLU, dropout/L2, sigmoid output).
- **Metrics:** ROC‑AUC (primary), F1/Precision/Recall (class imbalance), Accuracy (context).

## 7) Collaboration Guidelines
- **Branches/PRs** for features (`feature/preprocessing`, `feature/rf`, `feature/mlp`, `report`).
- Keep notebooks focused; place reusable logic in `src/`.
- Do **not** commit large data/artifacts; use `data/` and `outputs/` locally.

## 8) Timeline (to Dec 10)
- Week 1–2: setup, EDA, leakage checks, LR + DT baselines.
- Week 3–4: RF (+Boosting if used), MLP prototype.
- Week 5–6: tuning, plots, ablations.
- Week 7–8: slides + draft report; present (Dec 1/3). Finalize report by **Dec 10**.

