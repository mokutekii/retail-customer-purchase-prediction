# Report Outline (NeurIPS-style headings)
Last updated: 2025-11-30

1. Title & Abstract (≤200 words)
   - Problem, dataset, methods, headline metrics, takeaway.

2. Introduction
   - Motivation (business + academic).
   - Contributions: (i) leakage-safe pipeline; (ii) model family comparison; (iii) threshold+calibration; (iv) ablations; (v) business takeaways.

3. Related Work / Background
   - Short context for purchase intent prediction on tabular data.
   - Model families used in AI-4368: linear vs trees/ensembles vs MLP; probability calibration; thresholding.

4. Data
   - Online Shoppers Intention: schema, target, quick EDA (class imbalance).
   - Leakage policy and preprocessing choices.

5. Methods
   - Pipeline (ColumnTransformer; OHE + scaling).
   - Split (70/15/15; seed=42).
   - Models: LR, DT, RF (CV grid), (opt) GB, MLP (ReLU, early stopping).
   - Threshold selection via validation sweep; calibration (reliability curve + Brier).

6. Experiments
   - Metrics: ROC-AUC (primary), F1/Precision/Recall; reason for choices.
   - Implementation notes (versions, seeds).

7. Results
   - Validation comparison table across families.
   - Test ROC/PR/CM at chosen threshold.
   - Calibration plot + Brier score.

8. Interpretability
   - Feature importance / coefficients; interpret top features and directionality.

9. Ablations
   - Behavioral vs Temporal/Tech groups; ΔAUC discussion.
   - (Optional) RFM extension plan/results if added.

10. Discussion / Business Takeaways
    - What changes purchase likelihood; how a business would act on the signals and threshold.

11. Conclusion & Future Work
    - Summary; next steps (cost-sensitive loss, improved calibration, RFM, deployment sketch).

12. References
    - Cite course notes, Goodfellow (DL), scikit-learn, dataset, calibration papers.

13. Reproducibility Appendix
    - Version table, seeds, hardware; path of all exported figures/tables.
