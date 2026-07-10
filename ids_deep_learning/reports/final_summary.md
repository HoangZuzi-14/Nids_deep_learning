# NIDS Deep Learning - Submission Summary

This folder contains curated, submission-ready artifacts generated from local experiment outputs.
Raw datasets, trained checkpoints, cache files, and long training logs are intentionally kept out of Git.

## Best model by dataset

| dataset | family | best_model | accuracy | macro_f1 | weighted_f1 | far | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | ML | LightGBM | 0.9992 | 0.9658 | 0.9992 | 0.0005 | 20.74 |
| UNSW-NB15 | ML | LightGBM | 0.8761 | 0.5536 | 0.8817 | 0.0348 | 11.10 |
| CICIDS2017 | ML | Random Forest | 0.9986 | 0.9762 | 0.9985 | 0.0007 | 1703.98 |

## Best ML vs best DL

| dataset_label | best_ml_model | best_ml_macro_f1 | best_dl_model | best_dl_macro_f1 | macro_f1_gap_dl_minus_ml |
| --- | --- | --- | --- | --- | --- |
| NSL-KDD | LightGBM | 0.9658 | CNN1D | 0.7811 | -0.1847 |
| UNSW-NB15 | LightGBM | 0.5536 | MLP | 0.4759 | -0.0777 |
| CICIDS2017 | Random Forest | 0.9762 | BiLSTM Attention | 0.7248 | -0.2514 |

## Hybrid Decision Layer

The Hybrid Decision Layer is separate from the CNN-LSTM Hybrid model. It combines a supervised classifier with an anomaly detector to recover attacks that the classifier predicts as benign.

| dataset | setting | accuracy | far | attack_recall | f1 | missed_attacks | recovered_attacks | recovery_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSL-KDD | Classifier alone | 0.9983 | 0.0003 | 0.9967 | 0.9982 |  |  |  |
| NSL-KDD | Hybrid Decision (far_0.03) | 0.9850 | 0.0281 | 1.0000 | 0.9841 | 39 | 39 | 1.0000 |
| UNSW-NB15 | Classifier alone | 0.9599 | 0.0114 | 0.9147 | 0.9466 |  |  |  |
| UNSW-NB15 | Hybrid Decision (far_0.03) | 0.9637 | 0.0306 | 0.9547 | 0.9534 | 371 | 174 | 0.4690 |
| CICIDS2017 | RandomForest multiclass baseline | 0.9982 | 0.0008 |  | 0.9732 |  |  |  |
| CICIDS2017 | Hybrid RF + IsolationForest (far_0.03) | 0.9748 | 0.0290 | 0.9938 | 0.9303 | 53 | 0 | 0.0000 |

## Included figures

- `figures/final/`: macro-F1, weighted-F1, FAR, heatmap, trade-off plots, and selected confusion matrices.
- `figures/hybrid/`: hybrid recovery percentage and risk-score distributions.

## Included tables

- `tables/final_model_evaluation.csv`: full ML/DL evaluation table.
- `tables/final_model_ranking.csv`: ranking across datasets and model families.
- `tables/final_ml_vs_dl_comparison.csv`: best ML vs best DL comparison.
- `tables/final_per_class_metrics.csv`: per-class metrics for detailed discussion.
- `tables/hybrid_decision_summary.csv`: classifier vs hybrid decision trade-offs.

## Notebook

Open `notebooks/final_project_report.ipynb` to view the report with preserved summary tables and linked figures.
