# Final Model Evaluation Summary

Generated at: 2026-07-10 10:28:58

## Coverage

- CICIDS2017 DL: evaluated = 4
- CICIDS2017 ML: evaluated = 4
- NSL-KDD DL: evaluated = 4
- NSL-KDD ML: evaluated = 4
- UNSW-NB15 DL: evaluated = 4
- UNSW-NB15 ML: evaluated = 4

## Best Model by Dataset

- NSL-KDD: ML LightGBM with macro-F1 0.9658, weighted-F1 0.9992, accuracy 0.9992, FAR 0.0005.
- UNSW-NB15: ML LightGBM with macro-F1 0.5536, weighted-F1 0.8817, accuracy 0.8761, FAR 0.0348.
- CICIDS2017: ML Random Forest with macro-F1 0.9762, weighted-F1 0.9985, accuracy 0.9986, FAR 0.0007.

## Best ML vs Best DL

- NSL-KDD: best ML = LightGBM (0.9658); best DL = CNN1D (0.7811); DL minus ML gap = -0.1847. ML is ahead by macro-F1.
- UNSW-NB15: best ML = LightGBM (0.5536); best DL = MLP (0.4759); DL minus ML gap = -0.0777. ML is ahead by macro-F1.
- CICIDS2017: best ML = Random Forest (0.9762); best DL = BiLSTM Attention (0.7248); DL minus ML gap = -0.2514. ML is ahead by macro-F1.

## Missing or Train-Only Items

- All expected model-dataset combinations have evaluation metrics.

## Output Files

- Final evaluation table: `G:\My Drive\nids_deep_learning\ids_deep_learning\results\final_evaluation\final_model_evaluation.csv`
- Ranking table: `G:\My Drive\nids_deep_learning\ids_deep_learning\results\final_evaluation\final_model_ranking.csv`
- Per-class metrics: `G:\My Drive\nids_deep_learning\ids_deep_learning\results\final_evaluation\final_per_class_metrics.csv`
- ML vs DL comparison: `G:\My Drive\nids_deep_learning\ids_deep_learning\results\final_evaluation\final_ml_vs_dl_comparison.csv`