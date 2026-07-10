# NIDS Deep Learning

This repository is a student research project for Network Intrusion Detection
System (NIDS) experiments on NSL-KDD, UNSW-NB15, and CICIDS2017. It is organized
as a reproducible experiment workspace: preprocessing adapters, ML baselines,
deep-learning training pipelines, evaluation notebooks, and a curated submission
report.

The main goal is comparative evaluation, not a production IDS deployment. The
project compares traditional ML models with DL models, then adds a Hybrid
Decision Layer as an exploratory analysis for recovering attacks missed by the
supervised classifier.

Project files live in `ids_deep_learning/`. Run experiment commands from that
directory:

```powershell
cd ids_deep_learning
```

Supported dataset keys: `nsl_kdd`, `unsw_nb15`, `cicids2017`.

## Submission View

The curated submission artifacts are under `ids_deep_learning/reports/`.

- `ids_deep_learning/reports/notebooks/final_project_report.ipynb`: compact report notebook with preserved summary outputs and linked figures.
- `ids_deep_learning/reports/final_summary.md`: quick written summary of the main results.
- `ids_deep_learning/reports/figures/`: selected charts, confusion matrices, and hybrid plots.
- `ids_deep_learning/reports/tables/`: final metrics, rankings, ML-vs-DL comparison, per-class metrics, and hybrid summary.

Raw datasets, caches, trained checkpoints, and long experiment outputs are not
committed. They are generated locally under `ids_deep_learning/data/`,
`ids_deep_learning/artifacts/`, `ids_deep_learning/saved_models/`, and
`ids_deep_learning/results/`.

## Current Direction

The project direction is Hybrid NIDS:

- Known attack classification for labeled attacks seen during training.
- Unknown anomaly detection for unseen attacks, rare classes, and behavior that does not have enough supervised samples.
- Hybrid decision analysis to recover attacks missed by the supervised classifier.
- Adaptive imbalance handling without GAN in the default path.

For CICIDS2017, the current conclusion is:

- Random Forest is the primary known-attack classifier.
- DL models remain useful baselines, especially MLP and BiLSTM Attention.
- Focal loss and weighted sampling raised FAR and are not selected for the main pipeline.
- Rare or unknown behavior should be handled by a separate anomaly branch, not by forcing all rare CICIDS2017 labels through oversampling.

The legacy GAN result is kept as optional/future work, not as the main project
direction.

## Datasets

The project supports:

- NSL-KDD
- UNSW-NB15
- CICIDS2017

Default dataset paths are configured in `ids_deep_learning/config/datasets.yaml`.

Expected local layout:

```text
ids_deep_learning/
  data/
    cache/
      nsl_kdd/
      unsw_nb15/
      cicids2017/
    raw/
      cicids2017/
```

NSL-KDD and UNSW-NB15 have remote URLs configured for convenience. CICIDS2017 is
large and should be placed locally or uploaded in Colab when running the CICIDS
notebook.

## Models

ML baselines:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

DL models:

- MLP
- CNN1D
- BiLSTM Attention
- CNN-LSTM Hybrid

Hybrid decision experiments:

- RandomForest plus Autoencoder risk scoring for NSL-KDD and UNSW-NB15
- RandomForest plus IsolationForest for CICIDS2017 inline sample evaluation

## Current Key Results

Best model by dataset from the final evaluation:

| Dataset | Best model | Family | Macro-F1 | Accuracy | FAR |
| --- | --- | --- | ---: | ---: | ---: |
| NSL-KDD | LightGBM | ML | 0.9658 | 0.9992 | 0.0005 |
| UNSW-NB15 | LightGBM | ML | 0.5536 | 0.8761 | 0.0348 |
| CICIDS2017 | Random Forest | ML | 0.9762 | 0.9986 | 0.0007 |

Hybrid decision highlights:

| Dataset | Setting | Missed attacks recovered | Recovery rate | FAR |
| --- | --- | ---: | ---: | ---: |
| NSL-KDD | Hybrid Decision far_0.03 | 39/39 | 1.0000 | 0.0281 |
| UNSW-NB15 | Hybrid Decision far_0.03 | 174/371 | 0.4690 | 0.0306 |
| CICIDS2017 | Hybrid RF + IsolationForest far_0.03 | 0/53 | 0.0000 | 0.0290 |

## Environment

Create and activate a virtual environment, then install dependencies:

```powershell
cd ids_deep_learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Experiments

Run ML baselines:

```powershell
python -m src.pipeline.baseline_runner --dataset nsl_kdd --classification multi
python -m src.pipeline.baseline_runner --dataset unsw_nb15 --classification multi
python -m src.pipeline.baseline_runner --dataset cicids2017 --classification multi
```

Run DL suite:

```powershell
python run_dl_suite.py --dataset nsl_kdd
python run_dl_suite.py --dataset unsw_nb15
python run_dl_suite.py --dataset cicids2017
```

Run a quick DL smoke test:

```powershell
python run_dl_suite.py --smoke-test
```

Run hybrid decision experiments:

```powershell
python -m src.pipeline.hybrid_decision --dataset nsl_kdd
python -m src.pipeline.hybrid_decision --dataset unsw_nb15
python -m src.pipeline.hybrid_runner --dataset cicids2017 --test-sample-fraction 0.1
```

## Rebuild Submission Reports

After new results are generated under `ids_deep_learning/results/`, rebuild the
curated submission folder:

```powershell
python scripts/build_submission_reports.py
```

This updates:

- `ids_deep_learning/reports/final_summary.md`
- `ids_deep_learning/reports/notebooks/final_project_report.ipynb`
- `ids_deep_learning/reports/figures/`
- `ids_deep_learning/reports/tables/`
- `ids_deep_learning/reports/manifest.json`

## Notebooks

Main notebooks:

- `ids_deep_learning/notebooks/cicids2017_colab_ml_train_eval.ipynb`: CICIDS2017 ML train/eval workflow for Colab.
- `ids_deep_learning/notebooks/dl_colab_train_only.ipynb`: DL training workflow for Colab.
- `ids_deep_learning/notebooks/evaluate_final_results.ipynb`: local final evaluation builder.
- `ids_deep_learning/reports/notebooks/final_project_report.ipynb`: submission notebook for review.

Training notebooks are kept clean to avoid noisy logs and large outputs. The
report notebook keeps the important table outputs and links to selected figures.

## Tests

Run the test suite:

```powershell
cd ids_deep_learning
python -m pytest -q
```

Some tests may skip when local datasets or trained artifacts are intentionally
not present in the repository.
