## Modular baseline runner

Run leakage-safe ML baselines from the project root:

```bash
python -m src.pipeline.baseline_runner --dataset nsl_kdd --classification multi
```

Supported dataset keys: `nsl_kdd`, `unsw_nb15`, `cicids2017`.

The runner uses dataset adapters, fits preprocessing on train only, saves metrics under `results/`, and saves model/scaler/encoder artifacts under `artifacts/`.

## Current direction

The project direction is Hybrid NIDS:

- Known attack classification for labeled attacks seen during training.
- Unknown anomaly detection for unseen attacks, rare classes, and behavior that does not have enough supervised samples.
- Adaptive imbalance handling without GAN in the default path.

For CICIDS2017, the current conclusion is:

- RandomForest is the primary known-attack classifier.
- MLP remains a deep-learning baseline.
- Focal loss and weighted sampling raised FAR and are not selected for the main pipeline.
- CE + class weight is the most stable DL imbalance option, but it still does not beat RandomForest.
- Rare/unknown behavior should be handled by a separate anomaly branch, not by forcing all rare CICIDS2017 labels through oversampling.

The legacy GAN result is kept as optional/future work, not as the main project direction.

## Anomaly detection and hybrid results

The previous SOC log anomaly detection project is being reused selectively for the Hybrid NIDS direction. The first integrated branch is flow-feature Isolation Forest:

- Train scope: Benign-only rows from the training split.
- Threshold tuning: Benign validation rows at target FAR values.
- Evaluation: full test split as Benign vs Attack anomaly detection.
- Output: `results/cicids2017_multi_IsolationForest_BenignOnly_10pct_results.json` and `results/cicids2017_multi_IsolationForest_BenignOnly_25pct_results.json`.

Current result:

- Isolation Forest at threshold `far_0.03` is stable around 3% FAR and about 40% attack recall.
- Hybrid RF OR IF was tested and rejected for known CICIDS2017 labels: it recovered 0 of 53 RandomForest missed attacks on the 10% hybrid sample while increasing FAR from 0.000835 to 0.029001.
- Therefore, Isolation Forest is kept as an optional suspicious/unknown detector, not as an override for RandomForest on known CICIDS2017 classification.

DeepLog/LSTM from the SOC project is relevant for future log-sequence detection, but it is not directly used for CICIDS2017 flow CSV because the input type is different.

## Colab focal-sampler experiment

After the notebook copies CICIDS2017 CSV files to Colab local disk, run:

```bash
python -m src.pipeline.dl_runner \
  --dataset cicids2017 \
  --classification multi \
  --experiment MLP_FocalSampler \
  --focal-loss \
  --weighted-sampler \
  --cicids-raw-dir /content/cicids2017_raw \
  --cicids-cache-path /content/cicids2017_cache/merged.csv
```

This saves the result to `results/cicids2017_multi_MLP_FocalSampler_results.json` and the checkpoint to `artifacts/cicids2017/multi/MLP_FocalSampler.pt`.
