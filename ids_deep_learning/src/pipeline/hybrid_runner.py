from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.anomaly import compute_anomaly_scores
from src.evaluation.metrics import compute_classification_metrics
from src.pipeline.baseline_runner import benign_label_index, build_adapter
from src.utils.seed import set_seed


def _sample_split(X, y, fraction: float, seed: int):
    if fraction >= 1.0:
        return X, y
    if fraction <= 0.0:
        raise ValueError("sample_fraction must be greater than 0 and less than or equal to 1.")
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=fraction,
        random_state=seed,
        stratify=y,
    )
    return X_sample, y_sample


def _sample_output(output, fraction: float, seed: int):
    if fraction >= 1.0:
        return output
    X_test, y_test = _sample_split(output.X_test, output.y_test, fraction, seed)
    metadata = dict(output.metadata)
    metadata["hybrid_test_sample_fraction"] = fraction
    return replace(output, X_test=X_test, y_test=y_test, metadata=metadata)


def _binary_metrics(y_true_attack, y_pred_attack):
    return {
        "accuracy": float(accuracy_score(y_true_attack, y_pred_attack)),
        "precision": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
    }


def evaluate_hybrid_from_output(
    output,
    root: Path,
    dataset_key: str = "cicids2017",
    classification_type: str = "multi",
    experiment_name: str = "Hybrid_RF_IF_10pct",
    rf_artifact: Path | None = None,
    if_artifact: Path | None = None,
    if_threshold_key: str = "far_0.03",
    test_sample_fraction: float = 1.0,
):
    seed = 42
    try:
        experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
        seed = int(experiments_config.get("seed", 42))
    except FileNotFoundError:
        pass
    set_seed(seed)
    output = _sample_output(output, test_sample_fraction, seed)

    artifact_dir = root / "artifacts" / dataset_key / classification_type
    rf_artifact = rf_artifact or artifact_dir / "RandomForest.pkl"
    if_artifact = if_artifact or artifact_dir / "anomaly" / "IsolationForest_BenignOnly_10pct.pkl"

    rf_model = joblib.load(rf_artifact)
    if_bundle = joblib.load(if_artifact)
    if_model = if_bundle["model"]
    if_scaler = if_bundle["scaler"]
    thresholds = if_bundle["thresholds"]
    if if_threshold_key not in thresholds:
        raise KeyError(f"Threshold {if_threshold_key} not found. Available: {sorted(thresholds)}")
    threshold = float(thresholds[if_threshold_key])

    benign = benign_label_index(output.label_mapping)
    y_true = np.asarray(output.y_test)
    y_true_attack = (y_true != benign).astype(int)

    rf_pred = np.asarray(rf_model.predict(output.X_test))
    rf_attack = (rf_pred != benign).astype(int)

    if_scores = compute_anomaly_scores(if_model, if_scaler, output.X_test)
    if_attack = (if_scores >= threshold).astype(int)

    hybrid_attack = ((rf_attack == 1) | ((rf_pred == benign) & (if_attack == 1))).astype(int)
    suspicious_unknown = ((rf_pred == benign) & (if_attack == 1)).astype(int)

    rf_metrics = compute_classification_metrics(y_true, rf_pred, benign_label=benign)
    if_metrics = _binary_metrics(y_true_attack, if_attack)
    hybrid_metrics = _binary_metrics(y_true_attack, hybrid_attack)

    benign_mask = y_true == benign
    attack_mask = ~benign_mask
    rf_missed_attack_mask = attack_mask & (rf_attack == 0)
    if_metrics.update(
        {
            "far": float(if_attack[benign_mask].mean()) if benign_mask.any() else 0.0,
            "attack_recall": float(if_attack[attack_mask].mean()) if attack_mask.any() else 0.0,
        }
    )
    hybrid_metrics.update(
        {
            "far": float(hybrid_attack[benign_mask].mean()) if benign_mask.any() else 0.0,
            "attack_recall": float(hybrid_attack[attack_mask].mean()) if attack_mask.any() else 0.0,
            "suspicious_unknown_alerts": int(suspicious_unknown.sum()),
            "rf_missed_attacks": int(rf_missed_attack_mask.sum()),
            "rf_missed_attacks_caught_by_if": int((rf_missed_attack_mask & (if_attack == 1)).sum()),
            "rf_missed_attack_recovery_rate": float(
                (rf_missed_attack_mask & (if_attack == 1)).sum() / rf_missed_attack_mask.sum()
            )
            if rf_missed_attack_mask.any()
            else 0.0,
        }
    )

    report = {
        "dataset": dataset_key,
        "classification": classification_type,
        "experiment": experiment_name,
        "test_sample_fraction": test_sample_fraction,
        "benign_label": benign,
        "label_mapping": output.label_mapping,
        "rf_artifact": str(rf_artifact.relative_to(root)),
        "if_artifact": str(if_artifact.relative_to(root)),
        "if_threshold_key": if_threshold_key,
        "if_threshold": threshold,
        "n_test": int(len(y_true)),
        "n_test_benign": int(benign_mask.sum()),
        "n_test_attack": int(attack_mask.sum()),
        "rf_multiclass": rf_metrics,
        "if_binary": if_metrics,
        "hybrid_binary": hybrid_metrics,
    }

    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{dataset_key}_{classification_type}_{experiment_name}_results.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_from_config(
    root: Path,
    dataset_key: str,
    classification_type: str,
    experiment_name: str,
    test_sample_fraction: float,
    if_threshold_key: str,
):
    datasets_config = yaml.safe_load((root / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    split_cfg = experiments_config.get("split", {})
    adapter = build_adapter(dataset_key, datasets_config, root, seed)
    output = adapter.preprocess(
        classification_type=classification_type,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        scaler_type=experiments_config.get("preprocessing", {}).get("scaler", "standard"),
    )
    return evaluate_hybrid_from_output(
        output=output,
        root=root,
        dataset_key=dataset_key,
        classification_type=classification_type,
        experiment_name=experiment_name,
        test_sample_fraction=test_sample_fraction,
        if_threshold_key=if_threshold_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid known-attack + anomaly detection.")
    parser.add_argument("--dataset", choices=["cicids2017"], default="cicids2017")
    parser.add_argument("--classification", choices=["multi"], default="multi")
    parser.add_argument("--experiment", default="Hybrid_RF_IF_10pct")
    parser.add_argument("--test-sample-fraction", type=float, default=1.0)
    parser.add_argument("--if-threshold-key", default="far_0.03")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = run_from_config(
        root=args.root,
        dataset_key=args.dataset,
        classification_type=args.classification,
        experiment_name=args.experiment,
        test_sample_fraction=args.test_sample_fraction,
        if_threshold_key=args.if_threshold_key,
    )
    print(json.dumps({"if_binary": report["if_binary"], "hybrid_binary": report["hybrid_binary"]}, indent=2))


if __name__ == "__main__":
    main()
