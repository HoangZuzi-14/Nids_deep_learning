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

from src.anomaly import compute_anomaly_scores, fit_isolation_forest, threshold_at_far
from src.pipeline.baseline_runner import benign_label_index, build_adapter
from src.utils.seed import set_seed


def _sample_split(X, y, fraction: float, seed: int):
    if fraction >= 1.0:
        return X, y
    if fraction <= 0.0:
        raise ValueError("--sample-fraction must be greater than 0 and less than or equal to 1.")
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
    X_train, y_train = _sample_split(output.X_train, output.y_train, fraction, seed)
    X_val, y_val = _sample_split(output.X_val, output.y_val, fraction, seed)
    X_test, y_test = _sample_split(output.X_test, output.y_test, fraction, seed)
    metadata = dict(output.metadata)
    metadata["sample_fraction"] = fraction
    return replace(
        output,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata,
    )


def binary_metrics(y_true_attack, y_pred_attack):
    return {
        "accuracy": float(accuracy_score(y_true_attack, y_pred_attack)),
        "precision": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
    }


def evaluate_by_threshold(scores, y_true, benign_label: int, threshold: float):
    y_true = np.asarray(y_true)
    y_pred_attack = (np.asarray(scores) >= threshold).astype(int)
    y_true_attack = (y_true != benign_label).astype(int)
    benign_mask = y_true == benign_label
    attack_mask = ~benign_mask
    far = float(y_pred_attack[benign_mask].mean()) if benign_mask.any() else 0.0
    attack_recall = float(y_pred_attack[attack_mask].mean()) if attack_mask.any() else 0.0
    metrics = binary_metrics(y_true_attack, y_pred_attack)
    metrics.update(
        {
            "far": far,
            "attack_recall": attack_recall,
            "threshold": float(threshold),
            "n_alerts": int(y_pred_attack.sum()),
            "n_samples": int(len(y_pred_attack)),
        }
    )
    return metrics


def run_isolation_forest_experiment(
    output,
    dataset_key: str,
    classification_type: str,
    root: Path,
    experiment_name: str = "IsolationForest_BenignOnly",
    sample_fraction: float = 1.0,
    target_fars: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1),
    n_estimators: int = 200,
):
    experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    set_seed(seed)
    output = _sample_output(output, sample_fraction, seed)

    benign = benign_label_index(output.label_mapping)
    train_normal_mask = np.asarray(output.y_train) == benign
    val_normal_mask = np.asarray(output.y_val) == benign
    if not train_normal_mask.any():
        raise ValueError("No benign samples found in training split.")
    if not val_normal_mask.any():
        raise ValueError("No benign samples found in validation split.")

    model, scaler = fit_isolation_forest(
        output.X_train[train_normal_mask],
        contamination="auto",
        n_estimators=n_estimators,
        seed=seed,
    )
    val_scores = compute_anomaly_scores(model, scaler, output.X_val)
    test_scores = compute_anomaly_scores(model, scaler, output.X_test)

    thresholds = {
        f"far_{target_far:g}": threshold_at_far(val_scores[val_normal_mask], target_far)
        for target_far in target_fars
    }
    evaluations = {
        name: evaluate_by_threshold(test_scores, output.y_test, benign, threshold)
        for name, threshold in thresholds.items()
    }

    report = {
        "dataset": dataset_key,
        "classification": classification_type,
        "experiment": experiment_name,
        "method": "isolation_forest",
        "train_scope": "benign_only",
        "sample_fraction": sample_fraction,
        "benign_label": benign,
        "label_mapping": output.label_mapping,
        "n_train_normal": int(train_normal_mask.sum()),
        "n_val_normal": int(val_normal_mask.sum()),
        "score_stats": {
            "test_min": float(np.min(test_scores)),
            "test_max": float(np.max(test_scores)),
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores)),
        },
        "evaluations": evaluations,
    }

    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{dataset_key}_{classification_type}_{experiment_name}_results.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    artifact_dir = root / "artifacts" / dataset_key / classification_type / "anomaly"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "thresholds": thresholds}, artifact_dir / f"{experiment_name}.pkl")
    return report


def run_from_config(
    dataset_key: str,
    classification_type: str,
    root: Path,
    sample_fraction: float,
    experiment_name: str,
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
    return run_isolation_forest_experiment(
        output=output,
        dataset_key=dataset_key,
        classification_type=classification_type,
        root=root,
        experiment_name=experiment_name,
        sample_fraction=sample_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection experiments.")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], required=True)
    parser.add_argument("--classification", choices=["binary", "multi"], default="multi")
    parser.add_argument("--experiment", default="IsolationForest_BenignOnly")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = run_from_config(
        dataset_key=args.dataset,
        classification_type=args.classification,
        root=args.root,
        sample_fraction=args.sample_fraction,
        experiment_name=args.experiment,
    )
    print(json.dumps(report["evaluations"], indent=2))


if __name__ == "__main__":
    main()
