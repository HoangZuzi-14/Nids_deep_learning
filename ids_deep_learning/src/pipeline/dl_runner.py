from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_classification_metrics
from src.imbalance import (
    analyze_class_distribution,
    compute_balanced_class_weights,
    make_weighted_sampler,
    select_strategies,
)
from src.models import MLP, CNN1D, BiLSTMAttention, CNNLSTMHybrid
from src.training import TorchTrainer
from src.training.losses import FocalLoss
from src.utils.seed import set_seed

from .baseline_runner import benign_label_index, build_adapter, save_artifacts


def _tensor_dataset(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return TensorDataset(torch.tensor(X), torch.tensor(y))


def make_loader(X, y, batch_size: int, shuffle: bool = False, sampler=None):
    return DataLoader(
        _tensor_dataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
    )


def _override_cicids_paths(adapter, raw_dir: Path | None, cache_path: Path | None):
    if raw_dir is not None and hasattr(adapter, "raw_dir"):
        adapter.raw_dir = raw_dir
    if cache_path is not None and hasattr(adapter, "cache_path"):
        adapter.cache_path = cache_path


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


def build_dl_model(model_name: str, n_features: int, n_classes: int) -> torch.nn.Module:
    m = model_name.lower()
    if m == "mlp":
        return MLP(n_features=n_features, n_classes=n_classes)
    elif m == "cnn1d":
        return CNN1D(n_features=n_features, n_classes=n_classes)
    elif m == "bilstm":
        return BiLSTMAttention(n_features=n_features, n_classes=n_classes)
    elif m == "hybrid":
        return CNNLSTMHybrid(n_features=n_features, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from mlp, cnn1d, bilstm, hybrid.")


def run_dl_experiment(
    dataset_key: str,
    classification_type: str,
    root: Path,
    experiment_name: str,
    model_name: str = "mlp",
    use_focal_loss: bool = False,
    use_weighted_sampler: bool = False,
    focal_gamma: float = 2.0,
    sample_fraction: float = 1.0,
    cicids_raw_dir: Path | None = None,
    cicids_cache_path: Path | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    lr_override: float | None = None,
    patience_override: int | None = None,
) -> dict:
    datasets_config = yaml.safe_load((root / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    set_seed(seed)

    split_cfg = experiments_config.get("split", {})
    model_cfg = experiments_config.get("models", {})
    batch_size = batch_size_override if batch_size_override is not None else int(model_cfg.get("batch_size", 512))
    epochs = epochs_override if epochs_override is not None else int(model_cfg.get("epochs", 30))
    lr = lr_override if lr_override is not None else float(model_cfg.get("learning_rate", 1e-3))
    patience = patience_override if patience_override is not None else int(model_cfg.get("patience", 10))

    adapter = build_adapter(dataset_key, datasets_config, root, seed)
    if dataset_key == "cicids2017":
        _override_cicids_paths(adapter, cicids_raw_dir, cicids_cache_path)

    output = adapter.preprocess(
        classification_type=classification_type,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        scaler_type=experiments_config.get("preprocessing", {}).get("scaler", "standard"),
    )
    output = _sample_output(output, sample_fraction, seed)

    artifact_dir = root / "artifacts" / dataset_key / classification_type
    save_artifacts(output, artifact_dir)

    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    class_names = [None] * len(output.label_mapping)
    for name, idx in output.label_mapping.items():
        class_names[idx] = name
    imbalance_report = analyze_class_distribution(
        output.y_train,
        class_names=class_names,
        output_path=result_dir / f"{dataset_key}_{classification_type}_imbalance.json",
    )
    strategies = select_strategies(imbalance_report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_balanced_class_weights(output.y_train, device=device)
    sampler = make_weighted_sampler(output.y_train) if use_weighted_sampler else None
    train_loader = make_loader(
        output.X_train,
        output.y_train,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        sampler=sampler,
    )
    val_loader = make_loader(output.X_val, output.y_val, batch_size=batch_size)
    test_loader = make_loader(output.X_test, output.y_test, batch_size=batch_size)

    model = build_dl_model(model_name, n_features=output.X_train.shape[1], n_classes=len(output.label_mapping))
    criterion = FocalLoss(gamma=focal_gamma, weight=class_weights) if use_focal_loss else None
    trainer = TorchTrainer(
        model,
        device=device,
        lr=lr,
        class_weights=None if use_focal_loss else class_weights,
        criterion=criterion,
    )

    ckpt_path = artifact_dir / f"{experiment_name}.pt"
    start = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_path=ckpt_path, patience=patience)
    y_true, y_pred = trainer.predict(test_loader)
    _, y_probs = trainer.predict_proba(test_loader)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_probs=y_probs,
        benign_label=benign_label_index(output.label_mapping),
    )
    metrics["train_seconds"] = time.time() - start
    
    # Generate plots
    plots_dir = root / "results" / "plots" / f"{dataset_key}_{classification_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        from src.evaluation import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, plots_dir / f"{experiment_name}_confusion_matrix.png")
        plot_roc_curve(y_true, y_probs, class_names, plots_dir / f"{experiment_name}_roc_curve.png")
        plot_pr_curve(y_true, y_probs, class_names, plots_dir / f"{experiment_name}_pr_curve.png")
        print(f"[+] Generated plots for {experiment_name}.")
    except Exception as e:
        print(f"[-] Warning: Failed to generate plots for {experiment_name}: {e}")
    metrics["history"] = history
    metrics["checkpoint"] = str(ckpt_path.relative_to(root))

    report = {
        "dataset": dataset_key,
        "classification": classification_type,
        "experiment": experiment_name,
        "device": str(device),
        "model_name": model_name,
        "use_focal_loss": use_focal_loss,
        "use_weighted_sampler": use_weighted_sampler,
        "focal_gamma": focal_gamma if use_focal_loss else None,
        "sample_fraction": sample_fraction,
        "label_mapping": output.label_mapping,
        "imbalance_strategies": strategies,
        "models": {
            experiment_name: metrics,
        },
    }
    result_path = result_dir / f"{dataset_key}_{classification_type}_{experiment_name}_results.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"feature_names": output.feature_names}, artifact_dir / f"{experiment_name}_metadata.pkl")
    return report


def run_dl_experiment_from_output(
    output,
    dataset_key: str,
    classification_type: str,
    root: Path,
    experiment_name: str,
    model_name: str = "mlp",
    batch_size: int = 512,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 5,
    use_focal_loss: bool = False,
    use_weighted_sampler: bool = False,
    focal_gamma: float = 2.0,
    sample_fraction: float = 1.0,
) -> dict:
    """Run an DL experiment from an already-preprocessed AdapterOutput.

    This is intended for Colab notebooks where CICIDS2017 has already been
    loaded and split. It avoids spawning a subprocess that loads the full CSVs a
    second time, which can exceed System RAM on the free Colab runtime.
    """
    experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    set_seed(seed)
    output = _sample_output(output, sample_fraction, seed)

    artifact_dir = root / "artifacts" / dataset_key / classification_type
    save_artifacts(output, artifact_dir)

    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    class_names = [None] * len(output.label_mapping)
    for name, idx in output.label_mapping.items():
        class_names[idx] = name
    imbalance_report = analyze_class_distribution(
        output.y_train,
        class_names=class_names,
        output_path=result_dir / f"{dataset_key}_{classification_type}_imbalance.json",
    )
    strategies = select_strategies(imbalance_report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_balanced_class_weights(output.y_train, device=device)
    sampler = make_weighted_sampler(output.y_train) if use_weighted_sampler else None
    train_loader = make_loader(
        output.X_train,
        output.y_train,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        sampler=sampler,
    )
    val_loader = make_loader(output.X_val, output.y_val, batch_size=batch_size)
    test_loader = make_loader(output.X_test, output.y_test, batch_size=batch_size)

    model = build_dl_model(model_name, n_features=output.X_train.shape[1], n_classes=len(output.label_mapping))
    criterion = FocalLoss(gamma=focal_gamma, weight=class_weights) if use_focal_loss else None
    trainer = TorchTrainer(
        model,
        device=device,
        lr=lr,
        class_weights=None if use_focal_loss else class_weights,
        criterion=criterion,
    )

    ckpt_path = artifact_dir / f"{experiment_name}.pt"
    start = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_path=ckpt_path, patience=patience)
    y_true, y_pred = trainer.predict(test_loader)
    _, y_probs = trainer.predict_proba(test_loader)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_probs=y_probs,
        benign_label=benign_label_index(output.label_mapping),
    )
    metrics["train_seconds"] = time.time() - start
    
    # Generate plots
    plots_dir = root / "results" / "plots" / f"{dataset_key}_{classification_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        from src.evaluation import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, plots_dir / f"{experiment_name}_confusion_matrix.png")
        plot_roc_curve(y_true, y_probs, class_names, plots_dir / f"{experiment_name}_roc_curve.png")
        plot_pr_curve(y_true, y_probs, class_names, plots_dir / f"{experiment_name}_pr_curve.png")
        print(f"[+] Generated plots for {experiment_name}.")
    except Exception as e:
        print(f"[-] Warning: Failed to generate plots for {experiment_name}: {e}")
    metrics["history"] = history
    metrics["checkpoint"] = str(ckpt_path.relative_to(root))

    report = {
        "dataset": dataset_key,
        "classification": classification_type,
        "experiment": experiment_name,
        "device": str(device),
        "model_name": model_name,
        "use_focal_loss": use_focal_loss,
        "use_weighted_sampler": use_weighted_sampler,
        "focal_gamma": focal_gamma if use_focal_loss else None,
        "sample_fraction": sample_fraction,
        "label_mapping": output.label_mapping,
        "imbalance_strategies": strategies,
        "models": {
            experiment_name: metrics,
        },
    }
    result_path = result_dir / f"{dataset_key}_{classification_type}_{experiment_name}_results.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"feature_names": output.feature_names}, artifact_dir / f"{experiment_name}_metadata.pkl")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe PyTorch DL experiments.")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], required=True)
    parser.add_argument("--classification", choices=["binary", "multi"], default="multi")
    parser.add_argument("--experiment", default="MLP_FocalSampler")
    parser.add_argument("--model", choices=["mlp", "cnn1d", "bilstm", "hybrid"], default="mlp")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--cicids-raw-dir", type=Path)
    parser.add_argument("--cicids-cache-path", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    report = run_dl_experiment(
        dataset_key=args.dataset,
        classification_type=args.classification,
        root=args.root,
        experiment_name=args.experiment,
        model_name=args.model,
        use_focal_loss=args.focal_loss,
        use_weighted_sampler=args.weighted_sampler,
        focal_gamma=args.focal_gamma,
        sample_fraction=args.sample_fraction,
        cicids_raw_dir=args.cicids_raw_dir,
        cicids_cache_path=args.cicids_cache_path,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        patience_override=args.patience,
    )
    metrics = report["models"][args.experiment]
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "experiment": report["experiment"],
                "model": report.get("model_name", args.model),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "far": metrics["far"],
                "checkpoint": metrics["checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
