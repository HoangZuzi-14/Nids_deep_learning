from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import yaml

from src.adapters import CICIDS2017Adapter, NSLKDDAdapter, UNSWNB15Adapter
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)
from src.models.ml_baselines import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    build_lightgbm,
)
from src.utils.seed import set_seed


ADAPTERS = {
    "nsl_kdd": NSLKDDAdapter,
    "cicids2017": CICIDS2017Adapter,
    "unsw_nb15": UNSWNB15Adapter,
}


def _repo_path(path: str | None, root: Path):
    if not path:
        return None
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else root / path_obj


def build_adapter(dataset_key: str, datasets_config: dict, root: Path, seed: int):
    cfg = datasets_config["datasets"][dataset_key]
    adapter_cls = ADAPTERS[dataset_key]
    kwargs = {
        "cache_path": _repo_path(cfg.get("cache_path"), root),
        "remote_url": cfg.get("remote_url"),
        "seed": seed,
    }
    if dataset_key == "cicids2017":
        kwargs["raw_dir"] = _repo_path(cfg.get("raw_dir"), root)
    return adapter_cls(**kwargs)


def save_artifacts(output, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(output.scaler, artifact_dir / "scaler.pkl")
    joblib.dump(output.encoders, artifact_dir / "encoders.pkl")
    if getattr(output, "imputer", None) is not None:
        joblib.dump(output.imputer, artifact_dir / "imputer.pkl")
    (artifact_dir / "label_mapping.json").write_text(json.dumps(output.label_mapping, indent=2), encoding="utf-8")
    config = {
        "feature_names": output.feature_names,
        "label_mapping": "label_mapping.json",
        "scaler": "scaler.pkl",
        "encoders": "encoders.pkl",
        "imputer": "imputer.pkl",
    }
    (artifact_dir / "inference_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def benign_label_index(label_mapping: dict[str, int]) -> int:
    for name, idx in label_mapping.items():
        if str(name).lower() in {"benign", "normal"}:
            return int(idx)
    return 0


def run_baselines(dataset_key: str, classification_type: str, root: Path) -> dict:
    datasets_config = yaml.safe_load((root / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    experiments_config = yaml.safe_load((root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    set_seed(seed)

    split_cfg = experiments_config.get("split", {})
    adapter = build_adapter(dataset_key, datasets_config, root, seed)
    output = adapter.preprocess(
        classification_type=classification_type,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        scaler_type=experiments_config.get("preprocessing", {}).get("scaler", "standard"),
    )

    artifact_dir = root / "artifacts" / dataset_key / classification_type
    save_artifacts(output, artifact_dir)

    models = {
        "LogisticRegression": build_logistic_regression(seed),
        "RandomForest": build_random_forest(seed),
        "XGBoost": build_xgboost(seed),
        "LightGBM": build_lightgbm(seed),
    }
    report = {
        "dataset": dataset_key,
        "classification_type": classification_type,
        "n_features": len(output.feature_names),
        "label_mapping": output.label_mapping,
        "models": {},
    }
    
    class_names = [None] * len(output.label_mapping)
    for name_str, idx in output.label_mapping.items():
        class_names[idx] = name_str

    plots_dir = root / "results" / "plots" / f"{dataset_key}_{classification_type}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"[*] Training baseline model: {name} on {dataset_key}...")
        start = time.perf_counter()
        model.fit(output.X_train, output.y_train)
        train_seconds = time.perf_counter() - start
        pred = model.predict(output.X_test)
        probs = model.predict_proba(output.X_test)
        
        metrics = compute_classification_metrics(
            output.y_test, 
            pred, 
            y_probs=probs,
            benign_label=benign_label_index(output.label_mapping)
        )
        metrics["train_seconds"] = train_seconds
        report["models"][name] = metrics
        joblib.dump(model, artifact_dir / f"{name}.pkl")

        # Generate plots
        try:
            plot_confusion_matrix(metrics["confusion_matrix"], class_names, plots_dir / f"{name}_confusion_matrix.png")
            plot_roc_curve(output.y_test, probs, class_names, plots_dir / f"{name}_roc_curve.png")
            plot_pr_curve(output.y_test, probs, class_names, plots_dir / f"{name}_pr_curve.png")
            print(f"[+] Generated plots for {name}.")
        except Exception as e:
            print(f"[-] Warning: Failed to generate plots for {name}: {e}")

    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{dataset_key}_{classification_type}_ml_baselines.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe ML baselines for a configured dataset.")
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--classification", choices=["binary", "multi"], default="multi")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = run_baselines(args.dataset, args.classification, args.root)
    print(json.dumps({"dataset": report["dataset"], "models": list(report["models"])}, indent=2))


if __name__ == "__main__":
    main()
