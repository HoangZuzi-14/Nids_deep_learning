from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import joblib
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.adapters import CICIDS2017Adapter, NSLKDDAdapter, UNSWNB15Adapter
from src.evaluation import compute_classification_metrics
from src.pipeline.baseline_runner import benign_label_index
from src.utils.seed import set_seed


ADAPTERS = {
    "nsl_kdd": NSLKDDAdapter,
    "cicids2017": CICIDS2017Adapter,
    "unsw_nb15": UNSWNB15Adapter,
}

DEFAULT_MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "LightGBM")


def _path_or_none(value: str | None) -> Path | None:
    return Path(value) if value else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_artifact_zip(zip_path: Path, root: Path) -> dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"Artifact zip not found: {zip_path}")

    manifest = {}
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
        manifest_members = [
            member
            for member in archive.namelist()
            if member.endswith("_ml_train_only_manifest.json")
        ]
        if manifest_members:
            manifest = json.loads(archive.read(manifest_members[0]).decode("utf-8"))
    return manifest


def _load_manifest(root: Path, dataset: str, classification: str) -> dict:
    manifest_path = root / "results" / f"{dataset}_{classification}_ml_train_only_manifest.json"
    return _read_json(manifest_path) if manifest_path.exists() else {}


def _build_adapter(
    dataset: str,
    config: dict,
    seed: int,
    raw_dir: Path | None,
    cache_path: Path | None,
):
    dataset_config = config["datasets"][dataset]
    adapter_cls = ADAPTERS[dataset]

    kwargs = {
        "cache_path": cache_path,
        "remote_url": dataset_config.get("remote_url"),
        "seed": seed,
    }
    if dataset == "cicids2017":
        kwargs["raw_dir"] = raw_dir
    return adapter_cls(**kwargs)


def evaluate_artifacts(
    dataset: str,
    classification: str,
    root: Path,
    artifact_dir: Path,
    artifact_zip: Path | None,
    raw_dir: Path | None,
    cache_path: Path | None,
    output_path: Path | None,
) -> dict:
    root = root.resolve()
    if artifact_zip:
        manifest = _extract_artifact_zip(artifact_zip, root)
    else:
        manifest = _load_manifest(root, dataset, classification)

    config = yaml.safe_load((root / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    experiments_config = yaml.safe_load(
        (root / "config" / "experiments.yaml").read_text(encoding="utf-8")
    )
    seed = int(manifest.get("seed") or experiments_config.get("seed", 42))
    set_seed(seed)

    split_cfg = experiments_config.get("split", {})
    adapter = _build_adapter(dataset, config, seed, raw_dir, cache_path)
    output = adapter.preprocess(
        classification_type=classification,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        scaler_type=experiments_config.get("preprocessing", {}).get("scaler", "standard"),
    )

    artifact_dir = artifact_dir.resolve()
    manifest_models = manifest.get("models", {}) if isinstance(manifest, dict) else {}
    model_names = list(manifest_models) if manifest_models else list(DEFAULT_MODELS)

    report = {
        "dataset": dataset,
        "classification_type": classification,
        "evaluation_mode": "artifact_predict_only",
        "seed": seed,
        "n_features": len(output.feature_names),
        "label_mapping": output.label_mapping,
        "split_shapes": {
            "train": list(output.X_train.shape),
            "val": list(output.X_val.shape),
            "test": list(output.X_test.shape),
        },
        "models": {},
    }

    benign = benign_label_index(output.label_mapping)
    for model_name in model_names:
        artifact_path = artifact_dir / f"{model_name}.pkl"
        if not artifact_path.exists():
            print(f"[-] Missing artifact, skipped: {artifact_path}")
            continue

        print(f"[*] Evaluating artifact: {model_name}")
        model = joblib.load(artifact_path)
        start = time.perf_counter()
        pred = model.predict(output.X_test)
        probs = model.predict_proba(output.X_test) if hasattr(model, "predict_proba") else None
        eval_seconds = time.perf_counter() - start

        metrics = compute_classification_metrics(
            output.y_test,
            pred,
            y_probs=probs,
            benign_label=benign,
        )
        metrics["train_seconds"] = manifest_models.get(model_name, {}).get("train_seconds")
        metrics["eval_seconds"] = eval_seconds
        metrics["artifact"] = str(artifact_path.relative_to(root))
        report["models"][model_name] = metrics
        print(
            f"[+] {model_name}: macro_f1={metrics['macro_f1']:.4f} "
            f"far={metrics['far']:.4f} eval_seconds={eval_seconds:.2f}"
        )

    result_path = output_path or root / "results" / f"{dataset}_{classification}_ml_baselines.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] Saved evaluation report: {result_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved ML artifacts without retraining.")
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--classification", choices=["binary", "multi"], default="multi")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--artifact-zip", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or args.root / "artifacts" / args.dataset / args.classification
    evaluate_artifacts(
        dataset=args.dataset,
        classification=args.classification,
        root=args.root,
        artifact_dir=artifact_dir,
        artifact_zip=args.artifact_zip,
        raw_dir=args.raw_dir,
        cache_path=args.cache_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
