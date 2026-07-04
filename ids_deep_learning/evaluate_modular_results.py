from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULT_FILES = [
    RESULTS_DIR / "nsl_kdd_multi_modular_results.json",
    RESULTS_DIR / "unsw_nb15_multi_modular_results.json",
    RESULTS_DIR / "cicids2017_multi_modular_results.json",
]


def _round(value, digits: int = 6):
    return round(float(value), digits)


def _inverse_mapping(label_mapping: dict[str, int]) -> dict[int, str]:
    return {int(idx): label for label, idx in label_mapping.items()}


def _per_class_rows(dataset: str, model_name: str, result: dict) -> list[dict]:
    labels = _inverse_mapping(result["label_mapping"])
    report = result["models"][model_name]["classification_report"]
    rows = []
    for idx, label in sorted(labels.items()):
        metrics = report.get(
            str(idx),
            {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "label": label,
                "support": int(metrics.get("support", 0)),
                "precision": _round(metrics.get("precision", 0.0)),
                "recall": _round(metrics.get("recall", 0.0)),
                "f1": _round(metrics.get("f1-score", 0.0)),
            }
        )
    return rows


def _minority_summary(per_class: list[dict], threshold_ratio: float = 0.05) -> dict:
    total = sum(row["support"] for row in per_class)
    minority = [row for row in per_class if total and row["support"] / total < threshold_ratio]
    if not minority:
        minority = per_class
    return {
        "minority_classes": ", ".join(row["label"] for row in minority),
        "minority_recall_mean": _round(sum(row["recall"] for row in minority) / len(minority)),
        "minority_f1_mean": _round(sum(row["f1"] for row in minority) / len(minority)),
    }


def main() -> None:
    model_rows = []
    per_class_rows = []

    for path in RESULT_FILES:
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        dataset = result["dataset"]
        for model_name, metrics in result["models"].items():
            class_rows = _per_class_rows(dataset, model_name, result)
            minority = _minority_summary(class_rows)
            model_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "accuracy": _round(metrics["accuracy"]),
                    "macro_f1": _round(metrics["macro_f1"]),
                    "weighted_f1": _round(metrics["weighted_f1"]),
                    "roc_auc": _round(metrics.get("roc_auc", 0.0)),
                    "pr_auc": _round(metrics.get("pr_auc", 0.0)),
                    "far": _round(metrics["far"]),
                    "minority_recall_mean": minority["minority_recall_mean"],
                    "minority_f1_mean": minority["minority_f1_mean"],
                    "train_seconds": _round(metrics.get("train_seconds", 0.0), 3),
                    "minority_classes": minority["minority_classes"],
                }
            )
            per_class_rows.extend(class_rows)

    model_rows.sort(key=lambda row: (row["dataset"], -row["macro_f1"], row["far"]))
    per_class_rows.sort(key=lambda row: (row["dataset"], row["model"], row["label"]))

    comparison_path = RESULTS_DIR / "modular_model_evaluation.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    per_class_path = RESULTS_DIR / "modular_per_class_evaluation.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    best_by_dataset = {}
    for row in model_rows:
        best_by_dataset.setdefault(row["dataset"], row)

    lines = [
        "# Modular Evaluation Summary",
        "",
        "Selection rule: prefer higher Macro-F1, then lower FAR, then higher minority recall.",
        "",
        "## Best Model by Dataset",
        "",
        "| Dataset | Best model | Macro-F1 | FAR | Minority recall mean |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, row in sorted(best_by_dataset.items()):
        lines.append(
            f"| {dataset} | {row['model']} | {row['macro_f1']:.6f} | "
            f"{row['far']:.6f} | {row['minority_recall_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## All Model Results",
            "",
            "| Dataset | Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC | FAR | Minority recall mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['weighted_f1']:.6f} | "
            f"{row['roc_auc']:.6f} | {row['pr_auc']:.6f} | "
            f"{row['far']:.6f} | {row['minority_recall_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- RandomForest is currently the strongest corrected CICIDS2017 baseline.",
            "- LogisticRegression tends to raise FAR substantially, especially on CICIDS2017 and UNSW-NB15.",
            "- MLP is useful as the first deep-learning baseline, but it does not yet beat RandomForest on the modular runs.",
            "- ROC-AUC and PR-AUC are now fully integrated and calculated from class probability predictions.",
            "",
        ]
    )
    summary_path = RESULTS_DIR / "modular_evaluation_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", comparison_path)
    print("Saved:", per_class_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
