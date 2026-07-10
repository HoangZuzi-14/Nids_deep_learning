from __future__ import annotations

import base64
import csv
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = REPORTS_DIR / "notebooks"


FINAL_FIGURES = [
    "macro_f1_by_model.png",
    "weighted_f1_by_model.png",
    "far_by_model.png",
    "macro_f1_heatmap.png",
    "macro_f1_vs_far.png",
    "confusion_matrix_nsl_kdd_ml_LightGBM.png",
    "confusion_matrix_unsw_nb15_ml_LightGBM.png",
    "confusion_matrix_cicids2017_ml_RandomForest.png",
    "confusion_matrix_cicids2017_dl_BiLSTMAttention_Modular.png",
]

FINAL_TABLES = [
    "final_model_evaluation.csv",
    "final_model_ranking.csv",
    "final_ml_vs_dl_comparison.csv",
    "final_per_class_metrics.csv",
]

HYBRID_FIGURES = [
    ("results/plots/hybrid_recovery_percentage.png", "hybrid_recovery_percentage.png"),
    ("results/plots/nsl_kdd_hybrid_risk_distribution.png", "nsl_kdd_hybrid_risk_distribution.png"),
    ("results/plots/unsw_nb15_hybrid_risk_distribution.png", "unsw_nb15_hybrid_risk_distribution.png"),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def copy_if_exists(src: Path, dst: Path, missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src.relative_to(ROOT)))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_submission_assets() -> list[str]:
    missing: list[str] = []
    final_src = RESULTS_DIR / "final_evaluation"
    final_dst = FIGURES_DIR / "final"
    for name in FINAL_FIGURES:
        copy_if_exists(final_src / name, final_dst / name, missing)

    for name in FINAL_TABLES:
        copy_if_exists(final_src / name, TABLES_DIR / name, missing)

    hybrid_dst = FIGURES_DIR / "hybrid"
    for src_name, dst_name in HYBRID_FIGURES:
        copy_if_exists(ROOT / src_name, hybrid_dst / dst_name, missing)

    for name in [
        "nsl_kdd_hybrid_decision_results.json",
        "unsw_nb15_hybrid_decision_results.json",
        "cicids2017_multi_Hybrid_RF_IF_inline_10pct_results.json",
    ]:
        copy_if_exists(RESULTS_DIR / name, TABLES_DIR / name, missing)

    return missing


def build_key_results() -> list[dict[str, object]]:
    rows = read_csv_rows(TABLES_DIR / "final_model_evaluation.csv")
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    best: dict[str, dict[str, str]] = {}
    for row in evaluated:
        dataset = row["dataset"]
        if dataset not in best or as_float(row.get("macro_f1")) > as_float(best[dataset].get("macro_f1")):
            best[dataset] = row

    key_rows: list[dict[str, object]] = []
    for dataset in ["nsl_kdd", "unsw_nb15", "cicids2017"]:
        row = best.get(dataset)
        if not row:
            continue
        key_rows.append(
            {
                "dataset": row.get("dataset_label", dataset),
                "family": row.get("family", ""),
                "best_model": row.get("model", ""),
                "accuracy": fmt(row.get("accuracy")),
                "macro_f1": fmt(row.get("macro_f1")),
                "weighted_f1": fmt(row.get("weighted_f1")),
                "far": fmt(row.get("far")),
                "train_seconds": fmt(row.get("train_seconds"), 2),
            }
        )

    write_csv_rows(
        TABLES_DIR / "key_results.csv",
        key_rows,
        ["dataset", "family", "best_model", "accuracy", "macro_f1", "weighted_f1", "far", "train_seconds"],
    )
    return key_rows


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_hybrid_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for dataset_key, label in [("nsl_kdd", "NSL-KDD"), ("unsw_nb15", "UNSW-NB15")]:
        data = load_json(TABLES_DIR / f"{dataset_key}_hybrid_decision_results.json")
        if not data:
            continue
        clf = data.get("classifier_alone", {})
        rows.append(
            {
                "dataset": label,
                "setting": "Classifier alone",
                "accuracy": fmt(clf.get("accuracy")),
                "far": fmt(clf.get("far")),
                "attack_recall": fmt(clf.get("attack_recall")),
                "f1": fmt(clf.get("f1")),
                "suspicious_unknown_alerts": "",
                "missed_attacks": "",
                "recovered_attacks": "",
                "recovery_rate": "",
                "notes": "Supervised RandomForest baseline",
            }
        )
        for key in ["far_0.01", "far_0.03", "far_0.05", "far_0.1"]:
            metrics = data.get("hybrid_evaluations", {}).get(key)
            if not metrics:
                continue
            rows.append(
                {
                    "dataset": label,
                    "setting": f"Hybrid Decision ({key})",
                    "accuracy": fmt(metrics.get("accuracy")),
                    "far": fmt(metrics.get("far")),
                    "attack_recall": fmt(metrics.get("attack_recall")),
                    "f1": fmt(metrics.get("f1")),
                    "suspicious_unknown_alerts": metrics.get("suspicious_unknown_alerts", ""),
                    "missed_attacks": metrics.get("classifier_missed_attacks", ""),
                    "recovered_attacks": metrics.get("recovered_attacks_by_ae", ""),
                    "recovery_rate": fmt(metrics.get("attack_recovery_rate")),
                    "notes": "Classifier plus Autoencoder risk threshold",
                }
            )

    cicids = load_json(TABLES_DIR / "cicids2017_multi_Hybrid_RF_IF_inline_10pct_results.json")
    if cicids:
        rf = cicids.get("rf_multiclass", {})
        hybrid = cicids.get("hybrid_binary", {})
        rows.append(
            {
                "dataset": "CICIDS2017",
                "setting": "RandomForest multiclass baseline",
                "accuracy": fmt(rf.get("accuracy")),
                "far": fmt(rf.get("far")),
                "attack_recall": "",
                "f1": fmt(rf.get("macro_f1")),
                "suspicious_unknown_alerts": "",
                "missed_attacks": "",
                "recovered_attacks": "",
                "recovery_rate": "",
                "notes": "10 percent inline sample",
            }
        )
        rows.append(
            {
                "dataset": "CICIDS2017",
                "setting": "Hybrid RF + IsolationForest (far_0.03)",
                "accuracy": fmt(hybrid.get("accuracy")),
                "far": fmt(hybrid.get("far")),
                "attack_recall": fmt(hybrid.get("attack_recall")),
                "f1": fmt(hybrid.get("f1")),
                "suspicious_unknown_alerts": hybrid.get("suspicious_unknown_alerts", ""),
                "missed_attacks": hybrid.get("rf_missed_attacks", ""),
                "recovered_attacks": hybrid.get("rf_missed_attacks_caught_by_if", ""),
                "recovery_rate": fmt(hybrid.get("rf_missed_attack_recovery_rate")),
                "notes": "Good binary recall, but no missed-RF attack recovery in this sample",
            }
        )

    fieldnames = [
        "dataset",
        "setting",
        "accuracy",
        "far",
        "attack_recall",
        "f1",
        "suspicious_unknown_alerts",
        "missed_attacks",
        "recovered_attacks",
        "recovery_rate",
        "notes",
    ]
    write_csv_rows(TABLES_DIR / "hybrid_decision_summary.csv", rows, fieldnames)
    return rows


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")
    return (
        "<table>"
        "<thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def selected_hybrid_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keep = []
    for row in rows:
        setting = str(row.get("setting", ""))
        if setting == "Classifier alone" or "far_0.03" in setting or row.get("dataset") == "CICIDS2017":
            keep.append(row)
    return keep


def formatted_comparison_rows() -> list[dict[str, object]]:
    rows = read_csv_rows(TABLES_DIR / "final_ml_vs_dl_comparison.csv")
    formatted: list[dict[str, object]] = []
    for row in rows:
        formatted.append(
            {
                "dataset_label": row.get("dataset_label", ""),
                "best_ml_model": row.get("best_ml_model", ""),
                "best_ml_macro_f1": fmt(row.get("best_ml_macro_f1")),
                "best_dl_model": row.get("best_dl_model", ""),
                "best_dl_macro_f1": fmt(row.get("best_dl_macro_f1")),
                "macro_f1_gap_dl_minus_ml": fmt(row.get("macro_f1_gap_dl_minus_ml")),
            }
        )
    return formatted


def write_final_summary(key_rows: list[dict[str, object]], hybrid_rows: list[dict[str, object]]) -> None:
    comparison_rows = formatted_comparison_rows()
    hybrid_focus = selected_hybrid_rows(hybrid_rows)

    lines = [
        "# NIDS Deep Learning - Submission Summary",
        "",
        "This folder contains curated, submission-ready artifacts generated from local experiment outputs.",
        "Raw datasets, trained checkpoints, cache files, and long training logs are intentionally kept out of Git.",
        "",
        "## Best model by dataset",
        "",
        markdown_table(
            key_rows,
            ["dataset", "family", "best_model", "accuracy", "macro_f1", "weighted_f1", "far", "train_seconds"],
        ),
        "",
        "## Best ML vs best DL",
        "",
        markdown_table(
            comparison_rows,
            [
                "dataset_label",
                "best_ml_model",
                "best_ml_macro_f1",
                "best_dl_model",
                "best_dl_macro_f1",
                "macro_f1_gap_dl_minus_ml",
            ],
        ),
        "",
        "## Hybrid Decision Layer",
        "",
        "The Hybrid Decision Layer is separate from the CNN-LSTM Hybrid model. It combines a supervised classifier with an anomaly detector to recover attacks that the classifier predicts as benign.",
        "",
        markdown_table(
            hybrid_focus,
            [
                "dataset",
                "setting",
                "accuracy",
                "far",
                "attack_recall",
                "f1",
                "missed_attacks",
                "recovered_attacks",
                "recovery_rate",
            ],
        ),
        "",
        "## Included figures",
        "",
        "- `figures/final/`: macro-F1, weighted-F1, FAR, heatmap, trade-off plots, and selected confusion matrices.",
        "- `figures/hybrid/`: hybrid recovery percentage and risk-score distributions.",
        "",
        "## Included tables",
        "",
        "- `tables/final_model_evaluation.csv`: full ML/DL evaluation table.",
        "- `tables/final_model_ranking.csv`: ranking across datasets and model families.",
        "- `tables/final_ml_vs_dl_comparison.csv`: best ML vs best DL comparison.",
        "- `tables/final_per_class_metrics.csv`: per-class metrics for detailed discussion.",
        "- `tables/hybrid_decision_summary.csv`: classifier vs hybrid decision trade-offs.",
        "",
        "## Notebook",
        "",
        "Open `notebooks/final_project_report.ipynb` to view the report with preserved summary tables and linked figures.",
        "",
    ]
    (REPORTS_DIR / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_output_cell(source: str, rows: list[dict[str, object]], columns: list[str], execution_count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/html": html_table(rows, columns),
                    "text/plain": markdown_table(rows, columns),
                },
                "metadata": {},
                "output_type": "display_data",
            }
        ],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def write_report_notebook(key_rows: list[dict[str, object]], hybrid_rows: list[dict[str, object]]) -> None:
    comparison_rows = formatted_comparison_rows()
    hybrid_focus = selected_hybrid_rows(hybrid_rows)

    cells = [
        markdown_cell(
            "# Final Project Report - Network Intrusion Detection\n\n"
            "This notebook is the compact submission view of the project. It keeps the important tables and links to curated figures under `reports/`, while the raw datasets and model checkpoints remain local."
        ),
        markdown_cell(
            "## Experiment Scope\n\n"
            "- Datasets: NSL-KDD, UNSW-NB15, CICIDS2017.\n"
            "- ML baselines: Logistic Regression, Random Forest, XGBoost, LightGBM.\n"
            "- DL models: MLP, CNN1D, BiLSTM Attention, CNN-LSTM Hybrid.\n"
            "- Hybrid decision experiments: classifier plus anomaly detector for missed-attack recovery."
        ),
        make_output_cell(
            "# Stored output generated from ../tables/key_results.csv\nimport pandas as pd\npd.read_csv('../tables/key_results.csv')",
            key_rows,
            ["dataset", "family", "best_model", "accuracy", "macro_f1", "weighted_f1", "far", "train_seconds"],
            1,
        ),
        markdown_cell(
            "## Final Evaluation Figures\n\n"
            "![Macro F1 by model](../figures/final/macro_f1_by_model.png)\n\n"
            "![Weighted F1 by model](../figures/final/weighted_f1_by_model.png)\n\n"
            "![FAR by model](../figures/final/far_by_model.png)\n\n"
            "![Macro F1 heatmap](../figures/final/macro_f1_heatmap.png)\n\n"
            "![Macro F1 vs FAR](../figures/final/macro_f1_vs_far.png)"
        ),
        make_output_cell(
            "# Stored output generated from ../tables/final_ml_vs_dl_comparison.csv\nimport pandas as pd\npd.read_csv('../tables/final_ml_vs_dl_comparison.csv')",
            comparison_rows,
            [
                "dataset_label",
                "best_ml_model",
                "best_ml_macro_f1",
                "best_dl_model",
                "best_dl_macro_f1",
                "macro_f1_gap_dl_minus_ml",
            ],
            2,
        ),
        markdown_cell(
            "## Selected Confusion Matrices\n\n"
            "![NSL-KDD LightGBM confusion matrix](../figures/final/confusion_matrix_nsl_kdd_ml_LightGBM.png)\n\n"
            "![UNSW-NB15 LightGBM confusion matrix](../figures/final/confusion_matrix_unsw_nb15_ml_LightGBM.png)\n\n"
            "![CICIDS2017 Random Forest confusion matrix](../figures/final/confusion_matrix_cicids2017_ml_RandomForest.png)\n\n"
            "![CICIDS2017 BiLSTM Attention confusion matrix](../figures/final/confusion_matrix_cicids2017_dl_BiLSTMAttention_Modular.png)"
        ),
        markdown_cell(
            "## Hybrid Decision Layer\n\n"
            "The project uses two different meanings of hybrid. `CNN-LSTM Hybrid` is a deep-learning architecture. `Hybrid Decision Layer` is the research-oriented experiment that combines a classifier with an anomaly detector to flag suspicious unknown traffic."
        ),
        make_output_cell(
            "# Stored output generated from ../tables/hybrid_decision_summary.csv\nimport pandas as pd\npd.read_csv('../tables/hybrid_decision_summary.csv')",
            hybrid_focus,
            [
                "dataset",
                "setting",
                "accuracy",
                "far",
                "attack_recall",
                "f1",
                "missed_attacks",
                "recovered_attacks",
                "recovery_rate",
            ],
            3,
        ),
        markdown_cell(
            "## Hybrid Figures\n\n"
            "![Hybrid recovery percentage](../figures/hybrid/hybrid_recovery_percentage.png)\n\n"
            "![NSL-KDD hybrid risk distribution](../figures/hybrid/nsl_kdd_hybrid_risk_distribution.png)\n\n"
            "![UNSW-NB15 hybrid risk distribution](../figures/hybrid/unsw_nb15_hybrid_risk_distribution.png)"
        ),
        markdown_cell(
            "## Key Takeaways\n\n"
            "- Traditional ML baselines are strongest on the final macro-F1 comparison, especially LightGBM on NSL-KDD and UNSW-NB15, and Random Forest on CICIDS2017.\n"
            "- DL models still provide useful deep baselines, especially BiLSTM Attention on CICIDS2017.\n"
            "- The Hybrid Decision Layer is valuable as an additional analysis: it recovers missed attacks on NSL-KDD and UNSW-NB15 at controlled FAR targets, while the CICIDS2017 10 percent sample shows limited recovery because Random Forest already misses very few attacks."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS_DIR / "final_project_report.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_manifest(missing: list[str]) -> None:
    files = []
    for path in sorted(REPORTS_DIR.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(REPORTS_DIR)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "source": "Generated from local results/ directory.",
        "missing_sources": missing,
        "files": files,
    }
    (REPORTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Missing results directory: {RESULTS_DIR}")

    for directory in [REPORTS_DIR, TABLES_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    missing = copy_submission_assets()
    key_rows = build_key_results()
    hybrid_rows = build_hybrid_summary()
    write_final_summary(key_rows, hybrid_rows)
    write_report_notebook(key_rows, hybrid_rows)
    write_manifest(missing)

    print(f"Generated submission reports in {REPORTS_DIR}")
    if missing:
        print("Missing optional source files:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
