from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "visualize_report.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def build_notebook() -> dict:
    cells = [
        markdown(
            """
            # NIDS Deep Learning Project Report

            This notebook is the project report layer for the modular NIDS framework.
            It loads experiment artifacts from `results/`, checks for invalid one-class
            runs, and summarizes the supervised, imbalance, anomaly, hybrid, and
            domain-shift findings.

            Recommended final position: use tree-based supervised models such as
            RandomForest/XGBoost for known attacks, then keep anomaly detection as a
            separate suspicious/unknown branch instead of forcing very rare classes
            into ordinary supervised classification.
            """
        ),
        markdown(
            """
            ## 1. Setup

            The report cells below are intentionally data-driven. Re-run the pipeline
            scripts first, then re-run this notebook so tables reflect the current
            `results/` directory.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import pandas as pd

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            RESULTS_DIR = PROJECT_ROOT / "results"
            ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

            def load_json(name):
                path = RESULTS_DIR / name
                return json.loads(path.read_text(encoding="utf-8"))

            def read_text(name):
                return (RESULTS_DIR / name).read_text(encoding="utf-8")

            print("Project root:", PROJECT_ROOT)
            print("Results found:", len(list(RESULTS_DIR.glob("*"))))
            """
        ),
        markdown(
            """
            ## 2. Dataset and Preprocessing

            The project uses adapters for NSL-KDD, UNSW-NB15, and CICIDS2017. The
            corrected preprocessing contract is:

            1. Load and clean raw/cache data.
            2. Map labels.
            3. Reject one-class training data.
            4. Split train/validation/test.
            5. Fit categorical encoders, imputers, and scalers on train only.
            6. Transform validation/test with train-fitted artifacts.

            This order is important because fitting encoders or imputers on the full
            dataset leaks validation/test distribution into training.
            """
        ),
        code(
            """
            for path in sorted(RESULTS_DIR.glob("*_imbalance.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                print("\\n==", path.name, "==")
                if isinstance(data, dict):
                    print(json.dumps(data, indent=2)[:2000])
            """
        ),
        markdown(
            """
            ## 3. Imbalance Handling

            CICIDS2017 is highly imbalanced. The selected direction is adaptive
            cost-sensitive learning plus rare-class grouping. GAN augmentation is kept
            as legacy/future work because synthetic tabular IDS samples are difficult
            to validate and can increase false alarms.
            """
        ),
        code(
            """
            strategy_path = RESULTS_DIR / "cicids2017_imbalance_strategy.md"
            if strategy_path.exists():
                print(strategy_path.read_text(encoding="utf-8"))
            else:
                print("Missing cicids2017_imbalance_strategy.md")
            """
        ),
        markdown(
            """
            ## 4. Model Comparison: NSL-KDD, UNSW-NB15, CICIDS2017

            The main model selection rule is: prefer higher Macro-F1, then lower FAR,
            then higher minority recall. Accuracy alone is not sufficient for NIDS
            because a majority Benign class can hide poor attack detection.
            """
        ),
        code(
            """
            model_eval_path = RESULTS_DIR / "modular_model_evaluation.csv"
            if model_eval_path.exists():
                model_eval = pd.read_csv(model_eval_path)
                display_cols = [
                    "dataset", "model", "accuracy", "macro_f1", "weighted_f1",
                    "roc_auc", "pr_auc", "far", "minority_recall_mean",
                ]
                display(model_eval[display_cols])
            else:
                print("Missing modular_model_evaluation.csv. Run evaluate_modular_results.py first.")
            """
        ),
        code(
            """
            per_class_path = RESULTS_DIR / "modular_per_class_evaluation.csv"
            if per_class_path.exists():
                per_class = pd.read_csv(per_class_path)
                low_support = per_class.sort_values(["dataset", "support", "model"]).head(30)
                display(low_support)
            else:
                print("Missing modular_per_class_evaluation.csv.")
            """
        ),
        markdown(
            """
            ## 5. NSL-KDD Findings

            NSL-KDD is useful for fast debugging and model comparison. The stronger
            classical ML models generally outperform the deep models in Macro-F1 and
            FAR on the current modular results.
            """
        ),
        code(
            """
            nsl_path = RESULTS_DIR / "nsl_kdd_multi_modular_results.json"
            if nsl_path.exists():
                nsl = load_json("nsl_kdd_multi_modular_results.json")
                rows = []
                for model, metrics in nsl["models"].items():
                    rows.append({
                        "model": model,
                        "accuracy": metrics.get("accuracy"),
                        "macro_f1": metrics.get("macro_f1"),
                        "far": metrics.get("far"),
                        "roc_auc": metrics.get("roc_auc"),
                        "pr_auc": metrics.get("pr_auc"),
                    })
                display(pd.DataFrame(rows).sort_values("macro_f1", ascending=False))
            """
        ),
        markdown(
            """
            ## 6. CICIDS2017 Validity Check

            CICIDS2017 results must be checked carefully. A model result with a 1x1
            confusion matrix or only one active class is invalid for multi-class
            evaluation, even if it reports 100% accuracy.
            """
        ),
        code(
            """
            def invalid_reason(metrics):
                confusion = metrics.get("confusion_matrix", [])
                report = metrics.get("classification_report", {})
                class_rows = [v for k, v in report.items() if str(k).isdigit()]
                active = [row for row in class_rows if float(row.get("support", 0)) > 0]
                if len(confusion) < 2 or len(active) < 2:
                    return "degenerate_one_class"
                return ""

            for path in sorted(RESULTS_DIR.glob("*_modular_results.json")):
                result = json.loads(path.read_text(encoding="utf-8"))
                invalid = []
                for model, metrics in result.get("models", {}).items():
                    reason = invalid_reason(metrics)
                    if reason:
                        invalid.append({"dataset": result.get("dataset"), "model": model, "reason": reason})
                if invalid:
                    print("\\nInvalid results in", path.name)
                    display(pd.DataFrame(invalid))
            """
        ),
        markdown(
            """
            ## 7. Domain Shift

            Cross-dataset tests estimate how much performance changes when training
            and testing distributions differ. Large drops in Macro-F1 and increases
            in FAR support the conclusion that supervised NIDS models do not transfer
            cleanly across network environments without adaptation.
            """
        ),
        code(
            """
            report_path = RESULTS_DIR / "cross_dataset_report.md"
            if report_path.exists():
                print(report_path.read_text(encoding="utf-8"))
            else:
                print("Missing cross_dataset_report.md")
            """
        ),
        markdown(
            """
            ## 8. Anomaly Detection

            The anomaly branch is trained on Benign-only traffic and tuned by target
            FAR. Its role is suspicious/unknown detection, not replacing the known
            attack classifier for labels that are already well learned.
            """
        ),
        code(
            """
            for path in sorted(RESULTS_DIR.glob("*IsolationForest*results.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = []
                for threshold, metrics in data.get("evaluations", {}).items():
                    row = {"threshold": threshold}
                    row.update(metrics)
                    rows.append(row)
                print("\\n==", path.name, "==")
                display(pd.DataFrame(rows))
            """
        ),
        markdown(
            """
            ## 9. Hybrid Decision

            Hybrid results should be interpreted by the trade-off between recovered
            missed attacks and extra false alarms. For CICIDS2017, the tested
            RandomForest OR IsolationForest rule was rejected for known labels because
            it increased FAR without recovering RandomForest's missed attacks.
            """
        ),
        code(
            """
            for path in sorted(RESULTS_DIR.glob("*hybrid*results.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                print("\\n==", path.name, "==")
                keys = [key for key in ["classifier_alone", "rf_multiclass", "if_binary", "hybrid_binary"] if key in data]
                for key in keys:
                    print(key)
                    print(json.dumps(data[key], indent=2)[:2000])
            """
        ),
        markdown(
            """
            ## 10. Final Conclusion

            The strongest current story is not "deep learning wins everywhere." The
            stronger and more defensible conclusion is:

            - Use RandomForest/XGBoost/LightGBM as robust known-attack baselines.
            - Use Macro-F1, minority recall, and FAR as primary IDS metrics.
            - Treat very rare or unseen behavior as an anomaly/suspicious branch.
            - Keep GAN and over-aggressive sampling as future work unless false alarm
              impact is controlled.
            - Re-run CICIDS2017 only with a verified multi-class raw/cache dataset.
            """
        ),
    ]

    return {
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


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
