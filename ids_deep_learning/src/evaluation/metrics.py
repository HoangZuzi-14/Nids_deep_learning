import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def false_alarm_rate(y_true, y_pred, benign_label=0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    benign_mask = y_true == benign_label
    fp = np.sum(benign_mask & (y_pred != benign_label))
    tn = np.sum(benign_mask & (y_pred == benign_label))
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def compute_classification_metrics(y_true, y_pred, y_probs=None, benign_label=0):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "far": false_alarm_rate(y_true, y_pred, benign_label=benign_label),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "roc_auc": 0.0,
        "pr_auc": 0.0,
    }

    if y_probs is not None:
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        unique_classes = np.unique(y_true)
        n_classes = y_probs.shape[1]

        # Calculate ROC-AUC
        try:
            if n_classes == 2:
                # Binary classification
                p = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
                metrics["roc_auc"] = float(roc_auc_score(y_true, p))
            else:
                # Multi-class classification (OVR, macro average)
                if len(unique_classes) < 2:
                    metrics["roc_auc"] = 0.0
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true,
                            y_probs,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(n_classes)),
                        )
                    )
        except Exception:
            metrics["roc_auc"] = 0.0

        # Calculate PR-AUC (Average Precision score)
        try:
            from sklearn.metrics import average_precision_score
            from sklearn.preprocessing import label_binarize

            if n_classes == 2:
                p = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
                metrics["pr_auc"] = float(average_precision_score(y_true, p))
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                metrics["pr_auc"] = float(
                    average_precision_score(y_true_bin, y_probs, average="macro")
                )
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics

