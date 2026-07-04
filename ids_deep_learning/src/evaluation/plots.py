import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, class_names, output_path):
    """Draw a beautiful heatmap of the confusion matrix and save it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        square=True,
        cbar=True
    )
    plt.title("Confusion Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_probs, class_names, output_path):
    """Draw multi-class OVR ROC curves and save them."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:
        # Binary case
        p = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    else:
        # Multi-class OVR
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            # Check if there are active positive samples for this class in y_true
            if np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")
                
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=12, labelpad=10)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_pr_curve(y_true, y_probs, class_names, output_path):
    """Draw multi-class Precision-Recall curves and save them."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:
        # Binary case
        p = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
        precision, recall, _ = precision_recall_curve(y_true, p)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    else:
        # Multi-class
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AUC = {pr_auc:.4f})")
                
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12, labelpad=10)
    plt.ylabel("Precision", fontsize=12, labelpad=10)
    plt.title("Precision-Recall (PR) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
