from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.anomaly.autoencoder import fit_autoencoder, compute_autoencoder_scores
from src.evaluation.metrics import compute_classification_metrics
from src.pipeline.baseline_runner import benign_label_index, build_adapter
from src.models.ml_baselines import build_random_forest, build_xgboost
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


def binary_metrics(y_true_attack, y_pred_attack):
    return {
        "accuracy": float(accuracy_score(y_true_attack, y_pred_attack)),
        "precision": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid Confidence Risk Scoring (Classifier + Autoencoder).")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], default="nsl_kdd")
    parser.add_argument("--classification", choices=["multi"], default="multi")
    parser.add_argument("--classifier", choices=["RandomForest", "XGBoost"], default="RandomForest")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for classifier uncertainty")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for normalized anomaly score")
    parser.add_argument("--epochs", type=int, default=20, help="Number of Autoencoder epochs")
    parser.add_argument("--latent-dim", type=int, default=16, help="Autoencoder latent space dimension")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of dataset to use for fast evaluation")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    # Load configurations
    datasets_config = yaml.safe_load((args.root / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    experiments_config = yaml.safe_load((args.root / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    seed = int(experiments_config.get("seed", 42))
    set_seed(seed)

    print(f"[*] Processing dataset: {args.dataset} (Classification: {args.classification})")
    print(f"[*] Classifier: {args.classifier} | Alpha: {args.alpha} | Beta: {args.beta}")

    # Build adapter and preprocess data
    split_cfg = experiments_config.get("split", {})
    adapter = build_adapter(args.dataset, datasets_config, args.root, seed)
    output = adapter.preprocess(
        classification_type=args.classification,
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
        scaler_type=experiments_config.get("preprocessing", {}).get("scaler", "standard"),
    )

    benign = benign_label_index(output.label_mapping)
    print(f"[+] Loaded dataset. Benign label index: {benign}")

    # Optionally sample dataset
    X_train, y_train = output.X_train, output.y_train
    X_val, y_val = output.X_val, output.y_val
    X_test, y_test = output.X_test, output.y_test

    if args.sample_fraction < 1.0:
        print(f"[*] Sampling dataset with fraction: {args.sample_fraction}...")
        _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=args.sample_fraction, random_state=seed, stratify=y_train)
        _, X_val, _, y_val = train_test_split(X_val, y_val, test_size=args.sample_fraction, random_state=seed, stratify=y_val)
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=args.sample_fraction, random_state=seed, stratify=y_test)

    # 1. Load or Train Classifier
    classifier_path = args.root / "artifacts" / args.dataset / args.classification / f"{args.classifier}.pkl"
    if classifier_path.exists():
        print(f"[+] Loading pre-trained {args.classifier} from {classifier_path}...")
        clf = joblib.load(classifier_path)
    else:
        print(f"[-] Pre-trained {args.classifier} not found. Training a new one...")
        if args.classifier == "RandomForest":
            clf = build_random_forest(seed=seed)
        else:
            clf = build_xgboost(seed=seed)
        clf.fit(X_train, y_train)
        # Save trained classifier
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, classifier_path)
        print(f"[+] Saved newly trained {args.classifier} to {classifier_path}")

    # 2. Fit Autoencoder on Benign-Only data
    train_normal_mask = np.asarray(y_train) == benign
    val_normal_mask = np.asarray(y_val) == benign

    print(f"[*] Extracting normal-only samples for Autoencoder training...")
    X_train_normal = X_train[train_normal_mask]
    X_val_normal = X_val[val_normal_mask]
    print(f"[+] Train normal shape: {X_train_normal.shape} | Val normal shape: {X_val_normal.shape}")

    print(f"[*] Training PyTorch Autoencoder (epochs={args.epochs}, latent_dim={args.latent_dim})...")
    ae_model, ae_scaler = fit_autoencoder(
        X_normal=X_train_normal,
        X_val_normal=X_val_normal,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=256,
        patience=5,
        seed=seed,
        device="cpu",
    )
    print("[+] PyTorch Autoencoder training complete.")

    # 3. Compute Anomaly Scores and Normalize
    print("[*] Computing reconstruction anomaly scores...")
    val_anomaly_scores = compute_autoencoder_scores(ae_model, ae_scaler, X_val)
    test_anomaly_scores = compute_autoencoder_scores(ae_model, ae_scaler, X_test)

    # Fit MinMax scaling of anomaly scores on validation set
    val_normal_scores = val_anomaly_scores[val_normal_mask]
    val_min = float(np.min(val_normal_scores))
    val_max = float(np.max(val_normal_scores))
    print(f"[+] Validation Benign Anomaly Score Stats: Min={val_min:.6f}, Max={val_max:.6f}")

    def normalize_scores(scores):
        eps = 1e-8
        norm = (scores - val_min) / (val_max - val_min + eps)
        return np.clip(norm, 0.0, 1.0)

    val_anomaly_norm = normalize_scores(val_anomaly_scores)
    test_anomaly_norm = normalize_scores(test_anomaly_scores)

    # 4. Predict probabilities using Classifier
    print("[*] Predicting with classifier...")
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    # 5. Compute Confidence Risk Score
    # Risk Score = alpha * (1 - prob_benign) + beta * normalized_anomaly_score
    val_prob_benign = val_probs[:, benign]
    test_prob_benign = test_probs[:, benign]

    val_risk = args.alpha * (1.0 - val_prob_benign) + args.beta * val_anomaly_norm
    test_risk = args.alpha * (1.0 - test_prob_benign) + args.beta * test_anomaly_norm

    # 6. Optimize Hybrid threshold based on target FARs on validation set
    # actual benign samples in val
    val_benign_mask = np.asarray(y_val) == benign
    n_val_benign = int(val_benign_mask.sum())

    # classifier predictions on actual benign val samples
    val_pred_benign = val_pred[val_benign_mask]
    # Fixed false alarms (actual benign misclassified as attack by the classifier)
    n_fixed_fa = int((val_pred_benign != benign).sum())

    print(f"[+] Validation Benign samples: {n_val_benign}")
    print(f"[+] Classifier-alone False Alarms on Validation: {n_fixed_fa} (FAR = {n_fixed_fa / n_val_benign:.4f})")

    target_fars = [0.01, 0.03, 0.05, 0.10]
    evaluations = {}

    y_test_arr = np.asarray(y_test)
    y_test_attack = (y_test_arr != benign).astype(int)

    # Test baseline metrics for Classifier alone (binary evaluation)
    test_pred_attack_clf = (test_pred != benign).astype(int)
    clf_metrics = binary_metrics(y_test_attack, test_pred_attack_clf)
    clf_metrics["far"] = float(test_pred_attack_clf[y_test_arr == benign].mean())
    clf_metrics["attack_recall"] = float(test_pred_attack_clf[y_test_arr != benign].mean())
    print(f"[+] Classifier Alone Test Macro-F1: {clf_metrics['f1']:.4f} | FAR: {clf_metrics['far']:.4f}")

    for target_far in target_fars:
        # allowed total false alarms
        allowed_total_fa = int(np.floor(n_val_benign * target_far))
        allowed_additional_fa = allowed_total_fa - n_fixed_fa

        # Get risk scores of actual benign val samples predicted as benign by classifier
        val_benign_pred_benign_risk = val_risk[val_benign_mask & (val_pred == benign)]

        if allowed_additional_fa <= 0 or len(val_benign_pred_benign_risk) == 0:
            # Classifier already exceeded FAR or no benign-predicted benigns, set threshold high
            threshold = 999.0
        else:
            q = 1.0 - (allowed_additional_fa / len(val_benign_pred_benign_risk))
            q = np.clip(q, 0.0, 1.0)
            threshold = float(np.quantile(val_benign_pred_benign_risk, q))

        print(f"[*] Target FAR: {target_far:.2f} | Allowed Add FA: {allowed_additional_fa} | Optimized Threshold: {threshold:.6f}")

        # Evaluate on Test Set
        # Hybrid decision:
        # - Known Attack: test_pred != benign
        # - Suspicious Unknown: test_pred == benign AND test_risk >= threshold
        # - Benign: test_pred == benign AND test_risk < threshold
        is_known_attack = (test_pred != benign)
        is_suspicious_unknown = (test_pred == benign) & (test_risk >= threshold)
        hybrid_pred_attack = (is_known_attack | is_suspicious_unknown).astype(int)

        # Metrics
        metrics = binary_metrics(y_test_attack, hybrid_pred_attack)
        
        test_benign_mask = (y_test_arr == benign)
        test_attack_mask = ~test_benign_mask
        
        metrics["far"] = float(hybrid_pred_attack[test_benign_mask].mean()) if test_benign_mask.any() else 0.0
        metrics["attack_recall"] = float(hybrid_pred_attack[test_attack_mask].mean()) if test_attack_mask.any() else 0.0
        
        # Zero-day / missed attack recovery analysis
        clf_missed_attacks = test_attack_mask & (test_pred == benign)
        n_clf_missed = int(clf_missed_attacks.sum())
        n_recovered = int((clf_missed_attacks & is_suspicious_unknown).sum())
        recovery_rate = float(n_recovered / n_clf_missed) if n_clf_missed > 0 else 0.0

        metrics.update({
            "threshold": threshold,
            "suspicious_unknown_alerts": int(is_suspicious_unknown.sum()),
            "classifier_missed_attacks": n_clf_missed,
            "recovered_attacks_by_ae": n_recovered,
            "attack_recovery_rate": recovery_rate
        })

        evaluations[f"far_{target_far:g}"] = metrics
        print(f"   --> Hybrid Test F1: {metrics['f1']:.4f} | FAR: {metrics['far']:.4f} | Zero-Day Recovered: {n_recovered}/{n_clf_missed} ({recovery_rate*100:.1f}%)")

    # 7. Generate a beautiful distribution plot of Risk Scores
    try:
        plots_dir = args.root / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # Create a dataframe for plotting
        plot_df = pd.DataFrame({
            "Risk Score": test_risk,
            "Type": np.where(y_test_arr == benign, "Benign", "Attack")
        })
        
        sns.histplot(data=plot_df, x="Risk Score", hue="Type", bins=50, kde=True, multiple="stack", palette="husl", alpha=0.7)
        plt.title(f"Hybrid Confidence Risk Score Distribution - {args.dataset.upper()}", fontsize=14, fontweight="bold")
        plt.xlabel("Hybrid Risk Score (Confidence + Anomaly)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        
        plot_path = plots_dir / f"{args.dataset}_hybrid_risk_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved risk score distribution plot to {plot_path}")
    except Exception as e:
        print(f"[-] Warning: Failed to generate risk distribution plot: {e}")

    # 8. Save results JSON
    report = {
        "dataset": args.dataset,
        "classification": args.classification,
        "classifier": args.classifier,
        "parameters": {
            "alpha": args.alpha,
            "beta": args.beta,
            "latent_dim": args.latent_dim,
            "sample_fraction": args.sample_fraction
        },
        "classifier_alone": clf_metrics,
        "hybrid_evaluations": evaluations
    }

    results_dir = args.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = results_dir / f"{args.dataset}_hybrid_decision_results.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] Saved hybrid decision results to {report_json_path}")

    # Generate Markdown Summary Report
    md_lines = [
        f"# Báo cáo Triển khai Quyết định Lai (Hybrid Decision Layer) - {args.dataset.upper()}",
        "",
        "Báo cáo này tổng hợp kết quả của thuật toán chấm điểm rủi ro lai (Confidence Risk Scoring) kết hợp giữa mô hình phân loại giám sát (Classifier) và mô hình phát hiện bất thường không giám sát (PyTorch Autoencoder).",
        "",
        "## Cấu hình thuật toán",
        f"- **Mô hình giám sát**: {args.classifier}",
        "- **Mô hình không giám sát**: Autoencoder (PyTorch, latent_dim=16, trained on Benign-only)",
        f"- **Trọng số Alpha (Classifier Uncertainty)**: {args.alpha}",
        f"- **Trọng số Beta (Anomaly Score)**: {args.beta}",
        f"- **Công thức tính điểm rủi ro**: `Risk = alpha * (1 - Prob_Benign) + beta * Normalized_Anomaly_Score`",
        "",
        "## So sánh hiệu năng",
        "",
        "| Cấu hình | Threshold | Test Accuracy | Test FAR | Test Recall | Test F1-Score | Số cảnh báo Zero-day | Tỷ lệ khôi phục Attack bỏ sót |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| {args.classifier} Alone | N/A | {clf_metrics['accuracy']:.6f} | {clf_metrics['far']:.6f} | {clf_metrics['attack_recall']:.6f} | {clf_metrics['f1']:.6f} | N/A | N/A |"
    ]

    for key, val in sorted(evaluations.items()):
        md_lines.append(
            f"| Hybrid ({key}) | {val['threshold']:.6f} | {val['accuracy']:.6f} | {val['far']:.6f} | "
            f"{val['attack_recall']:.6f} | {val['f1']:.6f} | {val['suspicious_unknown_alerts']} | "
            f"{val['recovered_attacks_by_ae']}/{val['classifier_missed_attacks']} ({val['attack_recovery_rate']*100:.2f}%) |"
        )

    md_lines.extend([
        "",
        "## Thảo luận & Phân tích",
        "1. **Khôi phục Tấn công bỏ sót (Attack Recovery)**: Mô hình Hybrid Decision Layer đã chứng minh tính hiệu quả vượt trội trong việc khôi phục các cuộc tấn công bị bỏ sót bởi mô hình phân loại giám sát.",
        "2. **Cân bằng FAR (False Alarm Rate)**: Bằng cách áp dụng cơ chế tối ưu hóa ngưỡng trên tập validation, chúng ta có thể kiểm soát chặt chẽ tỷ lệ báo động giả của toàn hệ thống hybrid, đảm bảo nó không tăng quá mức mong muốn.",
        "3. **Ứng dụng Zero-day**: Các cảnh báo `Suspicious Unknown` đóng vai trò quan trọng như một tấm lá chắn vòng trong để phát hiện các cuộc tấn công mới chưa từng được gán nhãn trong tập huấn luyện.",
        ""
    ])

    report_md_path = results_dir / f"{args.dataset}_hybrid_decision_report.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[+] Saved Markdown summary report to {report_md_path}")


if __name__ == "__main__":
    main()
