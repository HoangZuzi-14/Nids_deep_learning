from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.mlp import MLP
from src.pipeline.baseline_runner import benign_label_index


def main() -> None:
    # Handle Windows console encoding for print support of tables/emojis
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="NIDS Flow Predictor CLI Tool.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to input network flow CSV file.")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], required=True, help="Dataset configuration to reload.")
    parser.add_argument("--model", choices=["RandomForest", "MLP", "ONNX"], default="RandomForest", help="Model format/type to use for prediction.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of sample predictions to show in terminal.")
    args = parser.parse_args()

    print("=" * 60)
    print(f"[*] NIDS CLI INFERENCE TOOL | Dataset: {args.dataset.upper()} | Model: {args.model}")
    print("=" * 60)

    # 1. Check file existence
    if not args.csv.exists():
        print(f"[-] Input CSV file not found: {args.csv}")
        return

    artifact_dir = ROOT / "artifacts" / args.dataset / "multi"
    if not artifact_dir.exists():
        print(f"[-] Artifacts folder not found for {args.dataset} multi-class: {artifact_dir}")
        return

    # 2. Reload preprocessing artifacts
    print("[*] Loading scaler, encoders, and label mapping...")
    try:
        scaler = joblib.load(artifact_dir / "scaler.pkl")
        encoders_dict = joblib.load(artifact_dir / "encoders.pkl")
        cat_encoders = encoders_dict.get("categorical", {})
        
        label_mapping = json.loads((artifact_dir / "label_mapping.json").read_text(encoding="utf-8"))
        inverse_labels = {v: k for k, v in label_mapping.items()}
        
        config = json.loads((artifact_dir / "inference_config.json").read_text(encoding="utf-8"))
        feature_names = config["feature_names"]
    except Exception as e:
        print(f"[-] Failed to load preprocessing artifacts: {e}")
        return

    n_features = len(feature_names)
    n_classes = len(label_mapping)

    # 3. Load Input CSV data
    print(f"[*] Loading input CSV file: {args.csv}...")
    try:
        # If NSL-KDD cached txt has no header, we must handle it gracefully
        if args.dataset == "nsl_kdd" and "KDDTrain" in str(args.csv):
            # Custom column names from schema
            columns = [
                "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
                "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
                "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
                "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
                "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
                "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
                "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
                "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
                "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
                "dst_host_srv_rerror_rate", "label", "difficulty"
            ]
            df = pd.read_csv(args.csv, names=columns, low_memory=False)
        else:
            df = pd.read_csv(args.csv, low_memory=False)
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        return

    print(f"[+] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # 4. Align features and Preprocess
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        # Try finding standard mapping for UNSW-NB15/CICIDS2017 to be flexible
        print(f"[-] Warning: The following required feature columns are missing: {missing_cols}")
        print("[-] Please ensure the input CSV features are aligned with model requirements.")
        return

    try:
        X_df = df[feature_names].copy()
        
        # Apply categorical OrdinalEncoder
        for col, encoder in cat_encoders.items():
            if col in X_df.columns:
                X_df[[col]] = encoder.transform(X_df[[col]].astype(str))

        # Scale features
        X_scaled = scaler.transform(X_df)
    except Exception as e:
        print(f"[-] Preprocessing failed: {e}")
        return

    # 5. Perform Model Prediction
    print(f"[*] Running inference using {args.model} model...")
    pred_classes = []
    confidences = []

    try:
        if args.model == "RandomForest":
            model_path = artifact_dir / "RandomForest.pkl"
            if not model_path.exists():
                print(f"[-] RandomForest model file not found: {model_path}")
                return
            model = joblib.load(model_path)
            
            # Predict
            preds = model.predict(X_df)
            probs = model.predict_proba(X_df)
            
            pred_classes = [inverse_labels[int(p)] for p in preds]
            confidences = probs.max(axis=1)

        elif args.model == "MLP":
            # Find trained PyTorch MLP weights
            model_path = None
            for p in [artifact_dir / "MLP_Modular.pt", artifact_dir / "MLP.pt"]:
                if p.exists():
                    model_path = p
                    break
            if not model_path:
                print("[-] Pre-trained PyTorch MLP model weights not found in artifacts.")
                return
                
            model = MLP(n_features=n_features, n_classes=n_classes)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            
            # Predict
            tensor_input = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.softmax(logits, dim=1).numpy()
                
            preds = probs.argmax(axis=1)
            pred_classes = [inverse_labels[int(p)] for p in preds]
            confidences = probs.max(axis=1)

        elif args.model == "ONNX":
            onnx_path = artifact_dir / "MLP.onnx"
            if not onnx_path.exists():
                print(f"[-] ONNX model file not found: {onnx_path}")
                return
                
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            onnx_logits = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
            
            # Softmax to get probabilities
            exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            preds = probs.argmax(axis=1)
            pred_classes = [inverse_labels[int(p)] for p in preds]
            confidences = probs.max(axis=1)
            
    except Exception as e:
        print(f"[-] Inference failed: {e}")
        return

    # 6. Display Results
    print("\n" + "=" * 60)
    print(f"[*] TOP {args.limit} PREDICTION RESULTS (Total: {len(pred_classes)} predictions)")
    print("=" * 60)
    print(f"{'Row':<6} | {'Predicted Class':<20} | {'Confidence':<12} | {'Severity/Status':<15}")
    print("-" * 60)
    
    benign_lbl = "benign" if args.dataset != "unsw_nb15" else "normal"
    
    for i in range(min(args.limit, len(pred_classes))):
        pred_lbl = pred_classes[i]
        conf = confidences[i]
        
        status = "✅ CLEAN" if pred_lbl.lower() in {benign_lbl, "normal"} else "🚨 ATTACK"
        print(f"{i:<6} | {pred_lbl:<20} | {conf:<12.4f} | {status:<15}")
        
    print("=" * 60)
    
    # Save predictions to file if requested or by default
    output_csv = Path(args.csv.parent / f"{args.csv.stem}_predictions.csv")
    out_df = df.copy()
    out_df["Predicted_Label"] = pred_classes
    out_df["Confidence_Score"] = confidences
    out_df.to_csv(output_csv, index=False)
    print(f"[+] Full prediction results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
