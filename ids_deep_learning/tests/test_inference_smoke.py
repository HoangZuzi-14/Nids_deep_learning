from __future__ import annotations

import json
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from src.pipeline.baseline_runner import benign_label_index

ROOT = Path(__file__).resolve().parents[1]


class TestInferenceSmoke(unittest.TestCase):
    def setUp(self):
        self.root = ROOT
        # Check if cache folders exist, otherwise skip
        self.datasets = ["nsl_kdd", "unsw_nb15", "cicids2017"]

    def test_nsl_kdd_inference_smoke(self):
        dataset = "nsl_kdd"
        artifact_dir = self.root / "artifacts" / dataset / "multi"
        cache_path = self.root / "data/cache/nsl_kdd/KDDTrain+.txt"

        if not cache_path.exists() or not artifact_dir.exists():
            self.skipTest(f"NSL-KDD data or artifacts missing at {cache_path}")

        # 1. Reload metadata, scaler, label mapping
        scaler = joblib.load(artifact_dir / "scaler.pkl")
        label_mapping = json.loads((artifact_dir / "label_mapping.json").read_text(encoding="utf-8"))
        inverse_labels = {v: k for k, v in label_mapping.items()}
        
        config = json.loads((artifact_dir / "inference_config.json").read_text(encoding="utf-8"))
        feature_names = config["feature_names"]

        # 2. Load some samples (columns from KDDTrain+ txt)
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
        df = pd.read_csv(cache_path, names=columns, nrows=10, low_memory=False)

        # 3. Fit categorical encoders if needed or just use numeric features.
        # Let's see: the feature names are exactly the 40 numeric/processed features.
        # Let's align features to scaler's expected columns!
        # Scaler expects `X` of shape (10, 40)
        # Let's preprocess the 10 samples exactly as the pipeline would
        X_df = df[feature_names].copy()
        
        # Squeeze numeric or encode if needed (since scaler is already fitted, we just transform)
        # First check features that are categoricals and map them
        encoders_dict = joblib.load(artifact_dir / "encoders.pkl")
        cat_encoders = encoders_dict.get("categorical", {})
        for col, encoder in cat_encoders.items():
            if col in X_df.columns:
                X_df[[col]] = encoder.transform(X_df[[col]].astype(str))

        X_scaled = scaler.transform(X_df)

        # 4. Predict using RandomForest.pkl
        rf_model = joblib.load(artifact_dir / "RandomForest.pkl")
        rf_preds = rf_model.predict(X_df) # RF usually takes raw encoded data before standard scaling or scaled depending on the pipeline
        self.assertEqual(len(rf_preds), 10)
        
        # 5. Predict using MLP.onnx
        onnx_path = artifact_dir / "MLP.onnx"
        self.assertTrue(onnx_path.exists(), f"ONNX model missing at {onnx_path}")
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
        
        # Softmax on logits
        exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_classes = probs.argmax(axis=1)
        
        self.assertEqual(len(pred_classes), 10)
        print(f"[+] NSL-KDD Smoke Test Successful! ONNX sample predictions: {[inverse_labels[c] for c in pred_classes]}")

    def test_unsw_nb15_inference_smoke(self):
        dataset = "unsw_nb15"
        artifact_dir = self.root / "artifacts" / dataset / "multi"
        cache_path = self.root / "data/cache/unsw_nb15/training-set.csv"

        if not cache_path.exists() or not artifact_dir.exists():
            self.skipTest(f"UNSW-NB15 data or artifacts missing at {cache_path}")

        # 1. Reload
        scaler = joblib.load(artifact_dir / "scaler.pkl")
        label_mapping = json.loads((artifact_dir / "label_mapping.json").read_text(encoding="utf-8"))
        inverse_labels = {v: k for k, v in label_mapping.items()}
        
        config = json.loads((artifact_dir / "inference_config.json").read_text(encoding="utf-8"))
        feature_names = config["feature_names"]

        # 2. Load some samples
        df = pd.read_csv(cache_path, nrows=10, low_memory=False)
        X_df = df[feature_names].copy()
        
        encoders_dict = joblib.load(artifact_dir / "encoders.pkl")
        cat_encoders = encoders_dict.get("categorical", {})
        for col, encoder in cat_encoders.items():
            if col in X_df.columns:
                X_df[[col]] = encoder.transform(X_df[[col]].astype(str))

        X_scaled = scaler.transform(X_df)

        # 3. Predict using MLP.onnx
        onnx_path = artifact_dir / "MLP.onnx"
        self.assertTrue(onnx_path.exists())
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
        
        pred_classes = onnx_logits.argmax(axis=1)
        self.assertEqual(len(pred_classes), 10)
        print(f"[+] UNSW-NB15 Smoke Test Successful! ONNX sample predictions: {[inverse_labels[c] for c in pred_classes]}")


if __name__ == "__main__":
    unittest.main()
