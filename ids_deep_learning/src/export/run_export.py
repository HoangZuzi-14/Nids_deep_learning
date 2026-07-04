from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.models.mlp import MLP
from src.export.export_onnx import export_onnx_model

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import sys
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="Export trained PyTorch MLP models to ONNX.")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "cicids2017"], default="nsl_kdd")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    artifact_dir = args.root / "artifacts" / args.dataset / "multi"
    config_path = artifact_dir / "inference_config.json"

    if not config_path.exists():
        print(f"[-] Inference config not found: {config_path}")
        return

    # Load inference config to get features & label mapping
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_names = config["feature_names"]
    
    label_mapping_file = config.get("label_mapping", "label_mapping.json")
    label_mapping_path = artifact_dir / label_mapping_file
    if not label_mapping_path.exists():
        print(f"[-] Label mapping not found: {label_mapping_path}")
        return
        
    label_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    
    n_features = len(feature_names)
    n_classes = len(label_mapping)
    print(f"[*] Dataset: {args.dataset} | Features: {n_features} | Classes: {n_classes}")

    # Try loading MLP model weights
    model_paths = [
        artifact_dir / "MLP_Modular.pt",
        artifact_dir / "MLP.pt",
        artifact_dir / "MLP_Smoke.pt"
    ]
    
    loaded = False
    model = MLP(n_features=n_features, n_classes=n_classes)
    
    for path in model_paths:
        if path.exists():
            print(f"[*] Trying to load PyTorch state dict from: {path}")
            try:
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                loaded = True
                print(f"[+] Successfully loaded model weights from {path}")
                break
            except Exception as e:
                print(f"[-] Failed to load weights from {path}: {e}")
                
    if not loaded:
        print("[-] Could not load any pre-trained MLP model weights. Cannot export.")
        return

    model.eval()

    # Create dummy/sample input
    sample_input = torch.randn(1, n_features, dtype=torch.float32)
    onnx_path = artifact_dir / "MLP.onnx"
    
    print(f"[*] Exporting model to ONNX: {onnx_path}...")
    try:
        export_onnx_model(model, sample_input, onnx_path)
        print(f"[+] Model exported to ONNX successfully!")
    except Exception as e:
        print(f"[-] Failed to export ONNX: {e}")
        return

    # Verify ONNX model using onnxruntime
    print("[*] Verifying ONNX model using onnxruntime...")
    try:
        import onnxruntime as ort
        
        # PyTorch prediction
        with torch.no_grad():
            py_logits = model(sample_input).numpy()
            
        # ONNX prediction
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: sample_input.numpy()})[0]
        
        # Check difference
        diff = np.max(np.abs(py_logits - onnx_logits))
        print(f"[+] ONNX validation complete. Max logit difference: {diff:.6e}")
        if diff < 1e-4:
            print("[+] ONNX model matches PyTorch model outputs perfectly!")
        else:
            print("[-] Warning: ONNX outputs differ slightly from PyTorch outputs.")
    except ImportError:
        print("[!] onnxruntime not installed. Skipping output match verification.")
    except Exception as e:
        print(f"[-] Error during ONNX verification: {e}")


if __name__ == "__main__":
    main()
