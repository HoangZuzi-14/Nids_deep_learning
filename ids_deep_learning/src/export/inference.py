import json
from pathlib import Path

import joblib
import pandas as pd
import torch


class TabularInferencePipeline:
    def __init__(self, model, scaler_path, label_mapping_path, feature_names, device=None):
        self.model = model
        self.scaler = joblib.load(scaler_path)
        self.label_mapping = json.loads(Path(label_mapping_path).read_text(encoding="utf-8"))
        self.inverse_labels = {v: k for k, v in self.label_mapping.items()}
        self.feature_names = list(feature_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        X = self.scaler.transform(df[self.feature_names])
        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu()
        pred = probs.argmax(dim=1).numpy()
        return pd.DataFrame(
            {
                "predicted_class": [self.inverse_labels[int(i)] for i in pred],
                "probability": probs.max(dim=1).values.numpy(),
            }
        )

