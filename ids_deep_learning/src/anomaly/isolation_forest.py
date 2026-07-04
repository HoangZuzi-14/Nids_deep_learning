from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_isolation_forest(
    X_normal,
    contamination: float | str = "auto",
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    seed: int = 42,
):
    """Fit Isolation Forest on normal-only feature rows."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def compute_anomaly_scores(model, scaler, X):
    """Return scores where higher means more anomalous."""
    X_scaled = scaler.transform(X)
    return -model.decision_function(X_scaled)


def threshold_at_far(normal_scores, target_far: float = 0.05) -> float:
    """Choose anomaly threshold from normal validation scores for target FAR."""
    if not 0.0 < target_far < 1.0:
        raise ValueError("target_far must be between 0 and 1.")
    normal_scores = np.asarray(normal_scores, dtype=float)
    return float(np.quantile(normal_scores, 1.0 - target_far))
