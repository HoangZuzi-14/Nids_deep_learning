import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def fit_scaler(X_train: pd.DataFrame, scaler_type: str = "standard"):
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    return X_scaled, scaler


def transform_with_scaler(X: pd.DataFrame, scaler):
    return scaler.transform(X)

