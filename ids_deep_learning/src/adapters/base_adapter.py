from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.encoding import fit_transform_categoricals, transform_categoricals
from src.preprocessing.label_mapping import map_binary_labels, map_multiclass_labels
from src.preprocessing.scaling import fit_scaler, transform_with_scaler


@dataclass
class AdapterOutput:
    X_train: object
    X_val: object
    X_test: object
    y_train: object
    y_val: object
    y_test: object
    feature_names: list[str]
    label_mapping: dict[str, int]
    metadata: dict
    scaler: object
    encoders: dict
    imputer: object


class DatasetAdapter:
    name = "base"
    columns: list[str] | None = None
    drop_columns: list[str] = []
    categorical_columns: list[str] = []
    label_column = "label"
    binary_label_column: str | None = None
    multiclass_mapping: dict[str, str] | None = None

    def __init__(
        self,
        cache_path: str | Path | None = None,
        remote_url: str | None = None,
        seed: int = 42,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.remote_url = remote_url
        self.seed = seed

    def load(self) -> pd.DataFrame:
        path = self._ensure_cache()
        if self.columns:
            return pd.read_csv(path, names=self.columns, low_memory=False)
        return pd.read_csv(path, low_memory=False)

    def _ensure_cache(self) -> Path:
        if self.cache_path and self.cache_path.exists():
            return self.cache_path
        if not self.remote_url or not self.cache_path:
            raise FileNotFoundError(f"No local cache or remote URL configured for {self.name}")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(self.remote_url, self.cache_path)
        return self.cache_path

    def clean(self, df: pd.DataFrame):
        df = df.drop(columns=self.drop_columns, errors="ignore")
        return clean_dataframe(
            df,
            fill_numeric_with_median=False,
            drop_constant_columns=False,
            drop_remaining_missing=False,
        )

    def map_labels(self, df: pd.DataFrame, classification_type: str = "multi"):
        if classification_type == "binary":
            source = self.binary_label_column or self.label_column
            return map_binary_labels(df[source]), None, {"Benign": 0, "Attack": 1}
        return map_multiclass_labels(df[self.label_column], self.multiclass_mapping)

    def split(self, X, y, test_size: float = 0.2, val_size: float = 0.2):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.seed,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val,
            random_state=self.seed,
            stratify=y_train_val,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(
        self,
        df: pd.DataFrame | None = None,
        classification_type: str = "multi",
        test_size: float = 0.2,
        val_size: float = 0.2,
        scaler_type: str = "standard",
    ) -> AdapterOutput:
        df = self.load() if df is None else df
        df, cleaning_report = self.clean(df)
        label_cols = [
            col
            for col in {self.label_column, self.binary_label_column}
            if col and col in df.columns
        ]
        if label_cols:
            df = df.dropna(subset=label_cols)
        y, label_encoder, label_mapping = self.map_labels(df, classification_type)
        if y.nunique() < 2:
            raise ValueError(
                f"{self.name} {classification_type} preprocessing requires at least 2 classes; "
                f"found {y.nunique()} class. Check dataset cache/raw files and label mapping."
            )

        drop_cols = {self.label_column, self.binary_label_column, "target", "target_name"}
        X = df.drop(columns=[c for c in drop_cols if c and c in df.columns], errors="ignore")

        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y, test_size, val_size)

        X_train, categorical_encoders = fit_transform_categoricals(X_train, self.categorical_columns)
        X_val = transform_categoricals(X_val, categorical_encoders)
        X_test = transform_categoricals(X_test, categorical_encoders)

        X_train = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_val = X_val.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        usable_columns = X_train.columns[~X_train.isna().all()].tolist()
        X_train = X_train[usable_columns]
        X_val = X_val[usable_columns]
        X_test = X_test[usable_columns]

        non_constant_columns = [
            col for col in usable_columns if X_train[col].nunique(dropna=True) > 1
        ]
        if non_constant_columns:
            X_train = X_train[non_constant_columns]
            X_val = X_val[non_constant_columns]
            X_test = X_test[non_constant_columns]

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)

        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
        X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_train.columns, index=X_val.index)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_train.columns, index=X_test.index)

        X_train_scaled, scaler = fit_scaler(X_train_imputed, scaler_type)
        X_val_scaled = transform_with_scaler(X_val_imputed, scaler)
        X_test_scaled = transform_with_scaler(X_test_imputed, scaler)

        return AdapterOutput(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=y_train.to_numpy(),
            y_val=y_val.to_numpy(),
            y_test=y_test.to_numpy(),
            feature_names=X_train.columns.tolist(),
            label_mapping=label_mapping,
            metadata={"dataset": self.name, "cleaning": cleaning_report.__dict__},
            scaler=scaler,
            encoders={"categorical": categorical_encoders, "label": label_encoder},
            imputer=imputer,
        )
