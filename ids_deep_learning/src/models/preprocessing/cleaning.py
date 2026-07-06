import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_before: int
    missing_after: int
    constant_columns_removed: list[str]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = []
    for column in df.columns:
        name = str(column).strip().lower()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        normalized.append(name)
    df.columns = normalized
    return df


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_constant_columns: bool = True,
    fill_numeric_with_median: bool = True,
    drop_remaining_missing: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    rows_before = len(df)
    df = normalize_column_names(df)

    duplicates = int(df.duplicated().sum())
    if drop_duplicates and duplicates:
        df = df.drop_duplicates()

    df = df.replace([np.inf, -np.inf], np.nan)
    missing_before = int(df.isna().sum().sum())

    if fill_numeric_with_median:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    if drop_remaining_missing:
        df = df.dropna()

    removed = []
    protected_columns = {"label", "attack_cat", "target", "target_name"}
    if drop_constant_columns:
        for col in list(df.columns):
            if col in protected_columns:
                continue
            if df[col].nunique(dropna=False) <= 1:
                removed.append(col)
        if removed:
            df = df.drop(columns=removed)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(df),
        duplicates_removed=duplicates if drop_duplicates else 0,
        missing_before=missing_before,
        missing_after=int(df.isna().sum().sum()),
        constant_columns_removed=removed,
    )
    return df, report
