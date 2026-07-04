import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fit_transform_categoricals(
    df: pd.DataFrame, categorical_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, OrdinalEncoder]]:
    df = df.copy()
    encoders = {}
    for col in categorical_columns:
        if col not in df.columns:
            continue
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[[col]] = encoder.fit_transform(df[[col]].astype(str))
        encoders[col] = encoder
    return df, encoders


def transform_categoricals(
    df: pd.DataFrame, encoders: dict[str, OrdinalEncoder]
) -> pd.DataFrame:
    df = df.copy()
    for col, encoder in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")
        df[[col]] = encoder.transform(df[[col]].astype(str))
    return df

