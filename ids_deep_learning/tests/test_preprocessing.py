import pandas as pd

from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.label_mapping import map_binary_labels


def test_clean_dataframe_normalizes_columns_and_inf():
    df = pd.DataFrame({" Flow Bytes/s ": [1.0, float("inf"), 3.0], "Label": ["BENIGN", "BENIGN", "Attack"]})
    cleaned, report = clean_dataframe(df, drop_duplicates=False)
    assert "flow_bytes_s" in cleaned.columns
    assert report.missing_before == 1
    assert report.missing_after == 0


def test_binary_label_mapping():
    labels = pd.Series(["BENIGN", "normal", "DoS Hulk"])
    assert map_binary_labels(labels).tolist() == [0, 0, 1]

