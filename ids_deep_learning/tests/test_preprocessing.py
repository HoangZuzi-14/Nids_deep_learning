import pandas as pd

from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.label_mapping import (
    NSL_KDD_MULTICLASS,
    map_binary_labels,
    map_multiclass_labels,
)


def test_clean_dataframe_normalizes_columns_and_inf():
    df = pd.DataFrame({" Flow Bytes/s ": [1.0, float("inf"), 3.0], "Label": ["BENIGN", "BENIGN", "Attack"]})
    cleaned, report = clean_dataframe(df, drop_duplicates=False)
    assert "flow_bytes_s" in cleaned.columns
    assert report.missing_before == 1
    assert report.missing_after == 0


def test_binary_label_mapping():
    labels = pd.Series(["BENIGN", "normal", "DoS Hulk"])
    assert map_binary_labels(labels).tolist() == [0, 0, 1]


def test_multiclass_mapping_normalizes_mapping_keys():
    labels = pd.Series(["buffer_overflow", "guess_passwd", "ftp_write"])

    encoded, _, mapping = map_multiclass_labels(labels, NSL_KDD_MULTICLASS)
    inverse = {idx: label for label, idx in mapping.items()}

    assert [inverse[int(value)] for value in encoded] == ["U2R", "R2L", "R2L"]
