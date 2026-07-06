import pandas as pd
import pytest

from src.adapters.base_adapter import DatasetAdapter


class FixedSplitAdapter(DatasetAdapter):
    name = "fixed-split"
    label_column = "label"
    categorical_columns = ["protocol"]

    def split(self, X, y, test_size=0.2, val_size=0.2):
        return (
            X.iloc[[0, 1]],
            X.iloc[[2]],
            X.iloc[[3]],
            y.iloc[[0, 1]],
            y.iloc[[2]],
            y.iloc[[3]],
        )


def test_preprocess_fits_categorical_encoder_on_train_only():
    df = pd.DataFrame(
        {
            "protocol": ["tcp", "udp", "future_val_only", "future_test_only"],
            "bytes": [10.0, 20.0, 30.0, 40.0],
            "label": ["Benign", "Attack", "Benign", "Attack"],
        }
    )

    output = FixedSplitAdapter().preprocess(df=df, classification_type="binary")

    encoder = output.encoders["categorical"]["protocol"]
    assert encoder.categories_[0].tolist() == ["tcp", "udp"]
    assert output.X_val[0, 0] == -3
    assert output.X_test[0, 0] == -3


def test_preprocess_fits_numeric_imputer_on_train_only_before_scaling():
    df = pd.DataFrame(
        {
            "protocol": ["tcp", "udp", "tcp", "udp"],
            "bytes": [10.0, 20.0, None, None],
            "label": ["Benign", "Attack", "Benign", "Attack"],
        }
    )

    output = FixedSplitAdapter().preprocess(df=df, classification_type="binary")

    assert output.imputer.statistics_.tolist() == [0.5, 15.0]


def test_preprocess_rejects_single_class_training_data():
    df = pd.DataFrame(
        {
            "protocol": ["tcp", "udp", "tcp", "udp"],
            "bytes": [10.0, 20.0, 30.0, 40.0],
            "label": ["Benign", "Benign", "Benign", "Benign"],
        }
    )

    with pytest.raises(ValueError, match="requires at least 2 classes"):
        FixedSplitAdapter().preprocess(df=df, classification_type="binary")
