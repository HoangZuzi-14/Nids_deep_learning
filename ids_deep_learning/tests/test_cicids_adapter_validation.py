import pandas as pd
import pytest

from src.adapters.cicids2017_adapter import CICIDS2017Adapter


def test_cicids_loader_rejects_one_class_raw_before_writing_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cache_path = tmp_path / "cache" / "merged.csv"
    pd.DataFrame(
        {
            " Flow Duration": [1, 2, 3],
            " Label": ["BENIGN", "BENIGN", "BENIGN"],
        }
    ).to_csv(raw_dir / "sample.csv", index=False)

    adapter = CICIDS2017Adapter(raw_dir=raw_dir, cache_path=cache_path)

    with pytest.raises(ValueError, match="contains fewer than 2 non-empty labels"):
        adapter.load()

    assert not cache_path.exists()
