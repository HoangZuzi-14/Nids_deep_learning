from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

from src.adapters.base_adapter import DatasetAdapter
from src.preprocessing.label_mapping import CICIDS2017_MULTICLASS


class CICIDS2017Adapter(DatasetAdapter):
    name = "CICIDS2017"
    categorical_columns = []
    label_column = "label"
    multiclass_mapping = CICIDS2017_MULTICLASS

    def __init__(self, raw_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.raw_dir = Path(raw_dir) if raw_dir else None

    def load(self) -> pd.DataFrame:
        if self.cache_path and self.cache_path.exists():
            df = self._read_csv(self.cache_path)
            self._validate_label_diversity(df, self.cache_path)
            return df
        if self.raw_dir and self.raw_dir.exists():
            frames = [self._read_csv(path) for path in sorted(self.raw_dir.glob("*.csv"))]
            if not frames:
                raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")
            df = pd.concat(frames, ignore_index=True)
            self._validate_label_diversity(df, self.raw_dir)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.cache_path, index=False)
            return df
        return super().load()

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except (UnicodeDecodeError, ParserError) as exc:
                last_error = exc
        try:
            return pd.read_csv(
                path,
                encoding="cp1252",
                encoding_errors="replace",
                low_memory=False,
            )
        except TypeError:
            return pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception:
            raise last_error

    @staticmethod
    def _validate_label_diversity(df: pd.DataFrame, source: Path) -> None:
        label_columns = [col for col in df.columns if str(col).strip().lower() == "label"]
        if not label_columns:
            return
        labels = df[label_columns[0]].dropna().astype(str).str.strip()
        labels = labels[~labels.str.lower().isin({"", "nan"})]
        unique_labels = sorted(labels.str.lower().unique().tolist())
        if len(unique_labels) < 2:
            raise ValueError(
                f"CICIDS2017 source {source} contains fewer than 2 non-empty labels "
                f"({unique_labels}). Replace the raw/cache files with the full multi-class dataset."
            )
