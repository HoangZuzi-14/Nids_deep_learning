from .base_adapter import AdapterOutput, DatasetAdapter
from .cicids2017_adapter import CICIDS2017Adapter
from .nsl_kdd_adapter import NSLKDDAdapter
from .unsw_nb15_adapter import UNSWNB15Adapter

# Backward-compatible alias for notebook cells that used a different spelling.
CICIDS2017Adapter = CICIDS2017Adapter

__all__ = [
    "AdapterOutput",
    "DatasetAdapter",
    "CICIDS2017Adapter",
    "CICIDS2017Adapter",
    "NSLKDDAdapter",
    "UNSWNB15Adapter",
]
