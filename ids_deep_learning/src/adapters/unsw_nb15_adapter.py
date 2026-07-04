from src.adapters.base_adapter import DatasetAdapter


class UNSWNB15Adapter(DatasetAdapter):
    name = "UNSW-NB15"
    drop_columns = ["id"]
    categorical_columns = ["proto", "service", "state"]
    label_column = "attack_cat"
    binary_label_column = "label"

