import json
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


NSL_KDD_MULTICLASS = {
    "normal": "Benign",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
}

CICIDS2017_MULTICLASS = {
    "benign": "Benign",
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "portscan": "PortScan",
    "bot": "Botnet",
    "ftp-patator": "BruteForce",
    "ftp patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "ssh patator": "BruteForce",
    "web attack brute force": "WebAttack",
    "web attack xss": "WebAttack",
    "web attack sql injection": "WebAttack",
    "web attack brute force": "WebAttack",
    "web attack xss": "WebAttack",
    "web attack sql injection": "WebAttack",
    "infiltration": "Infiltration",
    "heartbleed": "Rare_Attack",
}


def _normalize_label(value) -> str:
    text = str(value).strip().lower().replace("\ufffd", " ")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_binary_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.map(_normalize_label)
    return (~normalized.isin({"normal", "benign", "0"})).astype(int)


def map_multiclass_labels(
    labels: pd.Series, mapping: dict[str, str] | None = None
) -> tuple[pd.Series, LabelEncoder, dict[str, int]]:
    normalized = labels.map(_normalize_label)
    normalized_mapping = {
        _normalize_label(source): target for source, target in (mapping or {}).items()
    }
    mapped = normalized.map(normalized_mapping).fillna(normalized.str.title())
    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(mapped), index=labels.index)
    label_mapping = {name: int(idx) for idx, name in enumerate(encoder.classes_)}
    return encoded, encoder, label_mapping


def save_label_mapping(mapping: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
