import json
from pathlib import Path

import pandas as pd


def analyze_class_distribution(y, class_names=None, output_path=None):
    series = pd.Series(y)
    counts = series.value_counts().sort_index()
    total = int(counts.sum())
    rows = []
    for label, count in counts.items():
        label_int = int(label)
        name = class_names[label_int] if class_names and label_int < len(class_names) else label
        rows.append({"label": name, "count": int(count), "ratio": float(count / total)})
    report = {"total": total, "classes": rows}
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

