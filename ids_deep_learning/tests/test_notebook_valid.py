import json
from pathlib import Path


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def notebook_text(notebook: dict) -> str:
    pieces = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        pieces.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(pieces)


def test_existing_notebooks_are_valid_json_notebooks():
    notebook_paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert notebook_paths, "No notebooks found"

    for notebook_path in notebook_paths:
        notebook = load_notebook(notebook_path)
        assert "cells" in notebook, f"{notebook_path} missing cells"
        assert "metadata" in notebook, f"{notebook_path} missing metadata"
        assert "nbformat" in notebook, f"{notebook_path} missing nbformat"


def test_cicids2017_ml_train_eval_notebook_contract():
    notebook_path = Path("notebooks/cicids2017_colab_ml_train_eval.ipynb")
    assert notebook_path.exists(), f"Notebook file {notebook_path} does not exist"

    notebook = load_notebook(notebook_path)
    text = notebook_text(notebook)
    cell_ids = {cell.get("id") for cell in notebook.get("cells", [])}

    assert "eval-ml-baselines" in cell_ids
    assert "CICIDS2017 Colab ML Train + Eval" in text
    assert "train_eval" in text
    assert "cicids2017_multi_ml_train_eval_manifest.json" in text
    assert "cicids2017_multi_ml_baselines.json" in text
    assert "compute_classification_metrics" in text


def test_final_evaluation_notebook_contract():
    notebook_path = Path("notebooks/evaluate_final_results.ipynb")
    assert notebook_path.exists(), f"Notebook file {notebook_path} does not exist"

    notebook = load_notebook(notebook_path)
    text = notebook_text(notebook)

    assert "Final NIDS Model Evaluation" in text
    assert "final_model_evaluation.csv" in text
    assert "final_model_ranking.csv" in text
    assert "final_evaluation_summary.md" in text
    assert "cicids2017_multi_ml_baselines.json" in text
