import json
from pathlib import Path

def test_notebook_exists_and_is_valid():
    notebook_path = Path("notebooks/visualize_report.ipynb")
    assert notebook_path.exists(), f"Notebook file {notebook_path} does not exist"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as e:
            assert False, f"Failed to parse notebook as JSON: {e}"
            
    # Check basic notebook keys
    assert "cells" in nb, "Notebook missing 'cells' key"
    assert "metadata" in nb, "Notebook missing 'metadata' key"
    assert "nbformat" in nb, "Notebook missing 'nbformat' key"
    
    # Check sections
    cells = nb["cells"]
    markdown_content = ""
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            markdown_content += "\n" + "".join(cell.get("source", []))
            
    assert "Imbalance" in markdown_content, "Missing Imbalance Handling section"
    assert "NSL-KDD" in markdown_content, "Missing NSL-KDD comparison section"
    assert "Domain Shift" in markdown_content, "Missing Domain Shift section"
    assert "Hybrid" in markdown_content, "Missing Hybrid Decision section"

if __name__ == "__main__":
    test_notebook_exists_and_is_valid()
    print("Notebook validation check PASSED successfully!")

