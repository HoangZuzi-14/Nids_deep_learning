from pathlib import Path

import pytest

def test_pdf_exists_and_is_valid():
    pdf_path = Path("results/NIDS_Experimental_Evaluation_Report.pdf")
    if not pdf_path.exists():
        pytest.skip("Generated PDF report is optional and is not stored in Git.")
    
    # Check that the file size is reasonable (should be > 200 KB as it embeds 7 images)
    size_kb = pdf_path.stat().st_size / 1024
    assert size_kb > 200, f"PDF file size is too small ({size_kb:.1f} KB), images or content might be missing"
    
    # Verify PDF signature (%PDF)
    with open(pdf_path, "rb") as f:
        signature = f.read(4)
        assert signature == b"%PDF", f"PDF file has incorrect signature: {signature}"

if __name__ == "__main__":
    test_pdf_exists_and_is_valid()
    print("PDF validation check PASSED successfully!")
