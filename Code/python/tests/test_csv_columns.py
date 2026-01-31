from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_compute_estimates():
    """Run the compute_estimates script to ensure Data-derived CSVs are generated."""
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "Code" / "python" / "compute_estimates.py")], check=True)
    assert proc.returncode == 0


def test_employment_csv_has_expected_columns():
    run_compute_estimates()
    csv_path = REPO_ROOT / "Data-derived" / "Estimated Treatment Effect on Cumulative Employment (36 Quarters).csv"
    df = pd.read_csv(csv_path)

    expected = {
        "quarter",
        "experimental",
        "experimental_se",
        "surrogate_index",
        "surrogate_index_se",
        "naive",
        "naive_se",
        "single_surrogate",
        "single_surrogate_se",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing expected columns in employment CSV: {missing}"


def test_other_sites_csv_has_q6_columns():
    run_compute_estimates()
    csv_path = REPO_ROOT / "Data-derived" / "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Employment).csv"
    df = pd.read_csv(csv_path)

    expected_prefixes = ["rs", "la", "sd", "al"]
    expected_columns = set()
    for p in expected_prefixes:
        expected_columns.update({f"surrogate_index_{p}_q6", f"experimental_{p}_q6", f"experimental_{p}_q6_se"})

    missing = expected_columns - set(col.lower() for col in df.columns)
    assert not missing, f"Missing expected q6 columns in other-sites CSV: {missing}"
