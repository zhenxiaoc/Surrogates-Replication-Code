# Replication Code for "The Surrogate Index"

This repository contains replication code and a Python implementation of the analyses in Athey, Chetty, Imbens, and Kang (2019), "The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely." The Python implementation (recommended) lives under `Code/python/` and reproduces the main tables and figures using the simulated GAIN data included in `Data-raw/`.

---

## Quick overview ✅
- Primary entrypoint: `Code/python/compute_estimates.py` — computes estimates and writes intermediate CSV files to `Data-derived/`.
- Optional/auxiliary: `Code/python/bounds_ci.py` — computes bootstrap CIs / bounds on bias (used to create the CI-on-bounds output in `Data-derived/`).
- Output generation: `Code/python/figures.py` and `Code/python/tables.py` — generate figures (PNG files in `Output/`) and formatted tables (Excel in `Output/`).
- Tests: `Code/python/tests/` includes lightweight pytest tests that assert expected CSV columns and basic pipeline invariants.

---

## Stata code
- Legacy Stata replication scripts have been archived to `legacy-stata/` in this repository.
- For instructions on the original Stata-based workflow, and how to run those `.do` files, please consult the original repository from which this project was forked: https://github.com/OpportunityInsights/Surrogates-Replication-Code
  (We intentionally do not duplicate detailed Stata run instructions here.)

---

## Files and what to run 🔧
Below is a short description of the important files and the order to run them to reproduce the main outputs (Python workflow):

- `Code/python/config.py`
  - Small configuration values used across scripts (file paths, constants). No direct entrypoint — imported by other scripts.

- `Code/python/data.py`
  - Data loading and processing helpers. Used by other scripts; not an entrypoint by itself but useful for debugging and dataset inspection.

- `Code/python/compute_estimates.py`  (run first)
  - Core script that computes treatment-effect estimates and surrogate index estimates for the simulated GAIN dataset.
  - Writes intermediate CSVs into `Data-derived/` (these CSVs are read by `figures.py` and `tables.py`).

- `Code/python/bounds_ci.py`  (run after `compute_estimates.py` if you need bootstrap CIs)
  - Computes bootstrap-based confidence intervals and bias bounds. Generates the CSV `Data-derived/CI on Bounds for Degree of Bias.csv` and related outputs.

- `Code/python/regression.py`
  - Small utility wrapper for OLS regressions used by estimation and bootstrap code.

- `Code/python/figures.py`  (run after required CSVs exist)
  - Reads `Data-derived/` CSVs and produces the figures saved to `Output/` (PNG).

- `Code/python/tables.py`  (run after required CSVs exist)
  - Generates formatted tables and writes the formatted Excel output to `Output/`.

- `Code/python/bounds_ci.py` and other scripts may be configured via `config.py` for parameters such as bootstrap repetitions; see the top of each file for usage notes.

---

## Reproducing the main results (Python)
1. Ensure a Python 3.11+ environment with the required packages: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `openpyxl`, `pytest`.
2. From the repository root, run:

```bash
python Code/python/compute_estimates.py
python Code/python/bounds_ci.py    # optional, for CIs
python Code/python/figures.py
python Code/python/tables.py
```

This sequence produces CSVs in `Data-derived/` and figures/excel files in `Output/`.

---

## Tests and CI ✅
- Unit tests live in `Code/python/tests/` and are executed with `pytest`.
- A GitHub Actions workflow is included to run the test suite on push / PR for continuous integration.

---

## Notes and contribution pointers 💡
- The Python implementation is the recommended, actively maintained route for reproducing the analysis in this repo. The original Stata scripts are preserved in `legacy-stata/` for archival and reference.
- If you are contributing changes, please update or add tests under `Code/python/tests/` to avoid regressions.

---

If you'd like, I can also update `Code/python/README.md` to match this overview and/or add a `requirements.txt` to make environment setup reproducible. Would you like me to do either of those next? ✨

