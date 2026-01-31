# Python replication scripts (recommended)

This folder contains the Python implementation that reproduces the main analyses
from Athey, Chetty, Imbens, and Kang (2019), "The Surrogate Index." The Python
workflow is the recommended entrypoint for reproducing tables and figures using
the simulated GAIN data included in `Data-raw/`.

---

## Quick overview ✅
- Primary entrypoint: `Code/python/compute_estimates.py` — computes estimates and
  writes intermediate CSVs to `Data-derived/`.
- Optional/auxiliary: `Code/python/bounds_ci.py` — computes bootstrap CIs and
  bounds on bias.
- Output generation: `Code/python/figures.py` and `Code/python/tables.py` —
  produce PNG figures and formatted Excel tables in `Output/`.
- Tests: see `Code/python/tests/` (pytest-based smoke tests that validate CSV
  columns and pipeline invariants).

## Reproducing the Python results
1. Install a Python 3.11+ environment (see `requirements.txt` in the repo root).
2. Run in order:

```bash
python Code/python/compute_estimates.py
python Code/python/bounds_ci.py   # optional
python Code/python/figures.py
python Code/python/tables.py
```

This sequence produces CSVs in `Data-derived/` and figures/Excel in `Output/`.

---

## Tests and development
- Run the test suite with `pytest` from the repository root.
- When adding features, please add tests under `Code/python/tests/` to
  prevent regressions.

---

## Notes
- Legacy Stata scripts are archived in `legacy-stata/`. For instructions on the
  original Stata workflow, please consult the upstream repository at
  https://github.com/OpportunityInsights/Surrogates-Replication-Code (we do not
  duplicate detailed Stata run instructions here).

If you want, I can also add a brief `CONTRIBUTING.md` and a small `requirements-dev.txt` for development tooling. Let me know!
