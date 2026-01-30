# Python rewrite of the Stata replication scripts

This folder mirrors the Stata workflow with Python scripts that produce the same
CSV outputs and figures. The scripts are organized by responsibility:

- `config.py` – shared paths, dataset selection, and output configuration.
- `compute_estimates.py` – main estimation workflow (Sections A–E of
  `Estimate treatment effects.do`).
- `bounds_ci.py` – bootstrap confidence intervals for bounds on bias.
- `tables.py` – formatting outputs for Appendix Tables 1 and 2.
- `figures.py` – figure generation for all main and appendix figures.

## Quick start

```bash
python Code/python/compute_estimates.py
python Code/python/figures.py
python Code/python/tables.py
```

The scripts default to the simulated data bundled in `Data-raw/`. Adjust the
`data_type` or output folders in `config.py` if you are running against real
GAIN data.
