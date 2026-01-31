# Changelog

All notable changes to this repository are documented in this file.

## [Unreleased] - 2026-01-31

### Added
- Vectorized bootstrap and point-estimate computations in `Code/python/bounds_ci.py` using numpy least-squares for improved performance and lower memory usage. ⚡
- `pytest` tests to assert expected CSV columns produced by `Code/python/compute_estimates.py` (`Code/python/tests/test_csv_columns.py`). ✅
- GitHub Actions workflow to run the tests on push and pull requests (`.github/workflows/python-tests.yml`). 🔁

### Changed
- Rewrote `Code/python/compute_estimates.py` to batch column additions (reduces pandas fragmentation) and added defensive behavior to always include `single_surrogate` columns for downstream robustness. 🧰

### Fixed
- Ensured figures generation and tables run end-to-end and added dependency notes. 🐛

---

## 0.1.0 - 2026-01-31
- Initial release of Python rewrite of the Surrogate Index replication (estimation, bounds, figures, tables).
