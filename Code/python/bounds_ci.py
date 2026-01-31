from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import RunConfig, load_paths
from data import available_outcomes, outcome_columns, read_stata
from regression import fit_ols, predict_ols


def _ols_params_and_r2(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Fast OLS via numpy returning params (including intercept) and R^2.

    X can be 1-D or 2-D array of predictors. Returns beta vector (p+1,) and r2.
    """
    if X.ndim == 1:
        X_design = np.column_stack((np.ones_like(X), X))
    else:
        X_design = np.column_stack((np.ones(X.shape[0]), X))

    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return beta, float(r2)


def bootstrap_bounds(df: pd.DataFrame, outcomes: list[str], reps: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Numpy-optimized bootstrap for bounds on bias.

    This avoids assigning new columns to DataFrames inside loops and uses
    least-squares via numpy for speed.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    results: dict[str, dict[str, np.ndarray]] = {}

    # Pre-extract numeric arrays for each outcome to avoid repeated DataFrame access
    outcome_data: dict[str, dict[str, np.ndarray]] = {}
    treatment_full = df["treatment"].to_numpy()

    for outcome in outcomes:
        cols = [f"{outcome}{i}" for i in range(1, 37)]
        arr = df[cols].to_numpy(dtype=float)  # shape (n, 36)
        y_cm36 = df[[f"{outcome}_cm36"]].to_numpy(dtype=float).reshape(-1)
        outcome_data[outcome] = {"arr": arr, "y_cm36": y_cm36}

        results[outcome] = {"estimate": np.zeros((36, reps)), "bias": np.zeros((36, reps))}

    for rep in range(reps):
        # sample indices with replacement
        idx = rng.integers(0, n, size=n)
        sample_treatment = treatment_full[idx]
        var_treatment = np.var(sample_treatment, ddof=1)

        for outcome in outcomes:
            arr = outcome_data[outcome]["arr"][idx, :]
            y_cm36_sample = outcome_data[outcome]["y_cm36"][idx]

            for q in range(1, 37):
                X_q = arr[:, :q]

                # Fit surrogate model on treated sample
                treated_mask = sample_treatment == 1
                if treated_mask.sum() == 0:
                    results[outcome]["estimate"][q - 1, rep] = np.nan
                    results[outcome]["bias"][q - 1, rep] = np.nan
                    continue

                beta_s, _ = _ols_params_and_r2(y_cm36_sample[treated_mask], X_q[treated_mask])

                # Predict on full sample and regress predicted on treatment
                if X_q.ndim == 1 or X_q.shape[1] == 1:
                    X_design = np.column_stack((np.ones(X_q.shape[0]), X_q.reshape(-1)))
                else:
                    X_design = np.column_stack((np.ones(X_q.shape[0]), X_q))
                preds = X_design @ beta_s

                beta_pred, _ = _ols_params_and_r2(preds, sample_treatment)
                results[outcome]["estimate"][q - 1, rep] = float(beta_pred[1])

                _, r2_treatment = _ols_params_and_r2(sample_treatment, X_q)
                _, r2_outcome = _ols_params_and_r2(y_cm36_sample, X_q)
                var_outcome = np.var(y_cm36_sample, ddof=1)
                bias_base = var_outcome * (1 - r2_treatment) * (1 - r2_outcome) / var_treatment if var_treatment != 0 else np.nan
                results[outcome]["bias"][q - 1, rep] = np.sqrt(bias_base * 0.01) if bias_base >= 0 else 0.0

    return results


def compute_point_estimates(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    results = pd.DataFrame({"quarter": np.arange(1, 37)})
    var_treatment = df["treatment"].var(ddof=1)

    n = len(df)
    treatment_full = df["treatment"].to_numpy()

    for outcome in outcomes:
        cols = [f"{outcome}{i}" for i in range(1, 37)]
        arr = df[cols].to_numpy(dtype=float)
        y_cm36 = df[[f"{outcome}_cm36"]].to_numpy(dtype=float).reshape(-1)

        for q in range(1, 37):
            X_q = arr[:, :q]

            # Fit surrogate on treated observations only
            treated_mask = treatment_full == 1
            if treated_mask.sum() == 0:
                results.loc[q - 1, f"estimate_{outcome}"] = np.nan
                results.loc[q - 1, f"bias_01_{outcome}"] = np.nan
                continue

            beta_s, _ = _ols_params_and_r2(y_cm36[treated_mask], X_q[treated_mask])

            # predict on full sample
            if X_q.ndim == 1 or X_q.shape[1] == 1:
                X_design = np.column_stack((np.ones(X_q.shape[0]), X_q.reshape(-1)))
            else:
                X_design = np.column_stack((np.ones(X_q.shape[0]), X_q))
            preds = X_design @ beta_s

            beta_pred, _ = _ols_params_and_r2(preds, treatment_full)
            results.loc[q - 1, f"estimate_{outcome}"] = float(beta_pred[1])

            _, r2_treatment = _ols_params_and_r2(treatment_full, X_q)
            _, r2_outcome = _ols_params_and_r2(y_cm36, X_q)
            var_outcome = np.var(y_cm36, ddof=1)
            bias_base = var_outcome * (1 - r2_treatment) * (1 - r2_outcome) / var_treatment if var_treatment != 0 else np.nan
            results.loc[q - 1, f"bias_01_{outcome}"] = np.sqrt(bias_base * 0.01) if bias_base >= 0 else 0.0

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for bounds on bias.")
    parser.add_argument("--data-type", default=RunConfig().data_type)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    paths = load_paths()
    data_type = args.data_type

    riverside_path = paths.data_raw / f"{data_type} Riverside GAIN data.dta"
    df = read_stata(riverside_path)
    outcomes = available_outcomes(df, data_type)

    for outcome in outcomes:
        df[f"{outcome}_cm36"] = df[outcome_columns(outcome, range(1, 37))].sum(axis=1) / 36

    bootstrap = bootstrap_bounds(df, outcomes, args.reps, args.seed)

    ci_rows = []
    for outcome in outcomes:
        upper = bootstrap[outcome]["estimate"] + bootstrap[outcome]["bias"]
        lower = bootstrap[outcome]["estimate"] - bootstrap[outcome]["bias"]
        top_ci_se = upper.std(axis=1, ddof=1)
        bottom_ci_se = lower.std(axis=1, ddof=1)

        point_estimates = compute_point_estimates(df, [outcome])
        for idx in range(36):
            estimate = point_estimates.loc[idx, f"estimate_{outcome}"]
            bias = point_estimates.loc[idx, f"bias_01_{outcome}"]
            ci_rows.append(
                {
                    "quarter": idx + 1,
                    f"top_ci_01_{outcome}": estimate + bias + 1.645 * top_ci_se[idx],
                    f"bottom_ci_01_{outcome}": estimate - bias - 1.645 * bottom_ci_se[idx],
                    f"top_ci_{outcome}_se": top_ci_se[idx],
                    f"bottom_ci_{outcome}_se": bottom_ci_se[idx],
                }
            )

    output = pd.DataFrame(ci_rows)
    output = output.groupby("quarter", as_index=False).first()

    output_path = paths.data_derived / "CI on Bounds for Degree of Bias.csv"
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
