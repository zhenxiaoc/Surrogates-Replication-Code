from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import RunConfig, load_paths
from data import available_outcomes, outcome_columns, read_stata
from regression import fit_ols, predict_ols


def bootstrap_bounds(df: pd.DataFrame, outcomes: list[str], reps: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, np.ndarray]] = {}

    for outcome in outcomes:
        results[outcome] = {
            "estimate": np.zeros((36, reps)),
            "bias": np.zeros((36, reps)),
        }

    for rep in range(reps):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        var_treatment = sample["treatment"].var(ddof=1)

        for outcome in outcomes:
            for q in range(1, 37):
                outcome_q_cols = outcome_columns(outcome, range(1, q + 1))
                treated = sample[sample["treatment"] == 1]
                model_surrogate = fit_ols(treated, f"{outcome}_cm36", outcome_q_cols)
                sample[f"{outcome}_cm_pred{q}"] = predict_ols(sample, outcome_q_cols, model_surrogate.params)
                model_surrogate_te = fit_ols(sample, f"{outcome}_cm_pred{q}", ["treatment"])
                results[outcome]["estimate"][q - 1, rep] = model_surrogate_te.coef("treatment")

                model_treatment = fit_ols(sample, "treatment", outcome_q_cols)
                model_outcome = fit_ols(sample, f"{outcome}_cm36", outcome_q_cols)
                var_outcome = sample[f"{outcome}_cm36"].var(ddof=1)
                bias_base = var_outcome * (1 - model_treatment.r2) * (1 - model_outcome.r2) / var_treatment
                results[outcome]["bias"][q - 1, rep] = np.sqrt(bias_base * 0.01)

    return results


def compute_point_estimates(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    results = pd.DataFrame({"quarter": np.arange(1, 37)})
    var_treatment = df["treatment"].var(ddof=1)

    for outcome in outcomes:
        for q in range(1, 37):
            outcome_q_cols = outcome_columns(outcome, range(1, q + 1))
            treated = df[df["treatment"] == 1]
            model_surrogate = fit_ols(treated, f"{outcome}_cm36", outcome_q_cols)
            df[f"{outcome}_cm_pred{q}"] = predict_ols(df, outcome_q_cols, model_surrogate.params)
            model_surrogate_te = fit_ols(df, f"{outcome}_cm_pred{q}", ["treatment"])
            results.loc[q - 1, f"estimate_{outcome}"] = model_surrogate_te.coef("treatment")

            model_treatment = fit_ols(df, "treatment", outcome_q_cols)
            model_outcome = fit_ols(df, f"{outcome}_cm36", outcome_q_cols)
            var_outcome = df[f"{outcome}_cm36"].var(ddof=1)
            bias_base = var_outcome * (1 - model_treatment.r2) * (1 - model_outcome.r2) / var_treatment
            results.loc[q - 1, f"bias_01_{outcome}"] = np.sqrt(bias_base * 0.01)

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
