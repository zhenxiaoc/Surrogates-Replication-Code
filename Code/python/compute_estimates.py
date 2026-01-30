from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t

from config import RunConfig, load_paths
from data import available_outcomes, outcome_columns, read_stata
from regression import fit_ols, predict_ols


def cumulative_means(df: pd.DataFrame, outcomes: Iterable[str], max_quarter: int = 36) -> pd.DataFrame:
    for outcome in outcomes:
        for q in range(1, max_quarter + 1):
            cols = outcome_columns(outcome, range(1, q + 1))
            df[f"{outcome}_cm{q}"] = df[cols].sum(axis=1) / q
    return df


def add_estimate_columns(df: pd.DataFrame, outcomes: Iterable[str]) -> None:
    for estimate in ["experimental", "surrogate_index", "naive", "single_surrogate"]:
        for outcome in outcomes:
            df[f"{estimate}_{outcome}"] = np.nan
            df[f"{estimate}_se_{outcome}"] = np.nan


def _t_crit(n: int) -> float:
    return float(t.ppf(0.975, df=n - 2))


def compute_section_b(df: pd.DataFrame, outcomes: Iterable[str], n_obs: int) -> pd.DataFrame:
    results = pd.DataFrame({"quarter": np.arange(1, 37)})
    var_treatment = df["treatment"].var(ddof=1)
    t_crit = _t_crit(n_obs)

    for outcome in outcomes:
        for q in range(1, 37):
            outcome_q_cols = outcome_columns(outcome, range(1, q + 1))

            treated = df[df["treatment"] == 1]
            model_surrogate = fit_ols(treated, f"{outcome}_cm36", outcome_q_cols)

            for i, col in enumerate(outcome_q_cols, start=1):
                results.loc[i - 1, f"{outcome}_weight_{q}_b"] = model_surrogate.coef(col)
                results.loc[i - 1, f"{outcome}_weight_{q}_se"] = model_surrogate.se(col)

            results.loc[q - 1, f"{outcome}_constant"] = model_surrogate.coef("const")
            results.loc[q - 1, f"{outcome}_constant_se"] = model_surrogate.se("const")

            df[f"{outcome}_cm_pred{q}"] = predict_ols(df, outcome_q_cols, model_surrogate.params)
            model_surrogate_te = fit_ols(df, f"{outcome}_cm_pred{q}", ["treatment"])
            results.loc[q - 1, f"surrogate_index_{outcome}"] = model_surrogate_te.coef("treatment")
            results.loc[q - 1, f"surrogate_index_se_{outcome}"] = model_surrogate_te.se("treatment")

            model_treatment = fit_ols(df, "treatment", outcome_q_cols)
            model_outcome = fit_ols(df, f"{outcome}_cm36", outcome_q_cols)
            var_outcome = df[f"{outcome}_cm36"].var(ddof=1)
            bias_base = var_outcome * (1 - model_treatment.r2) * (1 - model_outcome.r2) / var_treatment
            results.loc[q - 1, f"bias_01_{outcome}"] = np.sqrt(bias_base * 0.01)
            results.loc[q - 1, f"bias_05_{outcome}"] = np.sqrt(bias_base * 0.05)

            model_single = fit_ols(treated, f"{outcome}_cm36", [f"{outcome}{q}"])
            df[f"{outcome}_cm_1_pred{q}"] = predict_ols(df, [f"{outcome}{q}"], model_single.params)
            model_single_te = fit_ols(df, f"{outcome}_cm_1_pred{q}", ["treatment"])
            results.loc[q - 1, f"single_surrogate_{outcome}"] = model_single_te.coef("treatment")
            results.loc[q - 1, f"single_surrogate_se_{outcome}"] = model_single_te.se("treatment")

            model_naive = fit_ols(df, f"{outcome}_cm{q}", ["treatment"])
            results.loc[q - 1, f"naive_{outcome}"] = model_naive.coef("treatment")
            results.loc[q - 1, f"naive_se_{outcome}"] = model_naive.se("treatment")

            model_experimental = fit_ols(df, f"{outcome}_cm36", ["treatment"])
            results.loc[q - 1, f"experimental_{outcome}"] = model_experimental.coef("treatment")
            results.loc[q - 1, f"experimental_se_{outcome}"] = model_experimental.se("treatment")

        for estimate in ["experimental", "surrogate_index", "naive"]:
            results[f"upper_{estimate}_{outcome}"] = (
                results[f"{estimate}_{outcome}"] + t_crit * results[f"{estimate}_se_{outcome}"]
            )
            results[f"lower_{estimate}_{outcome}"] = (
                results[f"{estimate}_{outcome}"] - t_crit * results[f"{estimate}_se_{outcome}"]
            )

    return results


def output_section_b(results: pd.DataFrame, outcomes: Iterable[str], n_obs: int, data_derived: str) -> None:
    t_crit = _t_crit(n_obs)
    for outcome in outcomes:
        columns = [
            "quarter",
            f"experimental_{outcome}",
            f"experimental_se_{outcome}",
            f"surrogate_index_{outcome}",
            f"surrogate_index_se_{outcome}",
            f"naive_{outcome}",
            f"naive_se_{outcome}",
        ]
        subset = results[columns].copy()
        for estimate in ["experimental", "surrogate_index", "naive"]:
            subset[f"upper_{estimate}"] = subset[estimate + f"_{outcome}"] + t_crit * subset[estimate + f"_se_{outcome}"]
            subset[f"lower_{estimate}"] = subset[estimate + f"_{outcome}"] - t_crit * subset[estimate + f"_se_{outcome}"]
        subset = subset.rename(columns=lambda c: c.replace(f"_{outcome}", ""))

        filename = (
            "Estimated Treatment Effect on Cumulative Employment (36 Quarters).csv"
            if outcome == "emp"
            else "Estimated Treatment Effect on Cumulative Earnings (36 Quarters).csv"
        )
        subset.to_csv(f"{data_derived}/{filename}", index=False)

        bounds_columns = [
            "quarter",
            "experimental",
            "surrogate_index",
            f"bias_01_{outcome}",
            f"bias_05_{outcome}",
        ]
        bounds = results[["quarter", f"experimental_{outcome}", f"surrogate_index_{outcome}", f"bias_01_{outcome}", f"bias_05_{outcome}"]].copy()
        bounds = bounds.rename(
            columns={
                f"experimental_{outcome}": "experimental",
                f"surrogate_index_{outcome}": "surrogate_index",
            }
        )
        bounds["upper_bias_01"] = bounds["surrogate_index"] + bounds[f"bias_01_{outcome}"]
        bounds["lower_bias_01"] = bounds["surrogate_index"] - bounds[f"bias_01_{outcome}"]
        bounds["upper_bias_05"] = bounds["surrogate_index"] + bounds[f"bias_05_{outcome}"]
        bounds["lower_bias_05"] = bounds["surrogate_index"] - bounds[f"bias_05_{outcome}"]
        bounds = bounds.drop(columns=[f"bias_01_{outcome}", f"bias_05_{outcome}"])

        filename = (
            "Estimated Bounds on Treatment Effect on Cumulative Employment (36 Quarters).csv"
            if outcome == "emp"
            else "Estimated Bounds on Treatment Effect on Cumulative Earnings (36 Quarters).csv"
        )
        bounds.to_csv(f"{data_derived}/{filename}", index=False)

    results.to_csv(f"{data_derived}/Unformatted Appendix Tables Output.csv", index=False)


def compute_section_c(df: pd.DataFrame, outcomes: Iterable[str], n_obs: int) -> pd.DataFrame:
    results = pd.DataFrame({"quarter": np.arange(1, 37)})
    for outcome in outcomes:
        for q in range(6, 37):
            treated = df[df["treatment"] == 1]
            model_surrogate = fit_ols(treated, f"{outcome}_cm{q}", outcome_columns(outcome, range(1, 7)))
            df[f"{outcome}_cm_pred{q}"] = predict_ols(df, outcome_columns(outcome, range(1, 7)), model_surrogate.params)
            model_surrogate_te = fit_ols(df, f"{outcome}_cm_pred{q}", ["treatment"])
            results.loc[q - 1, f"surrogate_index_{outcome}"] = model_surrogate_te.coef("treatment")
            results.loc[q - 1, f"surrogate_index_se_{outcome}"] = model_surrogate_te.se("treatment")

            model_experimental = fit_ols(df, f"{outcome}_cm{q}", ["treatment"])
            results.loc[q - 1, f"experimental_{outcome}"] = model_experimental.coef("treatment")
            results.loc[q - 1, f"experimental_se_{outcome}"] = model_experimental.se("treatment")

    return results


def output_section_c(results: pd.DataFrame, outcomes: Iterable[str], n_obs: int, data_derived: str) -> None:
    t_crit = _t_crit(n_obs)
    for outcome in outcomes:
        subset = results[["quarter", f"experimental_{outcome}", f"experimental_se_{outcome}", f"surrogate_index_{outcome}"]].copy()
        subset = subset[(subset["quarter"] >= 6) & (subset["quarter"] <= 36)]
        subset["upper_experimental"] = subset[f"experimental_{outcome}"] + t_crit * subset[f"experimental_se_{outcome}"]
        subset["lower_experimental"] = subset[f"experimental_{outcome}"] - t_crit * subset[f"experimental_se_{outcome}"]
        subset = subset.rename(columns=lambda c: c.replace(f"_{outcome}", ""))

        filename = (
            "Estimated Treatment Effect on Cumulative Employment, Varying Outcome Horizon.csv"
            if outcome == "emp"
            else "Estimated Treatment Effect on Cumulative Earnings, Varying Outcome Horizon.csv"
        )
        subset.to_csv(f"{data_derived}/{filename}", index=False)


def compute_section_d(df: pd.DataFrame, outcomes: Iterable[str], n_obs: int) -> pd.DataFrame:
    results = pd.DataFrame({"year": np.arange(1, 10)})

    for outcome in outcomes:
        for y in range(3, 10):
            year_start = 4 * y - 3
            year_end = 4 * y
            cols = outcome_columns(outcome, range(year_start, year_end + 1))
            df[f"{outcome}_annual_{y}"] = df[cols].sum(axis=1) / 4

        for y in range(3, 10):
            treated = df[df["treatment"] == 1]
            model_surrogate = fit_ols(treated, f"{outcome}_annual_{y}", outcome_columns(outcome, range(1, 7)))
            df[f"{outcome}_annual_pred{y}"] = predict_ols(df, outcome_columns(outcome, range(1, 7)), model_surrogate.params)
            model_surrogate_te = fit_ols(df, f"{outcome}_annual_pred{y}", ["treatment"])
            results.loc[y - 1, f"surrogate_index_{outcome}"] = model_surrogate_te.coef("treatment")

            model_experimental = fit_ols(df, f"{outcome}_annual_{y}", ["treatment"])
            results.loc[y - 1, f"experimental_{outcome}"] = model_experimental.coef("treatment")
            results.loc[y - 1, f"experimental_se_{outcome}"] = model_experimental.se("treatment")

    return results


def output_section_d(results: pd.DataFrame, outcomes: Iterable[str], n_obs: int, data_derived: str) -> None:
    t_crit = _t_crit(n_obs)
    for outcome in outcomes:
        subset = results[["year", f"experimental_{outcome}", f"experimental_se_{outcome}", f"surrogate_index_{outcome}"]].copy()
        subset = subset[(subset["year"] >= 3) & (subset["year"] <= 9)]
        subset["upper_experimental"] = subset[f"experimental_{outcome}"] + t_crit * subset[f"experimental_se_{outcome}"]
        subset["lower_experimental"] = subset[f"experimental_{outcome}"] - t_crit * subset[f"experimental_se_{outcome}"]
        subset = subset.rename(columns=lambda c: c.replace(f"_{outcome}", ""))

        filename = (
            "Estimated Treatment Effect on Yearly Employment.csv"
            if outcome == "emp"
            else "Estimated Treatment Effect on Yearly Earnings.csv"
        )
        subset.to_csv(f"{data_derived}/{filename}", index=False)


def compute_section_e(df: pd.DataFrame, outcomes: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    df["site"] = df["site"].replace(
        {
            "Los Angeles": "LA",
            "San Diego": "SD",
            "Riverside": "RS",
            "Alameda": "AL",
        }
    )
    sites = ["RS", "LA", "SD", "AL"]

    for outcome in outcomes:
        df[f"{outcome}_cm36"] = df[outcome_columns(outcome, range(1, 37))].sum(axis=1) / 36

    results = pd.DataFrame({"site": sites})
    for site in sites:
        results.loc[results["site"] == site, "site_n_obs"] = len(df[df["site"] == site])
    for outcome in outcomes:
        treated_rs = df[(df["treatment"] == 1) & (df["site"] == "RS")]
        model_surrogate = fit_ols(treated_rs, f"{outcome}_cm36", outcome_columns(outcome, range(1, 7)))
        df[f"{outcome}_cm_predQ6"] = predict_ols(df, outcome_columns(outcome, range(1, 7)), model_surrogate.params)

        for site in sites:
            site_df = df[df["site"] == site]
            model_surrogate_te = fit_ols(site_df, f"{outcome}_cm_predQ6", ["treatment"])
            results.loc[results["site"] == site, f"surrogate_index_{outcome}_{site}_Q6"] = model_surrogate_te.coef("treatment")

            model_experimental = fit_ols(site_df, f"{outcome}_cm36", ["treatment"])
            results.loc[results["site"] == site, f"experimental_{outcome}_{site}_Q6"] = model_experimental.coef("treatment")
            results.loc[results["site"] == site, f"experimental_{outcome}_{site}_Q6_se"] = model_experimental.se("treatment")

    return results


def output_section_e(results: pd.DataFrame, outcomes: Iterable[str], data_derived: str) -> None:
    for outcome in outcomes:
        subset = results[[col for col in results.columns if f"_{outcome}_" in col or col in {"site", "site_n_obs"}]].copy()
        subset.columns = [col.replace(f"_{outcome}_", "_") for col in subset.columns]
        subset = subset.set_index("site")

        for site in subset.index:
            n_obs = int(subset.loc[site, "site_n_obs"])
            t_crit = _t_crit(n_obs) if n_obs > 2 else 0
            upper = subset.loc[site, f"surrogate_index_{site}_Q6"] + t_crit * subset.loc[site, f"experimental_{site}_Q6_se"]
            lower = subset.loc[site, f"surrogate_index_{site}_Q6"] - t_crit * subset.loc[site, f"experimental_{site}_Q6_se"]
            subset.loc[site, f"upper_experimental_{site}_Q6"] = upper
            subset.loc[site, f"lower_experimental_{site}_Q6"] = lower

        subset = subset.reset_index().drop(columns=["site_n_obs"])
        filename = (
            "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Employment).csv"
            if outcome == "emp"
            else "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Earnings).csv"
        )
        subset.to_csv(f"{data_derived}/{filename}", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute surrogate index estimates.")
    parser.add_argument("--data-type", default=RunConfig().data_type)
    args = parser.parse_args()

    paths = load_paths()
    data_type = args.data_type

    riverside_path = paths.data_raw / f"{data_type} Riverside GAIN data.dta"
    riverside = read_stata(riverside_path)
    outcomes = available_outcomes(riverside, data_type)

    riverside = cumulative_means(riverside, outcomes)
    add_estimate_columns(riverside, outcomes)

    n_obs = len(riverside)
    section_b_results = compute_section_b(riverside, outcomes, n_obs)
    output_section_b(section_b_results, outcomes, n_obs, str(paths.data_derived))

    section_c_results = compute_section_c(riverside, outcomes, n_obs)
    output_section_c(section_c_results, outcomes, n_obs, str(paths.data_derived))

    section_d_results = compute_section_d(riverside, outcomes, n_obs)
    output_section_d(section_d_results, outcomes, n_obs, str(paths.data_derived))

    all_locations_path = paths.data_raw / f"{data_type} All Locations GAIN data.dta"
    if all_locations_path.exists():
        all_locations = read_stata(all_locations_path)
        outcomes_all = available_outcomes(all_locations, data_type)
        section_e_results = compute_section_e(all_locations, outcomes_all)
        output_section_e(section_e_results, outcomes_all, str(paths.data_derived))


if __name__ == "__main__":
    main()
