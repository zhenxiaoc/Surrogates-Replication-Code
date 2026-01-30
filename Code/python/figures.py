from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import RunConfig, load_paths
from data import available_outcomes, outcome_columns, read_stata

BLUE = (0 / 255, 115 / 255, 162 / 255)
TEAL = (31 / 255, 143 / 255, 141 / 255)
GREEN = (137 / 255, 199 / 255, 103 / 255)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def figure_2(data_path: Path, output: Path, extension: str) -> None:
    df = read_stata(data_path)
    quarters = range(1, 37)
    control_means = [df.loc[df["treatment"] == 0, f"emp{q}"].mean() for q in quarters]
    treatment_means = [df.loc[df["treatment"] == 1, f"emp{q}"].mean() for q in quarters]

    control_means = [m * 100 for m in control_means]
    treatment_means = [m * 100 for m in treatment_means]
    control_full = sum(control_means) / len(control_means)
    treatment_full = sum(treatment_means) / len(treatment_means)

    plt.figure(figsize=(8, 5))
    plt.plot(quarters, treatment_means, color=BLUE, marker="o", markersize=3, label="Treatment")
    plt.plot(quarters, [treatment_full] * len(quarters), color=BLUE, label="Treatment Mean Over 9 Years")
    plt.plot(quarters, control_means, color=GREEN, marker="^", markersize=3, label="Control")
    plt.plot(quarters, [control_full] * len(quarters), color=GREEN, label="Control Mean Over 9 Years")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Employment Rate (%)")
    plt.yticks(range(10, 41, 10))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Figure 2.{extension}")


def appendix_figure_1(data_path: Path, output: Path, extension: str) -> None:
    df = read_stata(data_path)
    if not any(col.startswith("earn") for col in df.columns):
        return

    quarters = range(1, 37)
    control_means = [df.loc[df["treatment"] == 0, f"earn{q}"].mean() for q in quarters]
    treatment_means = [df.loc[df["treatment"] == 1, f"earn{q}"].mean() for q in quarters]
    control_full = sum(control_means) / len(control_means)
    treatment_full = sum(treatment_means) / len(treatment_means)

    plt.figure(figsize=(8, 5))
    plt.plot(quarters, treatment_means, color=BLUE, marker="o", markersize=3, label="Treatment")
    plt.plot(quarters, [treatment_full] * len(quarters), color=BLUE, label="Treatment Mean Over 9 Years")
    plt.plot(quarters, control_means, color=GREEN, marker="^", markersize=3, label="Control")
    plt.plot(quarters, [control_full] * len(quarters), color=GREEN, label="Control Mean Over 9 Years")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Mean Quarterly Earnings ($)")
    plt.yticks(range(0, 1501, 500))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 1.{extension}")


def figure_3a(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in ["experimental", "upper_experimental", "lower_experimental", "surrogate_index", "naive"]:
        df[col] = df[col] * 100

    plt.figure(figsize=(8, 5))
    plt.plot(df["quarter"], df["upper_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["lower_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treatment Effect Over 36 Quarters")
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Surrogate Index Estimate")
    plt.scatter(df["quarter"], df["naive"], color=GREEN, marker="^", label="Naive Short-Run Estimate")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate Over 9 Years (%)")
    plt.yticks(range(0, 13, 2))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Figure 3A.{extension}")


def figure_3b(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in ["experimental", "upper_experimental", "lower_experimental", "single_surrogate"]:
        df[col] = df[col] * 100

    plt.figure(figsize=(8, 5))
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treatment Effect Over 36 Quarters")
    plt.plot(df["quarter"], df["upper_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["lower_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.scatter(df["quarter"], df["single_surrogate"], color=BLUE, label="Surrogate Estimate Using Emp. Rate in Quarter x Only")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate Over 9 Years (%)")
    plt.yticks(range(0, 10, 2))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Figure 3B.{extension}")


def figure_4(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in ["experimental", "upper_experimental", "lower_experimental", "surrogate_index"]:
        df[col] = df[col] * 100

    plt.figure(figsize=(8, 5))
    plt.scatter(df["quarter"], df["experimental"], color=GREEN, marker="^", label="Actual Experimental Estimate")
    plt.vlines(df["quarter"], df["lower_experimental"], df["upper_experimental"], color=GREEN, linewidth=1)
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Six-Quarter Surrogate Index Estimate")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate to Quarter x (%)")
    plt.xticks(range(6, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Figure 4.{extension}")


def figure_5(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in ["experimental", "upper_bias_01", "lower_bias_01", "upper_bias_05", "lower_bias_05", "surrogate_index"]:
        df[col] = df[col] * 100

    plt.figure(figsize=(8, 5))
    plt.fill_between(df["quarter"], df["lower_bias_01"], df["upper_bias_01"], color="lightgray", label="Bounds on Bias")
    plt.fill_between(df["quarter"], df["upper_bias_01"], df["upper_bias_05"], color="dimgray", alpha=0.6)
    plt.fill_between(df["quarter"], df["lower_bias_05"], df["lower_bias_01"], color="dimgray", alpha=0.6)
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treat. Eff. Over 36 Quart.")
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Surrogate Index Estimate")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate Over 9 Years (%)")
    plt.yticks(range(-20, 21, 10))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Figure 5 (Raw).{extension}")


def appendix_figure_2a(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["quarter"], df["upper_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["lower_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treatment Effect Over 36 Quarters")
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Surrogate Index Estimate")
    plt.scatter(df["quarter"], df["naive"], color=GREEN, marker="^", label="Naive Short-Run Estimate")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nQuarterly Earnings Over 9 Years ($)")
    plt.yticks(range(0, 401, 100))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 2A.{extension}")


def appendix_figure_2b(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treatment Effect Over 36 Quarters")
    plt.plot(df["quarter"], df["upper_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.plot(df["quarter"], df["lower_experimental"], linestyle="--", linewidth=1, color=TEAL)
    plt.scatter(df["quarter"], df["single_surrogate"], color=BLUE, label="Surrogate Estimate Using Earnings in Quarter x Only")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nQuarterly Earnings Over 9 Years ($)")
    plt.yticks(range(0, 351, 100))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 2B.{extension}")


def appendix_figure_3(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.vlines(df["quarter"], df["lower_experimental"], df["upper_experimental"], color=GREEN, linewidth=1)
    plt.scatter(df["quarter"], df["experimental"], color=GREEN, marker="^", label="Actual Experimental Estimate")
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Six-Quarter Surrogate Index Estimate")
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nQuarterly Earnings to Quarter x ($)")
    plt.yticks(range(150, 416, 50))
    plt.xticks(range(6, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 3.{extension}")


def appendix_figure_4(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.fill_between(df["quarter"], df["lower_bias_01"], df["upper_bias_01"], color="lightgray", label="Bounds on Bias")
    plt.fill_between(df["quarter"], df["upper_bias_01"], df["upper_bias_05"], color="dimgray", alpha=0.6)
    plt.fill_between(df["quarter"], df["lower_bias_05"], df["lower_bias_01"], color="dimgray", alpha=0.6)
    plt.plot(df["quarter"], df["experimental"], color=TEAL, label="Actual Mean Treat. Eff. Over 36 Quart.")
    plt.scatter(df["quarter"], df["surrogate_index"], color=BLUE, label="Surrogate Index Estimate")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Quarters Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nQuarterly Earnings Over 9 Years ($)")
    plt.yticks(range(-1000, 1001, 500))
    plt.xticks(range(1, 37, 5))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 4 (Raw).{extension}")


def appendix_figure_5a(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in ["experimental", "upper_experimental", "lower_experimental", "surrogate_index"]:
        df[col] = df[col] * 100

    plt.figure(figsize=(8, 5))
    plt.vlines(df["year"], df["lower_experimental"], df["upper_experimental"], color=GREEN, linewidth=1)
    plt.scatter(df["year"], df["experimental"], color=GREEN, marker="^", label="Actual Experimental Estimate")
    plt.scatter(df["year"], df["surrogate_index"], color=BLUE, label="Six-Quarter Surrogate Index Estimate")
    plt.xlabel("Years Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nEmployment Rate at Year x (%)")
    plt.xticks(range(3, 10, 1))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 5A.{extension}")


def appendix_figure_5b(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.vlines(df["year"], df["lower_experimental"], df["upper_experimental"], color=GREEN, linewidth=1)
    plt.scatter(df["year"], df["experimental"], color=GREEN, marker="^", label="Actual Experimental Estimate")
    plt.scatter(df["year"], df["surrogate_index"], color=BLUE, label="Six-Quarter Surrogate Index Estimate")
    plt.xlabel("Years Since Random Assignment")
    plt.ylabel("Estimated Treatment Effect on Mean\nQuarterly Earnings at Year x ($)")
    plt.xticks(range(3, 10, 1))
    plt.legend(loc="upper left")
    save_fig(output / f"Appendix Figure 5B.{extension}")


def figure_6a(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if "_q6" in col:
            df[col] = df[col] * 100

    labels = {"rs": "Riverside", "la": "Los Angeles", "sd": "San Diego", "al": "Alameda"}
    plt.figure(figsize=(6, 6))
    for key, label in labels.items():
        plt.scatter(df[f"experimental_{key}_q6"], df[f"surrogate_index_{key}_q6"], color="navy")
        plt.annotate(label, (df[f"experimental_{key}_q6"].iloc[0], df[f"surrogate_index_{key}_q6"].iloc[0]), color="navy")
        plt.vlines(
            df[f"experimental_{key}_q6"],
            df[f"lower_experimental_{key}_q6"],
            df[f"upper_experimental_{key}_q6"],
            color="navy",
            linewidth=1,
        )
    plt.plot([-3, 8], [-3, 8], linestyle="--", color="gray")
    plt.text(-1.7, -2.7, "45° Line", color="gray")
    plt.xlabel("Actual Treatment Effect on\nMean Employment Rate (%) Over 36 Quarters")
    plt.ylabel("Six-Quarter Surrogate Index Estimate of\nTreatment Effect on Mean Employment Rate (%)")
    plt.xlim(-3, 7.5)
    plt.ylim(-3, 7.5)
    plt.xticks(range(-2, 9, 2))
    plt.yticks(range(-2, 9, 2))
    save_fig(output / f"Figure 6A.{extension}")


def figure_6b(csv_path: Path, output: Path, extension: str) -> None:
    df = pd.read_csv(csv_path)
    labels = {"rs": "Riverside", "la": "Los Angeles", "sd": "San Diego", "al": "Alameda"}
    plt.figure(figsize=(6, 6))
    for key, label in labels.items():
        plt.scatter(df[f"experimental_{key}_q6"], df[f"surrogate_index_{key}_q6"], color="navy")
        plt.annotate(label, (df[f"experimental_{key}_q6"].iloc[0], df[f"surrogate_index_{key}_q6"].iloc[0]), color="navy")
        plt.vlines(
            df[f"experimental_{key}_q6"],
            df[f"lower_experimental_{key}_q6"],
            df[f"upper_experimental_{key}_q6"],
            color="navy",
            linewidth=1,
        )
    plt.plot([-100, 380], [-100, 380], linestyle="--", color="gray")
    plt.text(-99, -50, "45° Line", color="gray")
    plt.xlabel("Actual Treatment Effect on\nMean Quarterly Earnings ($) Over 36 Quarters")
    plt.ylabel("Six-Quarter Surrogate Index Estimate of\nTreatment Effect on Mean Quarterly Earnings ($)")
    plt.xlim(-50, 350)
    plt.ylim(-50, 350)
    plt.xticks(range(-100, 351, 100))
    plt.yticks(range(-100, 351, 100))
    save_fig(output / f"Figure 6B.{extension}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures.")
    parser.add_argument("--data-type", default=RunConfig().data_type)
    parser.add_argument("--extension", default=RunConfig().extension)
    args = parser.parse_args()

    paths = load_paths()
    data_type = args.data_type
    extension = args.extension

    riverside_path = paths.data_raw / f"{data_type} Riverside GAIN data.dta"

    figure_2(riverside_path, paths.output, extension)
    appendix_figure_1(riverside_path, paths.output, extension)

    figure_3a(paths.data_derived / "Estimated Treatment Effect on Cumulative Employment (36 Quarters).csv", paths.output, extension)
    figure_3b(paths.data_derived / "Estimated Treatment Effect on Cumulative Employment (36 Quarters).csv", paths.output, extension)
    figure_4(paths.data_derived / "Estimated Treatment Effect on Cumulative Employment, Varying Outcome Horizon.csv", paths.output, extension)
    figure_5(paths.data_derived / "Estimated Bounds on Treatment Effect on Cumulative Employment (36 Quarters).csv", paths.output, extension)

    if (paths.data_derived / "Estimated Treatment Effect on Cumulative Earnings (36 Quarters).csv").exists():
        appendix_figure_2a(paths.data_derived / "Estimated Treatment Effect on Cumulative Earnings (36 Quarters).csv", paths.output, extension)
        appendix_figure_2b(paths.data_derived / "Estimated Treatment Effect on Cumulative Earnings (36 Quarters).csv", paths.output, extension)
        appendix_figure_3(paths.data_derived / "Estimated Treatment Effect on Cumulative Earnings, Varying Outcome Horizon.csv", paths.output, extension)
        appendix_figure_4(paths.data_derived / "Estimated Bounds on Treatment Effect on Cumulative Earnings (36 Quarters).csv", paths.output, extension)
        appendix_figure_5b(paths.data_derived / "Estimated Treatment Effect on Yearly Earnings.csv", paths.output, extension)

    appendix_figure_5a(paths.data_derived / "Estimated Treatment Effect on Yearly Employment.csv", paths.output, extension)

    if (paths.data_derived / "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Employment).csv").exists():
        figure_6a(
            paths.data_derived / "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Employment).csv",
            paths.output,
            extension,
        )
    if (paths.data_derived / "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Earnings).csv").exists():
        figure_6b(
            paths.data_derived / "Estimated Six-Quarter Surrogate Index vs Actual Treatment Effects for Other Sites (Earnings).csv",
            paths.output,
            extension,
        )


if __name__ == "__main__":
    main()
