from __future__ import annotations

import argparse

import pandas as pd

from config import RunConfig, load_paths
from data import available_outcomes


def format_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.3f}"


def build_appendix_table_1(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = "Quarter " + df["quarter"].astype(int).astype(str)

    constants = {}
    surrogates = {}
    for outcome in outcomes:
        for q in [6, 12]:
            constants[(outcome, q)] = {
                "b": df.loc[df["quarter"] == q, f"{outcome}_constant"].mean(),
                "se": df.loc[df["quarter"] == q, f"{outcome}_constant_se"].mean(),
            }
            surrogates[(outcome, q)] = {
                "b": df.loc[df["quarter"] == q, f"surrogate_index_{outcome}"].mean(),
                "se": df.loc[df["quarter"] == q, f"surrogate_index_se_{outcome}"].mean(),
            }

    rows = []
    for quarter in range(1, 16):
        row_label = f"Quarter {quarter}"
        row = {"label": row_label}
        row_se = {"label": ""}
        for outcome in outcomes:
            for q in [6, 12]:
                row[f"{outcome}_weight_{q}"] = df.loc[df["quarter"] == quarter, f"{outcome}_weight_{q}_b"].iloc[0]
                row_se[f"{outcome}_weight_{q}"] = df.loc[df["quarter"] == quarter, f"{outcome}_weight_{q}_se"].iloc[0]
        rows.append(row)
        rows.append(row_se)

    rows.append({"label": "Constant", **{f"{outcome}_weight_{q}": constants[(outcome, q)]["b"] for outcome in outcomes for q in [6, 12]}})
    rows.append({"label": "", **{f"{outcome}_weight_{q}": constants[(outcome, q)]["se"] for outcome in outcomes for q in [6, 12]}})
    rows.append({"label": "Estimated Treatment Effect", **{f"{outcome}_weight_{q}": surrogates[(outcome, q)]["b"] for outcome in outcomes for q in [6, 12]}})
    rows.append({"label": "", **{f"{outcome}_weight_{q}": surrogates[(outcome, q)]["se"] for outcome in outcomes for q in [6, 12]}})

    table = pd.DataFrame(rows)

    for outcome in outcomes:
        for q in [6, 12]:
            col = f"{outcome}_weight_{q}"
            table[col] = table[col].apply(format_number)
            table.loc[table["label"].eq(""), col] = "(" + table.loc[table["label"].eq(""), col] + ")"
            table[col] = table[col].replace({"(.)": "", ".": ""})

    table = table.rename(
        columns={
            "emp_weight_6": "emp_weight_6",
            "emp_weight_12": "emp_weight_12",
            "earn_weight_6": "earn_weight_6",
            "earn_weight_12": "earn_weight_12",
        }
    )
    return table


def build_appendix_table_2(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for quarter in range(1, 37):
        row_label = str(quarter)
        row = {"quarter": row_label}
        row_se = {"quarter": ""}
        for outcome in outcomes:
            row[outcome] = df.loc[df["quarter"] == quarter, f"surrogate_index_{outcome}"].iloc[0]
            row_se[outcome] = df.loc[df["quarter"] == quarter, f"surrogate_index_se_{outcome}"].iloc[0]
        rows.append(row)
        rows.append(row_se)

    table = pd.DataFrame(rows)
    for outcome in outcomes:
        table[outcome] = table[outcome].apply(format_number)
        table.loc[table["quarter"].eq(""), outcome] = "(" + table.loc[table["quarter"].eq(""), outcome] + ")"
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Format appendix tables.")
    parser.add_argument("--data-type", default=RunConfig().data_type)
    args = parser.parse_args()

    paths = load_paths()
    data_type = args.data_type

    df = pd.read_csv(paths.data_derived / "Unformatted Appendix Tables Output.csv")
    outcomes = available_outcomes(df, data_type)

    table_1 = build_appendix_table_1(df, outcomes)
    table_2 = build_appendix_table_2(df, outcomes)

    output_path = paths.output / "Formatted Appendix Tables.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        table_1.to_excel(writer, sheet_name="Appendix Table 1 (RAW)", index=False)
        table_2.to_excel(writer, sheet_name="Appendix Table 2 (RAW)", index=False)


if __name__ == "__main__":
    main()
