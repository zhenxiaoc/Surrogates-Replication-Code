from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_stata(path: Path) -> pd.DataFrame:
    return pd.read_stata(path)


def outcome_columns(outcome: str, quarters: Iterable[int]) -> list[str]:
    return [f"{outcome}{q}" for q in quarters]


def available_outcomes(df: pd.DataFrame, data_type: str) -> list[str]:
    if data_type == "real":
        return ["emp", "earn"]
    if any(col.startswith("earn") for col in df.columns):
        return ["emp", "earn"]
    return ["emp"]
