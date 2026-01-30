from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    r2: float

    def coef(self, name: str) -> float:
        return float(self.params.get(name, np.nan))

    def se(self, name: str) -> float:
        return float(self.bse.get(name, np.nan))


def fit_ols(df: pd.DataFrame, y: str, x: Iterable[str]) -> OLSResult:
    x_list = list(x)
    subset = df[[y] + x_list].dropna()
    y_values = subset[y]
    x_values = sm.add_constant(subset[x_list], has_constant="add")
    model = sm.OLS(y_values, x_values)
    results = model.fit()
    return OLSResult(params=results.params, bse=results.bse, r2=float(results.rsquared))


def predict_ols(df: pd.DataFrame, x: Iterable[str], params: pd.Series) -> pd.Series:
    x_list = list(x)
    x_values = sm.add_constant(df[x_list], has_constant="add")
    aligned = x_values.reindex(columns=params.index, fill_value=1.0)
    return aligned @ params
