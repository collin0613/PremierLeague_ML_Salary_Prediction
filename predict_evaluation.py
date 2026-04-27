from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


@dataclass(frozen=True)
class RegressionEval:
    n: int
    mae: float
    baseline_mae: float
    improvement_pct: float
    median_y: float
    iqr_y: float
    nmae_vs_median: float
    mae_vs_iqr: float

    def as_print_lines(self, label: str, indent: str = "  ") -> list[str]:
        def pct(x: float) -> str:
            if np.isnan(x):
                return "NA"
            return f"{100.0 * x:.1f}%"

        lines = [
            f"{indent}{label} (n={self.n:,})",
            f"{indent}  MAE: {self.mae:,.2f}",
            f"{indent}  Baseline MAE (median): {self.baseline_mae:,.2f}",
            f"{indent}  Improvement vs baseline: {pct(self.improvement_pct)}",
            f"{indent}  Salary median (ref): {self.median_y:,.0f}",
            f"{indent}  Salary IQR (ref): {self.iqr_y:,.0f}",
            f"{indent}  NMAE vs median: {pct(self.nmae_vs_median)}",
            f"{indent}  MAE / IQR: {pct(self.mae_vs_iqr)}",
        ]
        return lines


def _to_1d_array(x: Any) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        arr = x.squeeze(axis=1).to_numpy()
    else:
        arr = np.asarray(x)
    return arr.reshape(-1)


def _robust_ref_stats(y_ref: np.ndarray) -> tuple[float, float]:
    y_ref = y_ref[~np.isnan(y_ref)]
    if y_ref.size == 0:
        return float("nan"), float("nan")
    q25, q50, q75 = np.quantile(y_ref, [0.25, 0.50, 0.75])
    return float(q50), float(q75 - q25)


def evaluate_regression(
    *,
    y_true: Any,
    y_pred: Any,
    y_ref: Any,
) -> RegressionEval:
    """
    Evaluate regression predictions using:
      - MAE
      - Baseline MAE using constant predictor = median(y_ref)
      - Normalized MAE vs median(y_ref)
      - MAE as % of IQR(y_ref)

    y_ref should generally be the TRAIN distribution for the same slice.
    """
    yt = _to_1d_array(y_true).astype(float)
    yp = _to_1d_array(y_pred).astype(float)
    yr = _to_1d_array(y_ref).astype(float)

    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt = yt[mask]
    yp = yp[mask]

    n = int(yt.size)
    mae = float(mean_absolute_error(yt, yp)) if n else float("nan")

    med, iqr = _robust_ref_stats(yr)
    baseline_pred = np.full_like(yt, fill_value=med, dtype=float) if n else np.array([], dtype=float)
    baseline_mae = float(mean_absolute_error(yt, baseline_pred)) if n else float("nan")

    improvement_pct = (baseline_mae - mae) / baseline_mae if baseline_mae and not np.isnan(baseline_mae) else float("nan")
    nmae_vs_median = mae / med if med and not np.isnan(med) else float("nan")
    mae_vs_iqr = mae / iqr if iqr and not np.isnan(iqr) else float("nan")

    return RegressionEval(
        n=n,
        mae=mae,
        baseline_mae=baseline_mae,
        improvement_pct=float(improvement_pct),
        median_y=med,
        iqr_y=iqr,
        nmae_vs_median=float(nmae_vs_median),
        mae_vs_iqr=float(mae_vs_iqr),
    )


def evaluate_regression_by_group(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    groups_true: pd.Series,
    y_ref: pd.Series,
    groups_ref: pd.Series,
    min_n: int = 3,
) -> dict[str, RegressionEval]:
    """
    Evaluate regression by group with per-group reference stats computed on (y_ref, groups_ref).

    Example:
      - y_true = y_test
      - y_pred = model.predict(X_test)
      - groups_true = X_test["position"]
      - y_ref = y_train
      - groups_ref = X_train["position"]
    """
    out: dict[str, RegressionEval] = {}
    # align to y_true index
    groups_true = groups_true.loc[y_true.index]

    for g in sorted([x for x in groups_true.dropna().unique()]):
        mask_true = groups_true == g
        if int(mask_true.sum()) < min_n:
            continue

        # reference slice is training distribution for that group
        mask_ref = groups_ref == g
        y_ref_g = y_ref.loc[mask_ref]
        if y_ref_g.notna().sum() < 5:
            # too little reference data -> skip normalized stats for this group
            continue

        ev = evaluate_regression(
            y_true=y_true.loc[mask_true],
            y_pred=y_pred[mask_true.to_numpy()],
            y_ref=y_ref_g,
        )
        out[str(g)] = ev

    return out

