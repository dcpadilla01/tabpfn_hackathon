"""Primary metric MAE; secondary diagnostic R²."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2(y_true, y_pred) -> float:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
