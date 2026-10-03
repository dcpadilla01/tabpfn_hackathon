"""Model backends behind the fixed evaluator. The agent never touches these.

`fit_predict(backend, X_train, y_train, X_eval, cat_cols)` returns (point_predictions, meta).
Point predictions are loss-consistent with MAE: TabPFN's predictive median,
XGBoost with an L1 objective.

Categorical columns arrive from the evaluator as integer codes (float, NaN = missing or
unseen in train). TabPFN gets them flagged via `categorical_features_indices`; XGBoost
gets them as `category` dtype with `enable_categorical=True`. Same encoding, every
experiment, both backends — harness plumbing the agent never touches.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import load_config, tabpfn_token

_api_ready = False


def _init_api() -> None:
    global _api_ready
    if not _api_ready:
        import tabpfn_client

        tabpfn_client.set_access_token(tabpfn_token())
        _api_ready = True


def make_tabpfn(backend: str | None = None, cat_idx: list[int] | None = None):
    cfg = load_config()["tabpfn"]
    backend = backend or cfg["backend"]
    if backend == "api":
        _init_api()
        from tabpfn_client import TabPFNRegressor

        return TabPFNRegressor(
            model_path=cfg["model_path"], random_state=cfg["random_state"],
            categorical_features_indices=cat_idx or [],
        )
    if backend == "local":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(random_state=cfg["random_state"], categorical_features_indices=cat_idx or [])
    raise ValueError(f"unknown tabpfn backend {backend!r}")


def fit_predict_tabpfn(X_train, y_train, X_eval, cat_cols=(), backend: str | None = None):
    cfg = load_config()["tabpfn"]
    backend = backend or cfg["backend"]
    model = make_tabpfn(backend, [list(X_train.columns).index(c) for c in cat_cols])
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_eval, output_type=cfg["output_type"]), dtype=float)
    meta = {
        "backend": f"tabpfn-{backend}",
        "fit_predict_seconds": time.perf_counter() - t0,
        "random_state": cfg["random_state"],
    }
    if backend == "api":
        meta["model_path"] = cfg["model_path"]
        meta["api_meta"] = _jsonable(getattr(model, "_last_meta", {}) or {})
        timings = getattr(model, "get_timings", None)
        meta["api_timings"] = _jsonable(timings()) if timings else None
    return preds, meta


def fit_predict_xgb(X_train, y_train, X_eval, cat_cols=()):
    import xgboost as xgb

    params = dict(load_config()["xgb"])
    X_train, X_eval = X_train.copy(), X_eval.copy()
    for c in cat_cols:  # integer codes → category dtype over the train vocabulary
        cats = sorted(int(v) for v in X_train[c].dropna().unique())  # xgboost wants int categories
        X_train[c] = pd.Categorical(X_train[c].astype("Int64"), categories=cats)
        X_eval[c] = pd.Categorical(X_eval[c].astype("Int64"), categories=cats)
    model = xgb.XGBRegressor(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)  # no eval_set, no early stopping — by design
    preds = model.predict(X_eval).astype(float)
    return preds, {
        "backend": "xgb",
        "fit_predict_seconds": time.perf_counter() - t0,
        "random_state": params["random_state"],
        "xgb_params": params,
    }


def fit_predict(backend: str, X_train, y_train, X_eval, cat_cols=()):
    if backend == "tabpfn":
        return fit_predict_tabpfn(X_train, y_train, X_eval, cat_cols)
    if backend == "xgb":
        return fit_predict_xgb(X_train, y_train, X_eval, cat_cols)
    raise ValueError(f"unknown backend {backend!r}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
