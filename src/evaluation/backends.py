"""Model backends behind the fixed evaluator. The agent never touches these.

`fit_predict(backend, X_train, y_train, X_eval)` returns (point_predictions, meta).
Point predictions are loss-consistent with MAE: TabPFN's predictive median,
XGBoost with an L1 objective.
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


def make_tabpfn(backend: str | None = None):
    cfg = load_config()["tabpfn"]
    backend = backend or cfg["backend"]
    if backend == "api":
        _init_api()
        from tabpfn_client import TabPFNRegressor

        return TabPFNRegressor(model_path=cfg["model_path"], random_state=cfg["random_state"])
    if backend == "local":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(random_state=cfg["random_state"])
    raise ValueError(f"unknown tabpfn backend {backend!r}")


def fit_predict_tabpfn(X_train, y_train, X_eval, backend: str | None = None):
    cfg = load_config()["tabpfn"]
    backend = backend or cfg["backend"]
    model = make_tabpfn(backend)
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


def fit_predict_xgb(X_train, y_train, X_eval):
    import xgboost as xgb

    params = dict(load_config()["xgb"])
    X_train, X_eval = _xgb_categoricals(X_train, X_eval)
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


def fit_predict(backend: str, X_train, y_train, X_eval):
    if backend == "tabpfn":
        return fit_predict_tabpfn(X_train, y_train, X_eval)
    if backend == "xgb":
        return fit_predict_xgb(X_train, y_train, X_eval)
    raise ValueError(f"unknown backend {backend!r}")


def _xgb_categoricals(X_train: pd.DataFrame, X_eval: pd.DataFrame):
    """Fixed harness encoding: object/string/bool columns → `category` dtype with the
    train vocabulary; categories unseen in train become missing in eval."""
    X_train, X_eval = X_train.copy(), X_eval.copy()
    for c in X_train.columns:
        if X_train[c].dtype == object or str(X_train[c].dtype) in ("string", "category", "bool"):
            cats = pd.Index(X_train[c].dropna().astype(str).unique()).sort_values()
            dtype = pd.CategoricalDtype(cats)
            X_train[c] = X_train[c].astype("string").astype(dtype)
            X_eval[c] = X_eval[c].astype("string").astype(dtype)
    return X_train, X_eval


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
