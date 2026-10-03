"""Immutable evaluator. Owns keys, target, split, schema validation, backend execution,
loss-consistent point prediction, metric and logging. Callers pass a feature table and
nothing else: they cannot alter the split, target, metric or backend configuration.

    evaluate(feature_table, backend="tabpfn"|"xgb") -> {"mae", "r2", "n_features", "runtime_seconds", "backend", ...}

A feature table has `household_key`, `snapshot_day` and one or more feature columns,
and covers exactly the canonical train ∪ validation keys. Labels never leave this module;
only aggregate metrics do.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import load_config, package_versions
from src.data.schema import HOUSEHOLD_KEY
from src.data.targets import KEYS, TARGET, load_targets
from src.evaluation.backends import fit_predict
from src.evaluation.logger import ExperimentLog, ExperimentRecord
from src.evaluation.metrics import mae, r2

FORBIDDEN_COLUMNS = {TARGET, "split"}
MAX_FEATURES = 500


class FeatureTableError(ValueError):
    """The feature table violates the evaluator's contract. Message is safe to show the agent."""


def validate_feature_table(ft: pd.DataFrame, splits: tuple[str, ...] = ("train", "validation")) -> tuple[list[str], list[str]]:
    """Check the contract; return (feature_columns, categorical_columns). Never reveals labels."""
    if not isinstance(ft, pd.DataFrame):
        raise FeatureTableError(f"feature table must be a pandas DataFrame, got {type(ft).__name__}")
    if not all(isinstance(c, str) for c in ft.columns):
        raise FeatureTableError("all column names must be strings")
    if ft.columns.has_duplicates:
        raise FeatureTableError(f"duplicate column names: {sorted(ft.columns[ft.columns.duplicated()].unique())}")
    missing = [k for k in KEYS if k not in ft.columns]
    if missing:
        raise FeatureTableError(f"missing key columns {missing}; required: {KEYS}")
    bad = sorted(FORBIDDEN_COLUMNS & set(ft.columns))
    if bad:
        raise FeatureTableError(f"forbidden columns {bad}")
    if ft.duplicated(KEYS).any():
        raise FeatureTableError(f"{int(ft.duplicated(KEYS).sum())} duplicate (household_key, snapshot_day) rows")

    t = load_targets()
    expected = t.loc[t["split"].isin(splits), KEYS]
    got = ft[KEYS].astype({HOUSEHOLD_KEY: "int64", "snapshot_day": "int64"})
    merged = expected.astype("int64").merge(got, on=KEYS, how="outer", indicator=True)
    n_missing = int((merged["_merge"] == "left_only").sum())
    n_extra = int((merged["_merge"] == "right_only").sum())
    if n_missing or n_extra:
        raise FeatureTableError(
            f"key set must equal the canonical {'+'.join(splits)} keys ({len(expected):,} rows): "
            f"{n_missing:,} missing, {n_extra:,} unexpected. Use build_features() to get the keys right."
        )

    features = [c for c in ft.columns if c not in KEYS]
    if not features:
        raise FeatureTableError("no feature columns")
    if len(features) > MAX_FEATURES:
        raise FeatureTableError(f"{len(features)} features > limit {MAX_FEATURES}")
    cats = []
    for c in features:
        s = ft[c]
        if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            cats.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
                raise FeatureTableError(f"column {c!r} contains inf; replace with NaN or a finite value")
        else:
            raise FeatureTableError(f"column {c!r} has unsupported dtype {s.dtype}; use numeric, bool, string or category")
    return features, cats


def _encode(train: pd.DataFrame, other: pd.DataFrame, features: list[str], cats: list[str]):
    """Numeric → float64. Categorical → integer codes over the sorted train vocabulary
    (NaN for missing or unseen). Fixed for every experiment and both backends."""
    Xtr = pd.DataFrame(index=train.index)
    Xev = pd.DataFrame(index=other.index)
    for c in features:
        if c in cats:
            vocab = {v: i for i, v in enumerate(sorted(train[c].dropna().astype(str).unique()))}
            Xtr[c] = train[c].astype("string").map(vocab).astype("float64")
            Xev[c] = other[c].astype("string").map(vocab).astype("float64")
        else:
            Xtr[c] = train[c].astype("float64")
            Xev[c] = other[c].astype("float64")
    return Xtr, Xev


def feature_table_hash(ft: pd.DataFrame) -> str:
    ordered = ft.sort_values(KEYS).reset_index(drop=True)
    ordered = ordered[KEYS + sorted(c for c in ft.columns if c not in KEYS)]
    return hashlib.md5(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def _fit_and_score(ft, backend, fit_splits, eval_split):
    features, cats = validate_feature_table(ft, splits=tuple(fit_splits) + (eval_split,))
    t = load_targets()
    data = t.merge(ft.astype({HOUSEHOLD_KEY: "int32", "snapshot_day": "int16"}), on=KEYS, how="inner")
    train = data[data["split"].isin(fit_splits)].sort_values(KEYS)
    ev = data[data["split"] == eval_split].sort_values(KEYS)
    Xtr, Xev = _encode(train, ev, features, cats)
    t0 = time.perf_counter()
    preds, meta = fit_predict(backend, Xtr, train[TARGET].to_numpy(), Xev, cats)
    runtime = time.perf_counter() - t0
    y = ev[TARGET].to_numpy()
    predictions = ev[KEYS].assign(prediction=preds).reset_index(drop=True)
    return {
        "_predictions": predictions,
        "mae": round(mae(y, preds), 4),
        "r2": round(r2(y, preds), 4),
        "n_features": len(features),
        "runtime_seconds": round(runtime, 2),
        "backend": backend,
        "n_train": len(train),
        "n_eval": len(ev),
        "eval_split": eval_split,
        "_features": features,
        "_categoricals": cats,
        "_meta": meta,
    }


def evaluate(
    feature_table: pd.DataFrame,
    backend: str = "tabpfn",
    *,
    log: ExperimentLog | None = None,
    hypothesis: str = "",
    transformation_description: str = "",
    parent_id: str | None = None,
    experiment_id: str | None = None,
    extra: dict | None = None,
    save_dir=None,
    invalid_reason: str | None = None,
) -> dict:
    """Fit on train, score on validation. Logs one record if `log` is given.
    Contract violations come back as status="invalid" with a safe error message.
    With `save_dir`, writes predictions.parquet (keys + prediction, no labels) for analysis.
    `invalid_reason` logs an invalid experiment without fitting (caller-side contract failure).
    `extra["wall_clock_seconds"]`, if given, is time spent before this call; evaluation time is added."""
    started = time.perf_counter()
    extra = dict(extra or {})
    prior_seconds = extra.pop("wall_clock_seconds", 0.0) or 0.0
    seed = log.seed if log else load_config()["seed"]  # run (agent) seed; model random_state is in backend_meta
    rec = ExperimentRecord(
        experiment_id=experiment_id or (log.next_id() if log else "E???"),
        parent_id=parent_id,
        arm=log.arm if log else "adhoc",
        backend=backend,
        seed=seed,
        hypothesis=hypothesis,
        transformation_description=transformation_description,
        feature_columns=[c for c in getattr(feature_table, "columns", []) if c not in KEYS],
        versions=package_versions(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    try:
        if invalid_reason:
            raise FeatureTableError(invalid_reason)
        res = _fit_and_score(feature_table, backend, ("train",), "validation")
        rec.mae, rec.r2, rec.n_features = res["mae"], res["r2"], res["n_features"]
        rec.runtime_seconds, rec.n_train, rec.n_eval = res["runtime_seconds"], res["n_train"], res["n_eval"]
        rec.backend_meta = {"categorical_columns": res["_categoricals"], **res["_meta"]}
        rec.feature_table_hash = feature_table_hash(feature_table)
        rec.tabpfn_estimated_credits = res["_meta"].get("estimated_credits")
        if save_dir is not None:
            res["_predictions"].to_parquet(Path(save_dir) / "predictions.parquet", index=False)
    except FeatureTableError as e:
        rec.status, rec.error = "invalid", str(e)
    except Exception as e:  # backend/API failure: logged, not swallowed silently
        first_line = str(e).strip().splitlines()[0] if str(e).strip() else ""
        rec.status, rec.error = "error", f"{type(e).__name__}: {first_line}"[:500]
    rec.wall_clock_seconds = round(prior_seconds + time.perf_counter() - started, 2)
    for k, v in extra.items():
        if not hasattr(rec, k):
            raise AttributeError(f"unknown experiment record field {k!r}")
        setattr(rec, k, v)
    if log:
        log.append(rec)
    return {
        "experiment_id": rec.experiment_id,
        "status": rec.status,
        "error": rec.error,
        "mae": rec.mae,
        "r2": rec.r2,
        "n_features": rec.n_features,
        "runtime_seconds": rec.runtime_seconds,
        "backend": backend,
    }


def evaluate_frozen_test(feature_table: pd.DataFrame, backend: str) -> dict:
    """Phase 12 only: fit on train+validation, score once on test. Not exposed to agents."""
    res = _fit_and_score(feature_table, backend, ("train", "validation"), "test")
    return {k: v for k, v in res.items() if not k.startswith("_")} | {"meta": res["_meta"]}
