"""The API available to agent code inside run_python (imported as `agent_api`, and
pre-imported into the namespace). It is the only route to data.

Isolation rules enforced here:
  * Direct views (`snapshot`, `history`) are capped at RESEARCH_VISIBLE_DAY — the first
    validation snapshot day — so no validation label is ever directly observable.
  * `build_features(fn)` runs fn once per snapshot day in a forked child process; only the
    returned DataFrame comes back. State stashed by fn (globals, files) dies with the child,
    so data seen at a later snapshot cannot leak into an earlier one.
"""

from __future__ import annotations

import os
import pickle
import traceback

import numpy as np  # noqa: F401  (re-exported for agent convenience)
import pandas as pd

from src.data import accessor as _acc
from src.data.load import load_table as _load_table
from src.data.splits import research_visible_day as _visible_day
from src.data.targets import KEYS, TARGET, load_targets as _load_targets

RESEARCH_VISIBLE_DAY = _visible_day()
_WORKSPACE = os.getcwd()
_IN_CHILD = False

from src.researcher.agent_api_public import PUBLIC_API as _PUBLIC

__all__ = [
    "RESEARCH_VISIBLE_DAY", "AsOf", "snapshot", "history", "build_features", "baseline_features",
    "train_targets", "snapshot_days", "save_table", "load_saved", "describe_tables", "KEYS", "TARGET", "pd", "np",
]

AsOf = _acc.AsOf
describe_tables = _acc.describe_tables


class AccessDenied(RuntimeError):
    pass


def _cap(day: int) -> int:
    if day > RESEARCH_VISIBLE_DAY:
        raise AccessDenied(
            f"as_of_day={day} is beyond the research horizon (day {RESEARCH_VISIBLE_DAY}). "
            "Data after that day is only reachable inside build_features(fn), per snapshot."
        )
    return int(day)


def snapshot(as_of_day: int = None) -> "_acc.AsOf":
    """As-of view of every table at `as_of_day` (default and max: RESEARCH_VISIBLE_DAY)."""
    return _acc.AsOf(_cap(RESEARCH_VISIBLE_DAY if as_of_day is None else as_of_day))


def history(household_key: int, as_of_day: int = None) -> pd.DataFrame:
    """One household's line items with day <= as_of_day (max RESEARCH_VISIBLE_DAY)."""
    return _acc.history(household_key, _cap(RESEARCH_VISIBLE_DAY if as_of_day is None else as_of_day))


def snapshot_days() -> dict[str, list[int]]:
    """Snapshot days per split visible to research (train, validation)."""
    k = _acc.target_keys()
    return {s: sorted(k.loc[k["split"] == s, "snapshot_day"].unique().tolist()) for s in ("train", "validation")}


def train_targets() -> pd.DataFrame:
    """Labels for TRAIN rows only: household_key, snapshot_day, future_spend_4w."""
    t = _load_targets()
    return t.loc[t["split"] == "train", KEYS + [TARGET]].reset_index(drop=True)


def _preload() -> None:
    """Load every table once in the parent so forked children share it copy-on-write."""
    for name in _acc.TABLES:
        if name == "causal_data":
            _acc._causal()
        else:
            _load_table(name)


def _run_isolated(fn, day: int, households: pd.Index):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        global _IN_CHILD
        _IN_CHILD = True
        os.close(r)
        try:
            payload = ("ok", fn(_acc.AsOf(day, households=households), day))
        except BaseException as e:  # noqa: BLE001
            payload = ("err", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}")
        try:
            data = pickle.dumps(payload)
        except Exception as e:  # unpicklable return value
            data = pickle.dumps(("err", f"fn must return a pandas DataFrame ({type(e).__name__}: {e})"))
        with os.fdopen(w, "wb") as f:
            f.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    os.waitpid(pid, 0)
    status, value = pickle.loads(data)
    if status == "err":
        raise RuntimeError(f"fn failed at snapshot_day={day}:\n{value}")
    return value


def build_features(fn, splits=("train", "validation")) -> pd.DataFrame:
    """Call fn(view, snapshot_day) once per snapshot day; return the aligned feature table
    (household_key, snapshot_day, your columns) for the train and validation keys.

    `view` is an AsOf: view.transactions, view.demographics, view.products, view.campaigns,
    view.campaign_targets, view.coupons, view.coupon_redemptions, view.display_mailer — all
    filtered to what was knowable on snapshot_day. view.households = the households that need
    a row at this snapshot. fn must return a DataFrame indexed by household_key.
    """
    if _IN_CHILD:
        raise AccessDenied("build_features cannot be called inside fn")
    if not set(splits) <= {"train", "validation"}:
        raise AccessDenied("only train and validation snapshots are available during research")
    _preload()
    keys = _acc.target_keys(tuple(splits))
    parts = []
    for day, grp in keys.groupby("snapshot_day", sort=True):
        hh = pd.Index(grp["household_key"].to_numpy(), name="household_key")
        out = _run_isolated(fn, int(day), hh)
        parts.append((int(day), hh, out))
    # Reuse the accessor's alignment/validation with precomputed outputs.
    cache = {d: o for d, _, o in parts}
    return _acc.build_features(lambda view, d: cache[d], splits=tuple(splits))


def baseline_features() -> pd.DataFrame:
    """The E000 feature table (demographics + snapshot calendar) for train and validation."""
    from src.features.baseline import build_baseline_features

    return build_baseline_features()


def save_table(df: pd.DataFrame, name: str) -> str:
    """Save a feature table to your workspace as <name>.parquet; returns the path to pass to experiment()."""
    if _IN_CHILD:
        raise AccessDenied("save_table cannot be called inside fn")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("save_table expects a pandas DataFrame")
    stem = name[: -len(".parquet")] if name.endswith(".parquet") else name
    safe = "".join(ch for ch in stem if ch.isalnum() or ch in "-_") or "table"
    path = f"{safe}.parquet"
    df.to_parquet(os.path.join(_WORKSPACE, path), index=False)
    return path

assert set(__all__) == set(_PUBLIC), "agent_api.__all__ and PUBLIC_API disagree"


def load_saved(path: str) -> pd.DataFrame:
    """Load a table you saved earlier with save_table (workspace only)."""
    if _IN_CHILD:
        raise AccessDenied("load_saved cannot be called inside fn")
    full = os.path.realpath(os.path.join(_WORKSPACE, path))
    if not full.startswith(os.path.realpath(_WORKSPACE) + os.sep) or not full.endswith(".parquet"):
        raise AccessDenied("load_saved only reads .parquet files saved in your workspace")
    return pd.read_parquet(full)
