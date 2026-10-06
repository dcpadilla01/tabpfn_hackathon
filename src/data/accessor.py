"""As-of data accessor — the only way any arm touches temporal data.

Leakage is prevented by construction: every temporal table is filtered to what was
knowable on `as_of_day` *before* caller code sees it.

    history(household_key, as_of_day)            -> that household's line items with day <= as_of_day
    snapshot(as_of_day)                          -> AsOf view of every table at that day
    build_features(fn, splits=("train", "validation"))
        calls fn(view: AsOf, snapshot_day) once per snapshot day; fn returns a DataFrame
        indexed by household_key (one row per household). The result is aligned to the
        canonical keys of the requested splits and returned with household_key and
        snapshot_day columns, ready for experiment().

`build_features` is vectorised per snapshot (22 calls, not ~49k): fn sees every
household's as-of history at once and is free to group by household_key.

Visibility rules (docs/data_schema.md):
    transaction_data, coupon_redempt   day <= as_of_day
    campaign_desc                      start_day <= as_of_day
    campaign_table, coupon             their campaign's start_day <= as_of_day
    causal_data (display/mailer)       week_no <= week_of_day(as_of_day)
    hh_demographic, product            static, always visible
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable

import pandas as pd

from src.data.load import load_table
from src.data.schema import HOUSEHOLD_KEY, TABLES, week_of_day

# Agent-facing names. `causal_data` is exposure data, not causal inference.
AGENT_NAMES = {
    "transactions": "transaction_data",
    "demographics": "hh_demographic",
    "products": "product",
    "campaigns": "campaign_desc",
    "campaign_targets": "campaign_table",
    "coupon_redemptions": "coupon_redempt",
    "coupons": "coupon",
    "display_mailer": "causal_data",
}


class AsOf:
    """Every table as it was knowable on `day`. Attributes are computed lazily and cached;
    each access returns a fresh copy so caller code cannot corrupt the view."""

    #: Latest day a view may be built for; None = unrestricted (harness). Set only inside the agent's
    #: subprocess (see agent_api._lock_horizon). Dunder name: agent code cannot read or write it.
    __horizon__: int | None = None

    def __init__(self, day: int, households: pd.Index | None = None):
        horizon = type(self).__horizon__
        if horizon is not None and int(day) > horizon:
            raise PermissionError(f"a view for day {int(day)} is beyond the allowed horizon (day {horizon})")
        self.day = int(day)
        self.week = int(week_of_day(self.day))
        #: households with a target row at this snapshot (None outside build_features)
        self.households = households

    # ---------------------------------------------------------------- temporal
    @cached_property
    def _transactions(self) -> pd.DataFrame:
        tx = load_table("transaction_data")
        return tx[tx["day"] <= self.day].reset_index(drop=True)

    @cached_property
    def _campaigns(self) -> pd.DataFrame:
        cd = load_table("campaign_desc")
        return cd[cd["start_day"] <= self.day].reset_index(drop=True)

    @cached_property
    def _campaign_targets(self) -> pd.DataFrame:
        ct = load_table("campaign_table")
        return ct[ct["campaign"].isin(self._campaigns["campaign"])].reset_index(drop=True)

    @cached_property
    def _coupon_redemptions(self) -> pd.DataFrame:
        cr = load_table("coupon_redempt")
        return cr[cr["day"] <= self.day].reset_index(drop=True)

    @cached_property
    def _coupons(self) -> pd.DataFrame:
        cp = load_table("coupon")
        return cp[cp["campaign"].isin(self._campaigns["campaign"])].reset_index(drop=True)

    @cached_property
    def _display_mailer(self) -> pd.DataFrame:
        dm = _causal()
        return dm[dm["week_no"] <= self.week].reset_index(drop=True)

    @property
    def transactions(self) -> pd.DataFrame:
        return self._transactions.copy()

    @property
    def campaigns(self) -> pd.DataFrame:
        return self._campaigns.copy()

    @property
    def campaign_targets(self) -> pd.DataFrame:
        return self._campaign_targets.copy()

    @property
    def coupon_redemptions(self) -> pd.DataFrame:
        return self._coupon_redemptions.copy()

    @property
    def coupons(self) -> pd.DataFrame:
        return self._coupons.copy()

    @property
    def display_mailer(self) -> pd.DataFrame:
        return self._display_mailer.copy()

    # ------------------------------------------------------------------ static
    @property
    def demographics(self) -> pd.DataFrame:
        return load_table("hh_demographic")

    @property
    def products(self) -> pd.DataFrame:
        return load_table("product")

    def table(self, name: str) -> pd.DataFrame:
        """Dynamic access by agent-facing name, e.g. view.table("transactions") (Env-2)."""
        if name not in AGENT_NAMES:
            raise KeyError(f"unknown table {name!r}; tables: {', '.join(AGENT_NAMES)}")
        return getattr(self, name)

    def __repr__(self) -> str:
        n = "all" if self.households is None else len(self.households)
        return f"AsOf(day={self.day}, week={self.week}, households={n}, tables={list(AGENT_NAMES)})"


_causal_cache: pd.DataFrame | None = None


def _causal() -> pd.DataFrame:
    global _causal_cache
    if _causal_cache is None:
        _causal_cache = load_table("causal_data")
    return _causal_cache


def snapshot(as_of_day: int) -> AsOf:
    return AsOf(as_of_day)


def history(household_key: int, as_of_day: int) -> pd.DataFrame:
    """One household's line items with day <= as_of_day."""
    tx = AsOf(as_of_day)._transactions
    return tx[tx[HOUSEHOLD_KEY] == household_key].reset_index(drop=True)


def target_keys(splits: tuple[str, ...] = ("train", "validation")) -> pd.DataFrame:
    """Canonical (household_key, snapshot_day, split) rows — never the labels."""
    from src.data.targets import KEYS, load_targets

    t = load_targets()
    return t.loc[t["split"].isin(splits), KEYS + ["split"]].reset_index(drop=True)


def build_features(
    fn: Callable[[AsOf, int], pd.DataFrame],
    splits: tuple[str, ...] = ("train", "validation"),
) -> pd.DataFrame:
    """Apply fn per snapshot day on the as-of view; align to the canonical keys.

    fn(view, snapshot_day) must return a DataFrame indexed by household_key (extra
    households are dropped, missing ones get NaN). Returns household_key, snapshot_day
    and the feature columns, one row per canonical key, in canonical order.
    """
    keys = target_keys(splits)
    parts = []
    for day, grp in keys.groupby("snapshot_day", sort=True):
        hh = pd.Index(grp[HOUSEHOLD_KEY].to_numpy(), name=HOUSEHOLD_KEY)
        out = fn(AsOf(int(day), households=hh), int(day))
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"fn must return a DataFrame indexed by {HOUSEHOLD_KEY}, got {type(out).__name__}")
        if out.index.name != HOUSEHOLD_KEY and HOUSEHOLD_KEY in out.columns:
            out = out.set_index(HOUSEHOLD_KEY)
        if out.index.has_duplicates:
            raise ValueError(f"fn returned duplicate {HOUSEHOLD_KEY} values at snapshot_day={day}")
        out = out.drop(columns=["snapshot_day", "split"], errors="ignore").reindex(hh)
        out.insert(0, "snapshot_day", int(day))
        parts.append(out.reset_index())
    features = pd.concat(parts, ignore_index=True)
    return keys[[HOUSEHOLD_KEY, "snapshot_day"]].merge(features, on=[HOUSEHOLD_KEY, "snapshot_day"], how="left")


def describe_tables() -> str:
    """Agent-facing description of every table (names, columns, visibility)."""
    lines = []
    for agent_name, table in AGENT_NAMES.items():
        spec = TABLES[table]
        lines.append(f"{agent_name}: {spec.description}")
        lines.append(f"  columns: {', '.join(spec.columns)}")
    return "\n".join(lines)
