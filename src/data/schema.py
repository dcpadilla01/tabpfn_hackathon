"""dunnhumby Complete Journey — table registry.

Single source of truth for file names, columns, grain, time semantics and
join keys. Imported by the as-of accessor (src/data/accessor.py) and by
the evaluator's feature-table validation. Column names are lowercased on
load; everything here is lowercase.

Time semantics
--------------
day      : integer day index, 1..~711. Snapshots are defined on `day`.
week_no  : integer week index, 1..102. Week 1 is days 1-5 (a partial week);
           from day 6 on, weeks are 7 days: week_no == (day + 8) // 7.
           Verified for every day in transaction_data by scripts/check_schema.py.

A table is *temporal* if it has a `time_col`; the accessor filters it with
`time_col <= as_of` (converted to weeks when `time_unit == "week"`).
A table with `time_col=None` is *static* and readable directly.
`campaign_table` has no date of its own: it becomes visible via
`campaign_desc.start_day <= as_of` (see `time_via`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TimeUnit = Literal["day", "week"]


@dataclass(frozen=True)
class TableSpec:
    name: str
    file: str
    columns: tuple[str, ...]
    grain: tuple[str, ...]
    description: str
    rows: int                          # expected row count of the raw release
    md5: str                           # md5 of the raw CSV (see data/README.md)
    dtypes: dict[str, str] = field(default_factory=dict)
    grain_unique: bool = True          # False → duplicates on the grain exist in the source (see notes)
    time_col: str | None = None
    time_unit: TimeUnit | None = None
    # (table, column): apply the as-of filter through this other table's column
    time_via: tuple[str, str] | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_static(self) -> bool:
        return self.time_col is None and self.time_via is None


TABLES: dict[str, TableSpec] = {
    # ------------------------------------------------------------------ data tables
    "transaction_data": TableSpec(
        name="transaction_data",
        file="transaction_data.csv",
        columns=(
            "household_key", "basket_id", "day", "product_id", "quantity",
            "sales_value", "store_id", "coupon_match_disc", "coupon_disc",
            "retail_disc", "trans_time", "week_no",
        ),
        grain=("household_key", "basket_id", "product_id", "day"),
        rows=2_595_732,
        md5="e5d2df5bfb59b7d658ea496539be5ec0",
        time_col="day",
        time_unit="day",
        dtypes={
            "household_key": "int32", "basket_id": "int64", "day": "int16",
            "product_id": "int32", "quantity": "int32", "sales_value": "float64",
            "store_id": "int32", "coupon_match_disc": "float64",
            "coupon_disc": "float64", "retail_disc": "float64",
            "trans_time": "int16", "week_no": "int8",
        },
        description=(
            "Line-item purchases for ~2,500 households over ~2 years. "
            "household_key: household. basket_id: purchase occasion. day: day of "
            "transaction. product_id: product. quantity: units bought in the trip. "
            "sales_value: dollars the retailer receives from the sale. store_id: store. "
            "coupon_match_disc: discount from retailer's match of a manufacturer coupon. "
            "coupon_disc: discount from a manufacturer coupon. retail_disc: discount "
            "from the retailer's loyalty-card programme. trans_time: time of day (HHMM). "
            "week_no: week of transaction, 1-102."
        ),
        notes=(
            "Target future_spend_4w = sum(sales_value) over (snapshot_day, snapshot_day+28].",
            "sales_value = dollars the retailer receives: net of loyalty (retail_disc) and "
            "coupon-match discounts, gross of manufacturer coupons (reimbursed to the retailer).",
            "Discounts are stored as negative numbers. 36 rows have retail_disc > 0 (source anomaly, kept).",
            "quantity is huge (up to ~90k) for fuel-like items; sales_value is the reliable spend measure.",
            "A basket belongs to exactly one household and one day.",
        ),
    ),
    "hh_demographic": TableSpec(
        name="hh_demographic",
        file="hh_demographic.csv",
        columns=(
            "household_key", "classification_1", "classification_2",
            "classification_3", "classification_4", "classification_5",
            "homeowner_desc", "kid_category_desc",
        ),
        grain=("household_key",),
        rows=801,
        md5="8c99c72097c3fdb5612aa6fbbf38eacd",
        dtypes={
            "household_key": "int32",
            **{f"classification_{i}": "category" for i in range(1, 6)},
            "homeowner_desc": "category", "kid_category_desc": "category",
        },
        description=(
            "Anonymised household demographics, one row per household, for 801 of the 2,500 "
            "households. Generic names, but values are ORDINAL: classification_1 'Age Group1'..'Age Group6'; "
            "classification_2 X/Y/Z (not ordered); classification_3 'Level1'..'Level12'; "
            "classification_4 household size '1'..'5+'; classification_5 'Group1'..'Group6'; "
            "homeowner_desc (Homeowner, Probable Owner, Probable Renter, Renter, Unknown); "
            "kid_category_desc ('1', '2', '3+', 'None/Unknown')."
        ),
        notes=(
            "Covers 801 / 2,500 households (32%); every demographic household transacts.",
            "Lexical order is wrong for ordinal codes: 'Level10' < 'Level2'.",
            "Static snapshot of unknown date; treated as known at every snapshot.",
        ),
    ),
    "campaign_table": TableSpec(
        name="campaign_table",
        file="campaign_table.csv",
        columns=("household_key", "campaign", "description"),
        grain=("household_key", "campaign"),
        rows=7_208,
        md5="0e52350f60d17d6a37d8f2020a7dff15",
        time_via=("campaign_desc", "start_day"),
        dtypes={"household_key": "int32", "campaign": "int16", "description": "category"},
        description=(
            "Which households were targeted by which campaign. description is the "
            "campaign type (e.g. TypeA/TypeB/TypeC). No date of its own."
        ),
        notes=("Visible to a snapshot only if campaign_desc.start_day <= snapshot_day.",),
    ),
    # ---------------------------------------------------------------- lookup tables
    "campaign_desc": TableSpec(
        name="campaign_desc",
        file="campaign_desc.csv",
        columns=("campaign", "description", "start_day", "end_day"),
        grain=("campaign",),
        rows=30,
        md5="d65df30fa3f148ed5581486e1d4064da",
        time_col="start_day",
        time_unit="day",
        dtypes={"campaign": "int16", "description": "category", "start_day": "int16", "end_day": "int16"},
        description="One row per campaign: type and the day range during which it ran.",
        notes=("end_day may be > snapshot_day for in-flight campaigns; that is allowed (start is what leaks).",),
    ),
    "coupon_redempt": TableSpec(
        name="coupon_redempt",
        file="coupon_redempt.csv",
        columns=("household_key", "day", "coupon_upc", "campaign"),
        grain=("household_key", "day", "coupon_upc", "campaign"),
        rows=2_318,
        md5="e5706268c103cdd92dadbe48106107d7",
        time_col="day",
        time_unit="day",
        dtypes={"household_key": "int32", "day": "int16", "coupon_upc": "int64", "campaign": "int16"},
        description="Coupon redemptions: which household redeemed which coupon on which day, under which campaign.",
    ),
    "coupon": TableSpec(
        name="coupon",
        file="coupon.csv",
        columns=("campaign", "coupon_upc", "product_id"),
        grain=("campaign", "coupon_upc", "product_id"),
        grain_unique=False,  # 5,164 exact duplicate rows in the source; kept as-is
        rows=124_548,
        md5="8dbe67d1284fc66cba5c5f56cd305d17",
        dtypes={"campaign": "int16", "coupon_upc": "int64", "product_id": "int32"},
        description="Which coupons were part of which campaign and which products they applied to.",
        notes=("Static, but only meaningful once its campaign has started; join through campaign_desc when deriving as-of features.",),
    ),
    "product": TableSpec(
        name="product",
        file="product.csv",
        columns=(
            "product_id", "department", "commodity_desc", "sub_commodity_desc",
            "manufacturer", "brand", "curr_size_of_product",
        ),
        grain=("product_id",),
        rows=92_353,
        md5="b53bbdd6cf49293c5a1927e56ad3157a",
        dtypes={
            "product_id": "int32", "department": "category", "commodity_desc": "category",
            "sub_commodity_desc": "category", "manufacturer": "int32",
            "brand": "category", "curr_size_of_product": "string",
        },
        description=(
            "Product hierarchy: department > commodity > sub-commodity; manufacturer code; "
            "brand (National/Private); package size as free text."
        ),
    ),
    "causal_data": TableSpec(
        name="causal_data",
        file="causal_data.csv",
        columns=("product_id", "store_id", "week_no", "display", "mailer"),
        grain=("product_id", "store_id", "week_no"),
        grain_unique=False,  # 15,245 rows: same product/store/week on two display locations
        rows=36_786_524,
        md5="db6edee61cc369cc5c86db4cafe0f1fb",
        time_col="week_no",
        time_unit="week",
        dtypes={"product_id": "int32", "store_id": "int32", "week_no": "int8", "display": "category", "mailer": "category"},
        # read display/mailer as strings: codes mix digits and letters ('0'..'9', 'A')
        description=(
            "In-store display location and weekly mailer placement per product x store x week. "
            "Marketing exposure context — 'causal' is dunnhumby's name, not a causal-inference claim."
        ),
        notes=(
            "Large (36.8M rows); covers weeks 9-101 only. Absence of a row = not on display and not in mailer.",
            "display: 0 none, 1 store front, 2 store rear, 3 front end cap, 4 mid-aisle end cap, "
            "5 rear end cap, 6 side-aisle end cap, 7 in-aisle, 9 secondary location, A in-shelf.",
            "mailer: 0 not on ad, A interior page feature, C interior line item, D front page feature, "
            "F back page feature, H wrap front feature, J wrap interior coupon, L wrap back feature, "
            "P interior page coupon, X free on interior page, Z free on front/back page or wrap.",
        ),
    ),
}


# (left_table, right_table, join_keys)
JOINS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("transaction_data", "hh_demographic", ("household_key",)),
    ("transaction_data", "product", ("product_id",)),
    ("transaction_data", "causal_data", ("product_id", "store_id", "week_no")),
    ("transaction_data", "campaign_table", ("household_key",)),
    ("transaction_data", "coupon_redempt", ("household_key",)),
    ("campaign_table", "campaign_desc", ("campaign",)),
    ("coupon_redempt", "campaign_desc", ("campaign",)),
    ("coupon_redempt", "coupon", ("coupon_upc", "campaign")),
    ("coupon", "campaign_desc", ("campaign",)),
    ("coupon", "product", ("product_id",)),
)

TEMPORAL_TABLES = tuple(t for t, s in TABLES.items() if s.time_col is not None)
STATIC_TABLES = tuple(t for t, s in TABLES.items() if s.is_static)
VIA_TABLES = tuple(t for t, s in TABLES.items() if s.time_via is not None)

HOUSEHOLD_KEY = "household_key"
SPEND_COL = "sales_value"
DAY_COL = "day"
WEEK_COL = "week_no"
DAYS_PER_WEEK = 7
HORIZON_DAYS = 28


def week_of_day(day):
    """dunnhumby week index for a day index: week 1 = days 1-5, then 7-day weeks.
    Works on ints and numpy/pandas arrays. Verified in check_schema.py."""
    return (day + 8) // DAYS_PER_WEEK


def as_of_bound(table: str, as_of_day: int) -> tuple[str, int] | None:
    """Return (column, inclusive_upper_bound) for the as-of filter on `table`,
    or None for static tables. For `time_via` tables the caller filters the
    referenced table first and semi-joins."""
    spec = TABLES[table]
    if spec.time_col is None:
        return None
    bound = week_of_day(as_of_day) if spec.time_unit == "week" else as_of_day
    return spec.time_col, bound


# ----------------------------------------------------------------- validation helpers
def assert_columns(table: str, df) -> None:
    expected = set(TABLES[table].columns)
    got = set(c.lower() for c in df.columns)
    missing, extra = expected - got, got - expected
    if missing:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")
    if extra:
        raise ValueError(f"{table}: unexpected columns {sorted(extra)}")


def assert_grain(table: str, df) -> None:
    spec = TABLES[table]
    if not spec.grain_unique:
        return
    dupes = df.duplicated(subset=list(spec.grain)).sum()
    if dupes:
        raise ValueError(f"{table}: grain {spec.grain} not unique ({dupes} duplicate rows)")


def assert_join_keys_exist() -> None:
    for left, right, keys in JOINS:
        for k in keys:
            for t in (left, right):
                if k not in TABLES[t].columns:
                    raise ValueError(f"join key {k!r} not in {t}")


assert_join_keys_exist()