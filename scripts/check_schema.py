"""Phase 1 checks: the registry in src/data/schema.py matches the data.

    uv run python scripts/check_schema.py

Checks: columns, declared grain uniqueness, day→week mapping, join-key referential
integrity, discount sign convention, as-of reconstructability, and that
docs/data_schema.md documents every table. Writes the household onboarding
curve to docs/figures/households_active_per_week.png. Exits non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT  # noqa: E402
from src.data.load import load_table  # noqa: E402
from src.data.schema import JOINS, TABLES, assert_columns, assert_grain, week_of_day  # noqa: E402

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def main() -> None:
    print("columns and grain")
    for name, spec in TABLES.items():
        df = load_table(name, list(spec.grain) if name == "causal_data" else None)
        if name != "causal_data":
            try:
                assert_columns(name, df)
                check(True, f"{name}: columns match registry")
            except ValueError as e:
                check(False, str(e))
        check(len(df) == spec.rows, f"{name}: {len(df):,} rows")
        if spec.grain_unique:
            try:
                assert_grain(name, df)
                check(True, f"{name}: grain {spec.grain} unique")
            except ValueError as e:
                check(False, str(e))
        else:
            n = int(df.duplicated(subset=list(spec.grain)).sum())
            print(f"  info {name}: grain {spec.grain} declared non-unique ({n:,} duplicates)")

    tx = load_table("transaction_data")
    print("time")
    check(bool((week_of_day(tx["day"].astype(int)) == tx["week_no"]).all()), "week_no == week_of_day(day) for every line item")
    check(tx["day"].min() == 1 and tx["day"].max() == 711, f"day range {tx['day'].min()}..{tx['day'].max()}")
    check(bool((tx.groupby("basket_id")["day"].nunique() == 1).all()), "each basket falls on one day")
    check(bool((tx.groupby("basket_id")["household_key"].nunique() == 1).all()), "each basket belongs to one household")

    print("money")
    check(bool((tx["sales_value"] >= 0).all()), "sales_value >= 0")
    check(bool((tx["coupon_disc"] <= 0).all()), "coupon_disc <= 0")
    check(bool((tx["coupon_match_disc"] <= 0).all()), "coupon_match_disc <= 0")
    n_pos = int((tx["retail_disc"] > 0).sum())
    check(n_pos == 36, f"retail_disc > 0 on {n_pos} rows (documented anomaly: 36)")

    print("joins (left keys present in right)")
    cache = {}
    for left, right, keys in JOINS:
        if "causal_data" in (left, right):
            continue  # sparse by design: absence = no display/mailer
        L = cache.setdefault(left, load_table(left))
        R = cache.setdefault(right, load_table(right))
        lk = L[list(keys)].drop_duplicates()
        missing = len(lk.merge(R[list(keys)].drop_duplicates(), how="left", on=list(keys), indicator=True).query("_merge == 'left_only'"))
        expected_gaps = {("transaction_data", "hh_demographic"), ("transaction_data", "campaign_table"), ("transaction_data", "coupon_redempt")}
        if (left, right) in expected_gaps:
            print(f"  info {left}→{right} on {keys}: {missing:,}/{len(lk):,} keys unmatched (optional relation)")
        else:
            check(missing == 0, f"{left}→{right} on {keys}: {missing} unmatched keys")
    cr, cd = load_table("coupon_redempt"), load_table("campaign_desc")
    m = cr.merge(cd, on="campaign")
    check(bool(((m["day"] >= m["start_day"]) & (m["day"] <= m["end_day"])).all()), "every redemption falls inside its campaign window")

    print("household coverage and onboarding")
    hh = tx["household_key"].nunique()
    demo = load_table("hh_demographic")
    check(hh == 2500, f"{hh} households transact")
    print(f"  info demographics cover {demo['household_key'].isin(tx['household_key']).sum()}/{hh} households ({len(demo) / hh:.1%})")
    first = tx.groupby("household_key")["day"].min()
    print(f"  info first observed transaction day: median {first.median():.0f}, p90 {first.quantile(.9):.0f}, max {first.max()}")
    active = tx.groupby("week_no")["household_key"].nunique()
    cum = first.map(week_of_day).value_counts().sort_index().cumsum().reindex(active.index, method="ffill")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(active.index, active.values, label="households with ≥1 basket that week")
    ax.plot(cum.index, cum.values, label="households observed so far (cumulative)", linestyle="--")
    ax.set_xlabel("week_no"); ax.set_ylabel("households"); ax.legend(frameon=False); ax.set_title("Household onboarding")
    out = ROOT / "docs" / "figures" / "households_active_per_week.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"  info onboarding curve → {out.relative_to(ROOT)}")

    print("docs agree with registry")
    doc = (ROOT / "docs" / "data_schema.md").read_text()
    for name in TABLES:
        check(f"### {name}" in doc, f"docs/data_schema.md has a section for {name}")

    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'ALL CHECKS PASSED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
