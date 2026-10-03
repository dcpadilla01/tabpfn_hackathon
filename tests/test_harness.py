"""Leakage-by-construction and evaluator-contract tests. No API calls."""

import pandas as pd
import pytest

from src.data.accessor import AsOf, build_features, history, target_keys
from src.data.schema import week_of_day
from src.evaluation.evaluator import FeatureTableError, validate_feature_table


@pytest.mark.parametrize("day", [95, 300, 543])
def test_as_of_view_never_sees_the_future(day):
    v = AsOf(day)
    assert v.transactions["day"].max() <= day
    assert v.coupon_redemptions["day"].max() <= day if len(v.coupon_redemptions) else True
    assert (v.campaigns["start_day"] <= day).all()
    assert v.campaign_targets["campaign"].isin(v.campaigns["campaign"]).all()
    assert v.coupons["campaign"].isin(v.campaigns["campaign"]).all()
    assert v.display_mailer["week_no"].max() <= week_of_day(day)


def test_no_campaigns_before_day_224():
    assert AsOf(223).campaigns.empty and AsOf(223).campaign_targets.empty


def test_history_is_one_household_as_of():
    h = history(1, 200)
    assert set(h["household_key"]) == {1} and h["day"].max() <= 200


def test_views_are_copies():
    v = AsOf(300)
    v.transactions.loc[:, "sales_value"] = -1.0
    assert (v.transactions["sales_value"] >= 0).all()


def test_build_features_sees_only_as_of_data_and_aligns_keys():
    seen = {}

    def fn(view, day):
        seen[day] = view.transactions["day"].max()
        return pd.DataFrame({"n_lines": view.transactions.groupby("household_key").size()})

    ft = build_features(fn)
    assert all(mx <= d for d, mx in seen.items())
    keys = target_keys()
    assert len(ft) == len(keys)
    assert ft[["household_key", "snapshot_day"]].equals(keys[["household_key", "snapshot_day"]])
    validate_feature_table(ft)


def _ok_table():
    k = target_keys()[["household_key", "snapshot_day"]]
    return k.assign(x=1.0)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda t: t.drop(index=0), "missing"),
        (lambda t: pd.concat([t, t.iloc[[0]]]), "duplicate"),
        (lambda t: t.assign(future_spend_4w=0.0), "forbidden"),
        (lambda t: t.assign(split="train"), "forbidden"),
        (lambda t: t.assign(x=float("inf")), "inf"),
        (lambda t: t[["household_key", "snapshot_day"]], "no feature"),
        (lambda t: t.assign(when=pd.Timestamp("2020-01-01")), "unsupported dtype"),
    ],
)
def test_evaluator_rejects_contract_violations(mutate, msg):
    with pytest.raises(FeatureTableError, match=msg):
        validate_feature_table(mutate(_ok_table()))


def test_test_keys_rejected_during_research():
    t = target_keys(("train", "validation", "test"))[["household_key", "snapshot_day"]].assign(x=1.0)
    with pytest.raises(FeatureTableError, match="unexpected"):
        validate_feature_table(t)
