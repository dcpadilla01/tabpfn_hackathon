import pandas as pd, numpy as np

def _gap_stats(days, lo, hi):
    days = np.sort(np.asarray(days, dtype=float))
    edges = np.concatenate([[lo], days, [hi]])
    dif = np.diff(edges)
    inner = dif[1:-1]
    return pd.Series({
        "maxgap": dif.max(),
        "meangap": inner.mean() if len(inner) else np.nan,
        "frac_gt28": (dif > 28).mean(),
        "cur_streak": dif[-1],
    })

def feats(view, day):
    d = day
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)].copy()
    tx["rd"] = (-tx.retail_disc).clip(lower=0)
    tx["cd"] = (-tx.coupon_disc).clip(lower=0)
    tx["cm"] = (-tx.coupon_match_disc).clip(lower=0)
    out = pd.DataFrame(index=hh)

    # ---- deal / discount reliance ----
    for lo, hi, tag in [(0, 84, "84"), (0, 364, "364")]:
        w = tx[(tx.day > d - hi) & (tx.day <= d - lo)]
        g = w.groupby("household_key")
        spend = g["sales_value"].sum().clip(lower=1e-9)
        trips = g["basket_id"].nunique().clip(lower=1)
        lines = g.size().clip(lower=1)
        out[f"deal_r_{tag}"] = (g["rd"].sum() / spend).clip(0, 3)
        out[f"deal_c_{tag}"] = (g["cd"].sum() / spend).clip(0, 3)
        out[f"deal_cm_{tag}"] = (g["cm"].sum() / spend).clip(0, 3)
        sv = w["sales_value"].where(w["sales_value"] > 0)
        deep = ((w["rd"] / sv > 0.3) & sv.notna()).groupby(w["household_key"]).sum()
        out[f"deep_frac_{tag}"] = deep / lines
        out[f"deal_per_trip_{tag}"] = g["rd"].sum() / trips

    # ---- coupon redemptions ----
    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hh)]
    out["red_84"] = cr[cr.day > d - 84].groupby("household_key").size().reindex(hh).fillna(0)
    out["red_364"] = cr[cr.day > d - 364].groupby("household_key").size().reindex(hh).fillna(0)
    out["red_ever"] = cr.groupby("household_key").size().gt(0).astype(float).reindex(hh).fillna(0)
    w364r = cr[cr.day > d - 364]
    out["red_distinct_364"] = w364r.groupby("household_key")["coupon_upc"].nunique().reindex(hh).fillna(0)
    out["red_camp_364"] = w364r.groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    last_red = cr.groupby("household_key")["day"].max().reindex(hh)
    out["days_since_red"] = (d - last_red).clip(upper=364).fillna(364)

    # ---- gap-hazard features ----
    t2 = tx[["household_key", "day"]].drop_duplicates()
    for hi, tag in [(112, "112"), (364, "364")]:
        w = t2[(t2.day > d - hi) & (t2.day <= d)]
        res = w.groupby("household_key")["day"].apply(lambda s: _gap_stats(s.values, d - hi, d))
        res = pd.DataFrame(list(res), index=res.index)
        for c in res.columns:
            out[f"{c}_{tag}"] = res[c].reindex(hh)

    # weekly activity over 112d
    w = t2[t2.day > d - 112].copy()
    w["wk"] = (d - w["day"]) // 7
    sets = w.groupby("household_key")["wk"].apply(lambda s: set(s.unique()))
    def streak(s):
        k = 0
        while k in s:
            k += 1
        return k
    out["active_streak_w"] = sets.apply(streak).reindex(hh).fillna(0)
    nactive_w = w.groupby("household_key")["wk"].nunique().reindex(hh).fillna(0)
    out["inactive7_frac_112"] = 1 - nactive_w / 16.0

    # aligned 28d windows over 364d
    w3 = t2[t2.day > d - 364].copy()
    w3["wb"] = (d - w3["day"]) // 28
    nact28 = w3.groupby("household_key")["wb"].nunique().reindex(hh).fillna(0)
    out["p_zero_28_364"] = 1 - nact28 / 13.0

    mg = out["meangap_364"]
    last_day = t2.groupby("household_key")["day"].max().reindex(hh)
    dsl = (d - last_day).clip(lower=0)
    out["exp_trips_28"] = (28.0 / mg.clip(lower=1)).clip(upper=28)
    out["hazard_ratio"] = (dsl / mg.clip(lower=1)).clip(upper=30)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

blk = agent_api.build_features(feats)
print("block:", blk.shape)
base = agent_api.load_saved("e011_table.parquet")
print("base:", base.shape)
m = base.merge(blk.drop(columns=["snapshot_day"]), on="household_key", how="left", validate="one_to_one")
print("merged:", m.shape, "nulls in new cols:", int(m.isna().sum().sum() - base.isna().sum().sum()))
key = agent_api.KEYS
assert set(map(tuple, m[key].values)) == set(map(tuple, base[key].values))
p = agent_api.save_table(m, "deal_hazard_union_v2.parquet")
print(p)
