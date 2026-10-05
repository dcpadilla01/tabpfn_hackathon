import numpy as np, pandas as pd, agent_api

e004 = agent_api.load_saved("e004_marketing_demo.parquet")
print("e004", e004.shape)
print(sorted(e004.columns.tolist()))

def feats(view, snapshot_day):
    d0 = int(snapshot_day)
    t = view.table("transactions")
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    day = t["day"].values
    sv = t["sales_value"].values.astype(float)
    hk = t["household_key"].values

    def wsum(lo, hi):  # spend on days lo < day <= hi
        m = (day > lo) & (day <= hi)
        if not m.any():
            return pd.Series(dtype=float)
        return pd.Series(sv[m], index=hk[m]).groupby(level=0).sum()

    for w in (7, 14):
        out[f"spend_{w}"] = wsum(d0 - w, d0).reindex(hh).fillna(0.0)

    # exponentially decayed spend sums (recency-weighted spend rate)
    for hl in (14, 28, 56, 112):
        wgt = np.power(0.5, (d0 - day) / hl)
        out[f"dec_{hl}"] = pd.Series(sv * wgt, index=hk).groupby(level=0).sum().reindex(hh).fillna(0.0)

    s28 = wsum(d0-28, d0).reindex(hh).fillna(0.0)
    s56 = wsum(d0-56, d0).reindex(hh).fillna(0.0)
    s84 = wsum(d0-84, d0).reindex(hh).fillna(0.0)
    s364 = wsum(d0-364, d0).reindex(hh).fillna(0.0)
    out["r_28_56"] = s28 / (s56 + 1.0)
    out["r_28_84"] = s28 / (s84 + 1.0)
    out["r_84_364"] = s84 / (s364 + 1.0)

    # household seasonality: same future window / trailing 28d, one year earlier
    out["ly_future28"] = wsum(d0-364, d0-336).reindex(hh).fillna(0.0)
    out["ly_trail28"] = wsum(d0-392, d0-364).reindex(hh).fillna(0.0)

    # weekly spend series (week_no = (day+8)//7)
    w0 = (d0 + 8) // 7
    wkarr = (day + 8) // 7
    wksum = pd.Series(sv, index=pd.MultiIndex.from_arrays([hk, wkarr])).groupby(level=[0,1]).sum().unstack(fill_value=0.0)

    def weekmat(weeks):
        P = wksum.reindex(columns=weeks).reindex(hh)
        return P.fillna(0.0).values

    W = weekmat(list(range(w0-12, w0)))
    out["active_weeks_12"] = (W > 0).sum(1) / 12.0
    out["weekly_mean_12"] = W.mean(1)
    out["weekly_std_12"] = W.std(1)
    out["weekly_max_12"] = W.max(1)
    x = np.arange(12, dtype=float); xm = x - x.mean()
    out["trend_slope_12"] = ((W - W.mean(1, keepdims=True)) * xm).sum(1) / (xm*xm).sum()

    W26 = weekmat(list(range(w0-26, w0)))
    Z = W26 <= 0
    streak = np.zeros(len(hh)); act = np.ones(len(hh), bool)
    for j in range(25, -1, -1):
        streak += Z[:, j] & act
        act &= Z[:, j]
    out["zero_streak_wk"] = streak

    out["spend_cur_wk"] = wsum(7*w0 - 9, d0).reindex(hh).fillna(0.0)

    # long-run baseline
    first = pd.Series(day, index=hk).groupby(level=0).min().reindex(hh)
    tot = pd.Series(sv, index=hk).groupby(level=0).sum().reindex(hh).fillna(0.0)
    out["total_spend"] = tot
    out["days_since_first"] = (d0 - first).fillna(0).astype(float)
    span = (d0 - first + 1).clip(lower=1).astype(float)
    out["avg_weekly_all"] = (tot * 7.0 / span).fillna(0.0)
    nwk = pd.Series(wkarr, index=hk).groupby(level=0).nunique().reindex(hh).fillna(0).astype(float)
    out["active_week_share"] = (nwk / np.ceil(span/7.0)).fillna(0.0)

    m84 = day > d0 - 84
    trips = t.loc[m84].groupby("household_key")["basket_id"].nunique().reindex(hh).fillna(0).astype(float)
    out["trips_84"] = trips
    out["avg_basket_84"] = (s84 / trips.replace(0.0, np.nan)).fillna(0.0)
    return out

newf = agent_api.build_features(feats)
print("newf", newf.shape)
print(newf[["dec_28","ly_future28","zero_streak_wk","trend_slope_12","avg_weekly_all"]].describe().round(2))

keys = ["household_key", "snapshot_day"]
new_cols = [c for c in newf.columns if c not in keys]
overlap = [c for c in new_cols if c in e004.columns]
if overlap:
    print("renaming overlaps:", overlap)
    newf = newf.rename(columns={c: c + "_ts" for c in overlap})
m = e004.merge(newf, on=keys, how="inner")
print("merged", m.shape)
assert len(m) == len(e004) == len(newf)
assert not m.duplicated(keys).any()
path = agent_api.save_table(m, "e005_trend_season")
print(path)

tt = agent_api.train_targets()
tr = m.merge(tt, on=keys)
y = tr["future_spend_4w"]
for c in new_cols:
    cc = c + "_ts" if c in overlap else c
    print(f"{cc:20s} corr={tr[cc].corr(y): .3f}")
