import numpy as np, pandas as pd
import agent_api

base = agent_api.load_saved("season_demo_v1.parquet")
print("base shape:", base.shape)
print("base last cols:", list(base.columns)[-5:])

def get_hh(view):
    h = view.households
    if hasattr(h, "columns"):
        hh = pd.Index(h["household_key"]) if "household_key" in getattr(h, "columns", []) else pd.Index(h.iloc[:, 0])
    else:
        hh = pd.Index(h)
    return hh

def fn(view, snapshot_day):
    hh = get_hh(view)
    out = pd.DataFrame(index=hh)
    t = view.table("transactions")
    t = t[t.household_key.isin(hh)]
    if len(t) == 0:
        return out
    t = t.copy()
    t["qty"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0).clip(lower=0)
    g = t.groupby("household_key")

    # tenure / lifetime scale
    out["b_tenure"] = (snapshot_day - g["day"].min()).reindex(hh)
    out["b_life_spend"] = g["sales_value"].sum().reindex(hh)
    out["b_life_trips"] = g["basket_id"].nunique().reindex(hh)
    out["b_life_spend_pw"] = out["b_life_spend"] / (out["b_tenure"] / 7.0).clip(lower=1)

    # active-week coverage in last 84d
    w84 = int(((snapshot_day + 8) // 7) - ((snapshot_day - 83 + 8) // 7) + 1)
    tw = t[t["day"] > snapshot_day - 84]
    act = (((tw["day"] + 8) // 7).groupby(tw["household_key"]).nunique())
    out["b_active_weeks_84"] = (act / w84).reindex(hh)

    # aligned 28d windows w1..w6: coverage, level stats, volatility, slope
    W = []
    cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        lo = snapshot_day - 28 * (k + 1)
        m = (t["day"] > lo) & (t["day"] <= lo + 28)
        s = t[m].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0)
        W.append(s.values)
        cnt = cnt + (s > 0).astype(float)
    Wdf = pd.DataFrame(np.array(W).T, index=hh, columns=[f"w{k}" for k in range(1, 7)])
    out["b_active_windows_6"] = cnt
    out["b_w_mean6"] = Wdf.mean(axis=1)
    out["b_w_std6"] = Wdf.std(axis=1)
    out["b_w_cv6"] = out["b_w_std6"] / out["b_w_mean6"].replace(0.0, np.nan)
    out["b_w_min6"] = Wdf.min(axis=1)
    out["b_w_max6"] = Wdf.max(axis=1)
    out["b_w_slope"] = Wdf[["w1", "w2"]].mean(axis=1) - Wdf[["w5", "w6"]].mean(axis=1)

    # max gap between purchases in last 112d (incl. window edges)
    tw2 = t[t["day"] > snapshot_day - 112]
    lo112 = snapshot_day - 111
    def max_gap(days):
        d = np.sort(days.values)
        gaps = np.diff(d)
        mg = gaps.max() if len(gaps) else 0
        return max(mg, d[0] - lo112, snapshot_day - d[-1])
    out["b_maxgap_112"] = tw2.groupby("household_key")["day"].apply(max_gap).reindex(hh)

    # trip concentration: recent trip rate vs 84d trip rate
    w28n = int(((snapshot_day + 8) // 7) - ((snapshot_day - 27 + 8) // 7) + 1)
    tr28 = t[t["day"] > snapshot_day - 28].groupby("household_key")["basket_id"].nunique()
    tr84 = t[t["day"] > snapshot_day - 84].groupby("household_key")["basket_id"].nunique()
    out["b_trip_conc"] = ((tr28 / w28n) / (tr84 / w84).replace(0.0, np.nan)).reindex(hh)

    # basket scale over last 84d
    gg = tw.groupby("household_key")
    trips = gg["basket_id"].nunique()
    out["b_units_84"] = gg["qty"].sum().reindex(hh)
    out["b_units_per_trip_84"] = (gg["qty"].sum() / trips).reindex(hh)
    out["b_nprod_84"] = gg["product_id"].nunique().reindex(hh)
    out["b_lines_per_trip_84"] = (gg.size() / trips).reindex(hh)

    return out.astype(float)

feat = agent_api.build_features(fn)
print("feat shape:", feat.shape)
fm = feat.copy() if "household_key" in feat.columns else feat.reset_index()
print("fm cols:", list(fm.columns)[:6])
m = base.merge(fm, on=["household_key", "snapshot_day"], how="left")
newcols = [c for c in m.columns if c.startswith("b_")]
print("merged shape:", m.shape, "| n new cols:", len(newcols))
print(newcols)
print(m[newcols].isna().mean().round(3).to_dict())
assert m.shape[0] == base.shape[0]
path = agent_api.save_table(m, "churn_vol_v1")
print("saved:", path)
