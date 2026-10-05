import numpy as np, pandas as pd, xgboost as xgb

def make_new(view, snap):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx["day"].values
    out = pd.DataFrame(index=hh)
    def agg(mask):
        sub = tx[mask]; g = sub.groupby("household_key")
        return (g["sales_value"].sum().reindex(hh, fill_value=0.0),
                g["basket_id"].nunique().reindex(hh, fill_value=0),
                g["day"].nunique().reindex(hh, fill_value=0))
    for w in [7, 14, 28, 84]:
        s, b, a = agg(d > snap - w)
        out[f"spend_{w}"] = s; out[f"bask_{w}"] = b; out[f"act_{w}"] = a
    for lo, hi, name in [(snap-56, snap-28, "mid_5_8"), (snap-84, snap-56, "mid_9_12")]:
        s, _, _ = agg((d > lo) & (d <= hi)); out[f"spend_{name}"] = s
    for k in [336, 364, 392]:
        s, _, _ = agg((d > snap - k) & (d <= snap - k + 28)); out[f"spend_seas_{k}"] = s
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    out["seas_ok"] = (fd <= snap - 392).astype(float)
    m84 = d > snap - 84
    daily = tx[m84].groupby(["household_key", "day"])["sales_value"].sum()
    g2 = daily.groupby("household_key")
    out["daily_mean_84"] = g2.mean().reindex(hh, fill_value=0.0)
    out["daily_std_84"] = g2.std().reindex(hh).fillna(0.0)
    out["daily_max_84"] = g2.max().reindex(hh, fill_value=0.0)
    txw = tx[m84].copy(); txw["wk"] = txw["day"] // 7
    wsum = txw.groupby(["household_key", "wk"])["sales_value"].sum().unstack()
    wks = np.arange((snap - 83) // 7, snap // 7 + 1)
    wsum = wsum.reindex(index=hh, columns=wks, fill_value=0.0)
    out["wk_mean_12"] = wsum.mean(axis=1); out["wk_std_12"] = wsum.std(axis=1).fillna(0.0)
    out["wk_max_12"] = wsum.max(axis=1); out["wk_zero_12"] = (wsum == 0).sum(axis=1).astype(float)
    m112 = d > snap - 112
    bd = tx[m112].groupby(["household_key", "basket_id"])["day"].min().sort_values()
    gaps = bd.groupby("household_key").diff(); gg = gaps.groupby("household_key")
    out["gap_mean_112"] = gg.mean().reindex(hh); out["gap_max_112"] = gg.max().reindex(hh)
    b84 = tx[m84].groupby(["household_key", "basket_id"])["day"].min()
    out["lead_gap_84"] = (b84.groupby("household_key").min() - (snap - 83)).reindex(hh)
    out["nstores_84"] = tx[m84].groupby("household_key")["store_id"].nunique().reindex(hh, fill_value=0)
    out["nlines_84"] = tx[m84].groupby("household_key").size().reindex(hh, fill_value=0)
    mb = tx[m84].groupby(["household_key", "basket_id"])["sales_value"].sum()
    out["max_basket_84"] = mb.groupby("household_key").max().reindex(hh, fill_value=0.0)
    out["r_seas"] = out["spend_28"] / (out["spend_seas_364"] + 5.0)
    out["r_mid"] = out["spend_28"] / (out["spend_mid_5_8"] + 5.0)
    out["r_7_28"] = out["spend_7"] * 4.0 / (out["spend_28"] + 5.0)
    out["seas_vs_84"] = out["spend_seas_364"] / (out["spend_84"] + 5.0)
    return out.reset_index()

new = agent_api.build_features(make_new)
agent_api.save_table(new, "e004_new.parquet")
print("new:", new.shape)

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
overlap = ["spend_28", "spend_84", "bask_28", "bask_84"]
full = fe.drop(columns=overlap).merge(new, on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape, "dups:", full.columns.duplicated().sum())
agent_api.save_table(full, "e004_features.parquet")

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]

def wmedian(y, w):
    o = np.argsort(y); y, w = np.asarray(y)[o], np.asarray(w)[o]
    return y[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

def fit_pred(Xtr, ytr, wtr, Xte, seed=0):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=800, learning_rate=0.05, max_depth=6,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(ytr, wtr)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return np.clip(m.predict(Xte), 0, None)

itr = (data.snapshot_day <= 403).values
ite = (data.snapshot_day == 431).values
y = data.future_spend_4w.values
w = 0.5 ** ((431 - data.snapshot_day.values) / 277.0)
p_old = fit_pred(data.loc[itr, old_cols], y[itr], w[itr], data.loc[ite, old_cols])
p_new = fit_pred(data.loc[itr, all_cols], y[itr], w[itr], data.loc[ite, all_cols])
yv = y[ite]
print("snap431 MAE old(E002):", np.abs(p_old - yv).mean())
print("snap431 MAE new:", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())