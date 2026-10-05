f = agent_api.load_saved("e002_features.parquet")
print(f.shape)
print(list(f.columns))
t = agent_api.train_targets()
print(t.shape)
print(t.future_spend_4w.describe())
print("zero share:", (t.future_spend_4w == 0).mean())
print(agent_api.snapshot_days())

# ---- cell ----
import numpy as np, pandas as pd

# ---------- new-feature function ----------
def make_new(view, snap):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx["day"].values
    out = pd.DataFrame(index=hh)

    def agg(mask):
        sub = tx[mask]
        g = sub.groupby("household_key")
        s = g["sales_value"].sum().reindex(hh, fill_value=0.0)
        b = g["basket_id"].nunique().reindex(hh, fill_value=0)
        a = g["day"].nunique().reindex(hh, fill_value=0)
        return s, b, a

    for w in [7, 14]:
        s, b, a = agg(d > snap - w)
        out[f"spend_{w}"] = s; out[f"bask_{w}"] = b; out[f"act_{w}"] = a

    # phase-of-cycle windows (weeks 5-8 and 9-12 back)
    for lo, hi, name in [(snap-56, snap-28, "mid_5_8"), (snap-84, snap-56, "mid_9_12")]:
        s, _, _ = agg((d > lo) & (d <= hi))
        out[f"spend_{name}"] = s

    # seasonally aligned windows: target window (snap, snap+28] shifted back by k days
    for k in [336, 364, 392]:
        s, _, _ = agg((d > snap - k) & (d <= snap - k + 28))
        out[f"spend_seas_{k}"] = s

    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    out["seas_ok"] = (fd <= snap - 392).astype(float)

    # daily stats, last 84d
    m84 = d > snap - 84
    daily = tx[m84].groupby(["household_key", "day"])["sales_value"].sum()
    g2 = daily.groupby("household_key")
    out["daily_mean_84"] = g2.mean().reindex(hh, fill_value=0.0)
    out["daily_std_84"] = g2.std().reindex(hh).fillna(0.0)
    out["daily_max_84"] = g2.max().reindex(hh, fill_value=0.0)

    # weekly sums, last 12 weeks (zero-filled)
    txw = tx[m84].copy(); txw["wk"] = txw["day"] // 7
    wsum = txw.groupby(["household_key", "wk"])["sales_value"].sum().unstack()
    wks = np.arange((snap - 83) // 7, snap // 7 + 1)
    wsum = wsum.reindex(index=hh, columns=wks, fill_value=0.0)
    out["wk_mean_12"] = wsum.mean(axis=1)
    out["wk_std_12"] = wsum.std(axis=1).fillna(0.0)
    out["wk_max_12"] = wsum.max(axis=1)
    out["wk_zero_12"] = (wsum == 0).sum(axis=1).astype(float)

    # basket-gap stats
    m112 = d > snap - 112
    bd = tx[m112].groupby(["household_key", "basket_id"])["day"].min().sort_values()
    gaps = bd.groupby("household_key").diff()
    gg = gaps.groupby("household_key")
    out["gap_mean_112"] = gg.mean().reindex(hh)
    out["gap_max_112"] = gg.max().reindex(hh)
    b84 = tx[m84].groupby(["household_key", "basket_id"])["day"].min()
    out["lead_gap_84"] = (b84.groupby("household_key").min() - (snap - 83)).reindex(hh)

    # structure
    out["nstores_84"] = tx[m84].groupby("household_key")["store_id"].nunique().reindex(hh, fill_value=0)
    out["nlines_84"] = tx[m84].groupby("household_key").size().reindex(hh, fill_value=0)
    mb = tx[m84].groupby(["household_key", "basket_id"])["sales_value"].sum()
    out["max_basket_84"] = mb.groupby("household_key").max().reindex(hh, fill_value=0.0)

    # ratios
    out["r_seas"] = out["spend_28"] / (out["spend_seas_364"] + 5.0)
    out["r_mid"] = out["spend_28"] / (out["spend_mid_5_8"] + 5.0)
    out["r_7_28"] = out["spend_7"] * 4.0 / (out["spend_28"] + 5.0)
    out["seas_vs_84"] = out["spend_seas_364"] / (out["spend_84"] + 5.0)
    return out

new = agent_api.build_features(make_new)
print("new feats:", new.shape)

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
full = fe.merge(new.reset_index(), on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape)

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]

import xgboost as xgb
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

# internal check: train on snapshots <= 403, evaluate on 431
itr = data.snapshot_day <= 403
ite = data.snapshot_day == 431
y = data.future_spend_4w.values
w = 0.5 ** ((431 - data.snapshot_day.values) / 277.0)

p_old = fit_pred(data.loc[itr, old_cols], y[itr.values], w[itr.values], data.loc[ite, old_cols])
p_new = fit_pred(data.loc[itr, all_cols], y[itr.values], w[itr.values], data.loc[ite, all_cols])
yv = y[ite.values]
print("snap431 MAE old(E002):", np.abs(p_old - yv).mean())
print("snap431 MAE new(E002+new):", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())

# ---- cell ----
import numpy as np, pandas as pd

def make_new(view, snap):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx["day"].values
    out = pd.DataFrame(index=hh)

    def agg(mask):
        sub = tx[mask]
        g = sub.groupby("household_key")
        s = g["sales_value"].sum().reindex(hh, fill_value=0.0)
        b = g["basket_id"].nunique().reindex(hh, fill_value=0)
        a = g["day"].nunique().reindex(hh, fill_value=0)
        return s, b, a

    for w in [7, 14, 28]:
        s, b, a = agg(d > snap - w)
        out[f"spend_{w}"] = s; out[f"bask_{w}"] = b; out[f"act_{w}"] = a

    for lo, hi, name in [(snap-56, snap-28, "mid_5_8"), (snap-84, snap-56, "mid_9_12")]:
        s, _, _ = agg((d > lo) & (d <= hi))
        out[f"spend_{name}"] = s

    for k in [336, 364, 392]:
        s, _, _ = agg((d > snap - k) & (d <= snap - k + 28))
        out[f"spend_seas_{k}"] = s

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
    out["wk_mean_12"] = wsum.mean(axis=1)
    out["wk_std_12"] = wsum.std(axis=1).fillna(0.0)
    out["wk_max_12"] = wsum.max(axis=1)
    out["wk_zero_12"] = (wsum == 0).sum(axis=1).astype(float)

    m112 = d > snap - 112
    bd = tx[m112].groupby(["household_key", "basket_id"])["day"].min().sort_values()
    gaps = bd.groupby("household_key").diff()
    gg = gaps.groupby("household_key")
    out["gap_mean_112"] = gg.mean().reindex(hh)
    out["gap_max_112"] = gg.max().reindex(hh)
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
    return out

new = agent_api.build_features(make_new)
print("new feats:", new.shape)

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
full = fe.merge(new.reset_index(), on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape)

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
agent_api.save_table(full, "e004_features.parquet")

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]

import xgboost as xgb
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
print("snap431 MAE new(E002+new):", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())

# ---- cell ----
import numpy as np, pandas as pd

def make_new(view, snap):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx["day"].values
    out = pd.DataFrame(index=hh)

    def agg(mask):
        sub = tx[mask]
        g = sub.groupby("household_key")
        s = g["sales_value"].sum().reindex(hh, fill_value=0.0)
        b = g["basket_id"].nunique().reindex(hh, fill_value=0)
        a = g["day"].nunique().reindex(hh, fill_value=0)
        return s, b, a

    for w in [7, 14, 28, 84]:
        s, b, a = agg(d > snap - w)
        out[f"spend_{w}"] = s; out[f"bask_{w}"] = b; out[f"act_{w}"] = a

    for lo, hi, name in [(snap-56, snap-28, "mid_5_8"), (snap-84, snap-56, "mid_9_12")]:
        s, _, _ = agg((d > lo) & (d <= hi))
        out[f"spend_{name}"] = s

    for k in [336, 364, 392]:
        s, _, _ = agg((d > snap - k) & (d <= snap - k + 28))
        out[f"spend_seas_{k}"] = s

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
    out["wk_mean_12"] = wsum.mean(axis=1)
    out["wk_std_12"] = wsum.std(axis=1).fillna(0.0)
    out["wk_max_12"] = wsum.max(axis=1)
    out["wk_zero_12"] = (wsum == 0).sum(axis=1).astype(float)

    m112 = d > snap - 112
    bd = tx[m112].groupby(["household_key", "basket_id"])["day"].min().sort_values()
    gaps = bd.groupby("household_key").diff()
    gg = gaps.groupby("household_key")
    out["gap_mean_112"] = gg.mean().reindex(hh)
    out["gap_max_112"] = gg.max().reindex(hh)
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
    return out

new = agent_api.build_features(make_new)
print("new feats:", new.shape)

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
full = fe.merge(new.reset_index(), on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape)

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
agent_api.save_table(full, "e004_features.parquet")

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]

import xgboost as xgb
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
print("snap431 MAE new(E002+new):", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
new = agent_api.load_saved("e004_features.parquet")
# new table was saved AFTER the merge attempt? check columns
print("e004 cols sample:", [c for c in new.columns if c.startswith("spend")][:10])
print(new.shape)

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
new = agent_api.load_saved("e004_features.parquet")

# keep only the new columns (those with _x/_y suffixes plus genuinely new ones)
new_only = new[["household_key", "snapshot_day"] + [c for c in new.columns if c.endswith("_y") or c.endswith("_x")]]
rename = {c: c[:-2] for c in new_only.columns if c.endswith("_x") or c.endswith("_y")}
new_only = new_only.rename(columns=rename)
# drop the old columns from fe that are superseded, then merge
fe2 = fe.drop(columns=[c for c in rename.values() if c in fe.columns])
full = fe2.merge(new_only, on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape)
agent_api.save_table(full, "e004_features.parquet")

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
print("n old:", len(old_cols), "n all:", len(all_cols))

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
print("snap431 MAE new(E002+new):", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
new = agent_api.load_saved("e004_features.parquet")

overlap = {"spend_28", "spend_84", "bask_28", "bask_84"}
new_cols = []
for c in new.columns:
    if c in ("household_key", "snapshot_day", "index"):
        continue
    if c.endswith("_y"):
        base = c[:-2]
        if base in overlap:
            new_cols.append(c)  # keep new version
    elif c.endswith("_x") or c in overlap:
        continue
    else:
        new_cols.append(c)
sel = new[["household_key", "snapshot_day"] + new_cols].rename(
    columns={c: c[:-2] for c in new_cols if c.endswith("_y")})
fe2 = fe.drop(columns=list(overlap))
full = fe2.merge(sel, on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape, "dup:", full.columns.duplicated().sum())
agent_api.save_table(full, "e004_features.parquet")

tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)

old_cols = [c for c in fe.columns if c not in ("household_key", "snapshot_day")]
all_cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
print("n old:", len(old_cols), "n all:", len(all_cols))

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
print("snap431 MAE new(E002+new):", np.abs(p_new - yv).mean())
print("n val rows snap431:", ite.sum())

# ---- cell ----
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

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
y = data.future_spend_4w.values
sd = data.snapshot_day.values

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

def mk(**kw):
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=800,
             learning_rate=0.05, max_depth=6, min_child_weight=10, subsample=0.8,
             colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0)
    p.update(kw); return xgb.XGBRegressor(**p)

def run(train_mask, val_mask, variant):
    w = 0.5 ** ((431 - sd[train_mask]) / 277.0)
    Xtr, Xte = data.loc[train_mask, cols], data.loc[val_mask, cols]
    ytr = y[train_mask]
    if variant == "base":
        m = mk(base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    elif variant == "snapday":
        Xtr2 = Xtr.copy(); Xtr2["snap"] = sd[train_mask]
        Xte2 = Xte.copy(); Xte2["snap"] = sd[val_mask]
        m = mk(base_score=float(wmedian(ytr, w)))
        m.fit(Xtr2, ytr, sample_weight=w); p = m.predict(Xte2)
    elif variant == "log":
        m = mk(base_score=float(np.median(np.log1p(ytr))))
        m.fit(Xtr, np.log1p(ytr), sample_weight=w); p = np.expm1(m.predict(Xte))
    elif variant == "twopart":
        z = (ytr > 0).astype(int)
        mc = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                               n_jobs=-1, random_state=0)
        mc.fit(Xtr, z, sample_weight=w)
        pos = ytr > 0
        m = mk(base_score=float(wmedian(ytr[pos], w[pos])))
        m.fit(Xtr[pos], ytr[pos], sample_weight=w[pos])
        p = m.predict(Xte) * mc.predict_proba(Xte)[:, 1]
    elif variant == "deep":
        m = mk(max_depth=8, min_child_weight=20, n_estimators=1200, learning_rate=0.04,
               base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    elif variant == "shallow":
        m = mk(max_depth=4, n_estimators=1500, learning_rate=0.04,
               base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    return np.clip(p, 0, None)

for vs in [(403,), (431,), (403, 431)]:
    tr = data.snapshot_day <= (vs[0] - 28)
    va = data.snapshot_day.isin(vs)
    yv = y[va.values]
    line = [f"val{vs}"]
    for variant in ["base", "snapday", "log", "twopart", "deep", "shallow"]:
        p = run(tr.values, va.values, variant)
        line.append(f"{variant}={np.abs(p - yv).mean():.2f}")
    print("  ".join(line))
# naive: predict spend_28
for vs in [(403,), (431,)]:
    va = data.snapshot_day.isin(vs)
    print(f"naive spend_28 val{vs}:", np.abs(data.loc[va, "spend_28"].values - y[va.values]).mean())

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
y = data.future_spend_4w.values
sd = data.snapshot_day.values

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

def fit(train_mask, val_mask, hl, seed=0, blend=0.0):
    ref = 543  # mimic final: weight rel to last train snap
    w = 0.5 ** ((sd[train_mask].max() - sd[train_mask]) / hl)
    Xtr, Xte = data.loc[train_mask, cols], data.loc[val_mask, cols]
    ytr = y[train_mask]
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=1500, learning_rate=0.04, max_depth=4,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(ytr, w)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=w)
    p = np.clip(m.predict(Xte), 0, None)
    if blend > 0:
        p = (1 - blend) * p + blend * data.loc[val_mask, "spend_28"].values
    return p

va = data.snapshot_day.isin([403, 431]).values
yv = y[va]
tr = (data.snapshot_day <= 375).values
for hl in [277, 400, 600, 10000]:
    p = fit(tr, va, hl)
    print(f"hl={hl}: MAE={np.abs(p-yv).mean():.2f}")
p = fit(tr, va, 277, blend=0.3)
print("blend0.3:", np.abs(p - yv).mean())
# ensemble of seeds
ps = [fit(tr, va, 277, seed=s) for s in [0, 1, 2, 3]]
print("seed-ens:", np.abs(np.mean(ps, axis=0) - yv).mean())

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
y = data.future_spend_4w.values
sd = data.snapshot_day.values

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

tr = data.snapshot_day <= 431
w = 0.5 ** ((431 - sd[tr.values]) / 600.0)
Xtr = data.loc[tr.values, cols]
preds = []
for seed in [0, 1, 2, 3]:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=1500, learning_rate=0.04, max_depth=4,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(y[tr.values], w)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, y[tr.values], sample_weight=w)
    preds.append(np.clip(m.predict(data.loc[~tr.values, cols]), 0, None))
p = np.mean(preds, axis=0)

out = data.loc[~tr.values, ["household_key", "snapshot_day"]].copy()
out["prediction"] = p
agent_api.save_table(out, "e004_preds.parquet")
print("saved", out.shape)

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
y = data.future_spend_4w.values
sd = data.snapshot_day.values

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

mask_tr = (sd <= 431)
mask_va = ~mask_tr
w = 0.5 ** ((431 - sd[mask_tr]) / 600.0)
Xtr = data.loc[mask_tr, cols].values.astype(np.float32)
Xva = data.loc[mask_va, cols].values.astype(np.float32)
preds = []
for seed in [0, 1, 2, 3]:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=1500, learning_rate=0.04, max_depth=4,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(y[mask_tr], w)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, y[mask_tr], sample_weight=w)
    preds.append(np.clip(m.predict(Xva), 0, None))
p = np.mean(preds, axis=0)

out = pd.DataFrame({
    "household_key": data.loc[mask_va, "household_key"].to_numpy(),
    "snapshot_day": data.loc[mask_va, "snapshot_day"].to_numpy(),
    "prediction": p,
})
print(out.shape, out.head(3))
agent_api.save_table(out, "e004_preds.parquet")
print("saved")

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
y = data["future_spend_4w"].to_numpy()
sd = data["snapshot_day"].to_numpy()
hk = data["household_key"].to_numpy()

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

mask_tr = (sd <= 431)
mask_va = ~mask_tr
w = 0.5 ** ((431 - sd[mask_tr]) / 600.0)
Xtr = data.loc[mask_tr, cols].to_numpy(dtype=np.float32)
Xva = data.loc[mask_va, cols].to_numpy(dtype=np.float32)
preds = []
for seed in [0, 1, 2, 3]:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=1500, learning_rate=0.04, max_depth=4,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(y[mask_tr], w)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, y[mask_tr], sample_weight=w)
    preds.append(np.clip(m.predict(Xva), 0, None))
p = np.mean(preds, axis=0).ravel()
print("p shape:", p.shape, "hk:", hk[mask_va].shape, "sd:", sd[mask_va].shape)

out = pd.DataFrame({
    "household_key": hk[mask_va],
    "snapshot_day": sd[mask_va],
    "prediction": p,
})
print(out.shape)
print(out.head(3))
agent_api.save_table(out, "e004_preds.parquet")
print("saved")

# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
y = data["future_spend_4w"].to_numpy()
sd_tr = data["snapshot_day"].to_numpy()

va = full[full["snapshot_day"].isin([459, 487, 515, 543])].copy()
print("val rows:", va.shape)

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

w = 0.5 ** ((431 - sd_tr) / 600.0)
Xtr = data[cols].to_numpy(dtype=np.float32)
Xva = va[cols].to_numpy(dtype=np.float32)
preds = []
for seed in [0, 1, 2, 3]:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                         n_estimators=1500, learning_rate=0.04, max_depth=4,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", base_score=float(wmedian(y, w)),
                         n_jobs=-1, random_state=seed)
    m.fit(Xtr, y, sample_weight=w)
    preds.append(np.clip(m.predict(Xva), 0, None))
p = np.mean(preds, axis=0).ravel()

out = pd.DataFrame({
    "household_key": va["household_key"].to_numpy(),
    "snapshot_day": va["snapshot_day"].to_numpy(),
    "prediction": p,
})
print(out.shape)
print(out.head(3))
agent_api.save_table(out, "e004_preds.parquet")
print("saved")