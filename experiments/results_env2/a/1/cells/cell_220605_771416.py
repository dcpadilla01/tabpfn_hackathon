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