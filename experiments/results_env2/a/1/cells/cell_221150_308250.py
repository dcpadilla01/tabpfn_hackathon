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