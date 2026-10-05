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