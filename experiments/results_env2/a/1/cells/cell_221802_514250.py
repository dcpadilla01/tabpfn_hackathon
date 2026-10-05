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