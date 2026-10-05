import agent_api, pandas as pd, numpy as np, time, xgboost as xgb
m = agent_api.load_saved("feats_v6.parquet")
sd = agent_api.snapshot_days()
tt = agent_api.train_targets()
tr = m[m.snapshot_day.isin(sd["train"])].merge(tt, on=["household_key","snapshot_day"])
va = m[m.snapshot_day.isin(sd["validation"])]
FEATS = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(FEATS))
Xtr, ytr = tr[FEATS].values, tr.future_spend_4w.values
Xva = va[FEATS].values
t0 = time.time()
preds = []
for seed in [0, 1]:
    mdl = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=seed)
    mdl.fit(Xtr, ytr); preds.append(mdl.predict(Xva))
    print("seed", seed, "done", round(time.time()-t0,1))
p = np.mean(preds, axis=0)
out = va[["household_key","snapshot_day"]].copy(); out["prediction"] = p
agent_api.save_table(out, "pred_seasonal.parquet")
print("saved", out.shape)
