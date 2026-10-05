
import agent_api, pandas as pd, numpy as np, time, xgboost as xgb
feats = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
train = feats.merge(tt, on=["household_key","snapshot_day"])
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype)=="category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

configs = [(4,20),(5,40),(6,20),(5,60)]
t0 = time.time()
rows = []
for s in sorted(train.snapshot_day.unique()):
    tr = train[train.snapshot_day != s]
    Xtr, ytr = prep(tr), tr.future_spend_4w.values
    va = train[train.snapshot_day == s]
    Xva = prep(va)
    preds = []
    for d,w in configs:
        for seed in (7,13):
            mdl = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                                   learning_rate=0.08, n_estimators=400, n_jobs=4,
                                   tree_method="hist", enable_categorical=True,
                                   random_state=seed, max_depth=d, min_child_weight=w)
            mdl.fit(Xtr, ytr)
            preds.append(mdl.predict(Xva))
    p = np.mean(preds, axis=0)
    rows.append(pd.DataFrame({"household_key": va.household_key.values, "snapshot_day": s,
                              "oof": p, "y": va.future_spend_4w.values}))
oof = pd.concat(rows, ignore_index=True)
print("OOF done %.0fs" % (time.time()-t0))
print("OOF MAE %.3f" % np.abs(oof.oof - oof.y).mean())
agent_api.save_table(oof, "oof_e013.parquet")
