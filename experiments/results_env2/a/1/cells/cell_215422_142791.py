
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].dropna(subset=["y"])
val = f[f.snapshot_day.isin([459,487,515,543])]
print("train rows:", len(tr), "val rows:", len(val))

w = np.exp(-(431 - tr.snapshot_day.values)/400.0)
m = xgb.XGBRegressor(n_estimators=1800, objective="reg:quantileerror", quantile_alpha=0.5,
                     learning_rate=0.02, max_depth=5, min_child_weight=5, subsample=0.8,
                     colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
m.fit(tr[Xcols], tr.y, sample_weight=w)
clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5, subsample=0.8,
                        colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8, eval_metric="logloss")
clf.fit(tr[Xcols], (tr.y==0).astype(int))
p = m.predict(val[Xcols])
p0 = clf.predict_proba(val[Xcols])[:,1]
p = p * (1 - np.minimum(p0, 0.9))
p = np.clip(p, 0, None)
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = p
print("pred stats: mean", round(p.mean(),1), "median", round(np.median(p),1), "zeros", round((p<1).mean(),3), "max", round(p.max(),1))
path = agent_api.save_table(out, "e003_preds.parquet")
print(path)
