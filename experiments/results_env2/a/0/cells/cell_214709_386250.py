import pandas as pd, numpy as np, agent_api, xgboost as xgb

f1 = agent_api.load_saved("feats_v1.parquet")
if "household_key" not in f1.columns: f1 = f1.reset_index()
tt = agent_api.train_targets()
df = f1.merge(tt, on=["household_key","snapshot_day"], how="left")
feat_cols = [c for c in f1.columns if c not in ("household_key","snapshot_day")]
tr_days = agent_api.snapshot_days()["train"]; va_days = agent_api.snapshot_days()["validation"]

def fit(dtr, params, seed=0):
    X = dtr[feat_cols].values.astype(np.float32); y = dtr.future_spend_4w.values
    m = xgb.XGBRegressor(**params, random_state=seed, n_jobs=8, tree_method="hist")
    m.fit(X, y); return m

def mae(m, d):
    p = np.clip(m.predict(d[feat_cols].values.astype(np.float32)), 0, None)
    return np.abs(p - d.future_spend_4w.values).mean()

base = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)
p_l2 = dict(base, objective="reg:squarederror")
p_l1 = dict(base, objective="reg:absoluteerror")

# internal sanity check: hold out last train snapshot 431
dtr_in = df[df.snapshot_day.isin(tr_days[:-1])]
dho = df[df.snapshot_day==431]
m2 = fit(dtr_in, p_l2); m1 = fit(dtr_in, p_l1)
print("holdout-431 MAE  L2: %.3f   L1: %.3f" % (mae(m2,dho), mae(m1,dho)))

# final: train on all train snapshots, predict validation
dtr = df[df.snapshot_day.isin(tr_days)]
m = fit(dtr, p_l1)
va = f1[f1.snapshot_day.isin(va_days)].copy()
va["prediction"] = np.clip(m.predict(va[feat_cols].values.astype(np.float32)), 0, None)
print("val pred stats:", va.prediction.describe()[["mean","50%","min","max"]].round(2).to_dict())
out = va[["household_key","snapshot_day","prediction"]]
path = agent_api.save_table(out, "pred_e003_l1")
print("saved:", path)
