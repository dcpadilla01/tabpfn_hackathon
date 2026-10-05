import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
p8 = A.load_saved("pred_e008.parquet")
train_days = A.snapshot_days()["train"]; val_days = A.snapshot_days()["validation"]
feat_cols = [c for c in f3.columns if c not in ("household_key","snapshot_day")]
tr = tt.merge(f3, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
val = f3[f3.snapshot_day.isin(val_days)].sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
print("train rows", len(tr), "val rows", len(val))

def fit_two_stage(X, y, seed):
    ypos = (y>0).astype(int).values if hasattr(y,"values") else (y>0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=seed, eval_metric="logloss")
    clf.fit(X, ypos)
    reg = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=1200, learning_rate=0.03,
        max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=seed)
    m = ypos.astype(bool)
    reg.fit(X[m], np.asarray(y)[m])
    return clf, reg

def pred_two_stage(clfs, regs, X):
    ps = np.mean([c.predict_proba(X)[:,1] for c in clfs], axis=0)
    vs = np.mean([np.clip(r.predict(X),0,None) for r in regs], axis=0)
    return ps*vs, ps, vs

# ---- OOF check on last-4 train snapshots (same split as oof_e008) ----
fit = tr[tr.snapshot_day < 347]; hold = tr[tr.snapshot_day >= 347].reset_index(drop=True)
clfs, regs = [], []
for seed in (7, 42):
    c, r = fit_two_stage(fit[feat_cols].values, fit["future_spend_4w"].values, seed)
    clfs.append(c); regs.append(r)
stageA, p, v = pred_two_stage(clfs, regs, hold[feat_cols].values)
y = hold["future_spend_4w"].values
h = hold[["household_key","snapshot_day"]].merge(oo, on=["household_key","snapshot_day"])
base_med = h["oof_med"].values; base_mix = 0.5*(h["oof_sq"].values+h["oof_med"].values)
def mae(pr): return round(float(np.mean(np.abs(np.asarray(pr)-y))),3)
print("OOF holdout MAE (n=%d):"%len(y))
print("  oof_med", mae(base_med), "| oof_sq", mae(h["oof_sq"].values), "| oof_log", mae(h["oof_log"].values))
cands = {
 "stageA": stageA,
 "stageB_hard0": np.where(p<0.5, 0.0, v),
 "blendA50_med": 0.5*stageA+0.5*base_med,
 "blendA30_med": 0.3*stageA+0.7*base_med,
 "blendA70_med": 0.7*stageA+0.3*base_med,
 "blendA50_mix": 0.5*stageA+0.5*base_mix,
}
for k,pr in cands.items(): print("  %-14s"%k, mae(pr))
best = min(cands, key=lambda k: mae(cands[k]))
print("BEST OOF variant:", best)

# ---- final: train on ALL train snapshots, predict validation ----
clfsF, regsF = [], []
for seed in (7, 42):
    c, r = fit_two_stage(tr[feat_cols].values, tr["future_spend_4w"].values, seed)
    clfsF.append(c); regsF.append(r)
stageA_v, p_v, v_v = pred_two_stage(clfsF, regsF, val[feat_cols].values)
pv8 = val[["household_key","snapshot_day"]].merge(p8, on=["household_key","snapshot_day"], how="left")["prediction"].values
print("val pred_e008 aligned NaNs:", int(np.isnan(pv8).sum()))
vmap = {
 "stageA": stageA_v,
 "stageB_hard0": np.where(p_v<0.5, 0.0, v_v),
 "blendA50_med": 0.5*stageA_v+0.5*pv8,
 "blendA30_med": 0.3*stageA_v+0.7*pv8,
 "blendA70_med": 0.7*stageA_v+0.3*pv8,
 "blendA50_mix": 0.5*stageA_v+0.5*pv8,
}
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = vmap[best]
print("saved variant:", best, "| val pred mean", round(out.prediction.mean(),2), "| rows", len(out))
path = A.save_table(out, "pred_e009.parquet")
print(path)
