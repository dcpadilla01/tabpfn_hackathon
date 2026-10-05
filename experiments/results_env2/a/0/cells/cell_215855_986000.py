import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
va_days = [459,487,515,543]

Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w
Xva = f[f.snapshot_day.isin(va_days)]
print(Xtr.shape, Xes.shape, Xva.shape)

def fit_predict(objective, ytr_raw, params=None, n=1200, lr=0.03):
    p = dict(n_estimators=n, learning_rate=lr, max_depth=7, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective=objective,
             tree_method='hist', n_jobs=4)
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr_raw, eval_set=[(Xes, yes)], verbose=False)
    return m

# A: plain squared on raw (E004-style reference, quick check)
mA = fit_predict('reg:squarederror', ytr)
# B: squared on log1p
mB = fit_predict('reg:squarederror', np.log1p(ytr))
# C: pseudo-huber raw
mC = fit_predict('reg:pseudohubererror', ytr)

va_pred_A = np.clip(mA.predict(Xva[feat_cols]), 0, None)
va_pred_B = np.clip(np.expm1(mB.predict(Xva[feat_cols])), 0, None)
va_pred_C = np.clip(mC.predict(Xva[feat_cols]), 0, None)

# check on ES snapshots (375/403/431) as a proxy for val
esA = np.clip(mA.predict(Xes), 0, None); esB = np.clip(np.expm1(mB.predict(Xes)),0,None); esC = np.clip(mC.predict(Xes),0,None)
print("ES MAE  raw:", np.abs(esA-yes).mean().round(3), " log:", np.abs(esB-yes).mean().round(3), " huber:", np.abs(esC-yes).mean().round(3))
print("ES R2   raw:", 1-((esA-yes)**2).sum()/((yes-yes.mean())**2).sum(), " log:", 1-((esB-yes)**2).sum()/((yes-yes.mean())**2).sum(), " huber:", 1-((esC-yes)**2).sum()/((yes-yes.mean())**2).sum())
for w in [0.0,0.25,0.5,0.75,1.0]:
    print("blend w_log=",w, "ES MAE", np.abs(w*esB+(1-w)*esA-yes).mean().round(3))
