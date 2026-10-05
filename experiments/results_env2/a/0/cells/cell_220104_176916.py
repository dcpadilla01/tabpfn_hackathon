import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, obj, n, lr, md, mcw, subs=0.8, col=0.7, lam=1.0):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=subs, colsample_bytree=col, reg_lambda=lam, objective=obj,
        tree_method='hist', n_jobs=4)
    m.fit(Xtr, ytr, verbose=False)
    p = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p-yes).mean(),3), "R2", round(1-((p-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))

run("huber n1200 lr.03 d7", 'reg:pseudohubererror', 1200, .03, 7, 5)
run("huber n2500 lr.015 d7", 'reg:pseudohubererror', 2500, .015, 7, 5)
run("huber n2500 lr.015 d6", 'reg:pseudohubererror', 2500, .015, 6, 5)
run("huber n2500 lr.015 d8", 'reg:pseudohubererror', 2500, .015, 8, 5)
run("huber n2500 lr.015 d7 mcw20", 'reg:pseudohubererror', 2500, .015, 7, 20)
run("quant.5 n2500 lr.015 d7", 'reg:quantileerror', 2500, .015, 7, 5, params=None) if False else None
run("quant.5 n2500 lr.015 d7", 'reg:quantileerror', 2500, .015, 7, 5)
