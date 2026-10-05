import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, obj, n, lr, md, mcw, extra=None, ret=False):
    p = dict(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective=obj,
        tree_method='hist', n_jobs=4)
    if extra: p.update(extra)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr, verbose=False)
    p_ = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p_-yes).mean(),3), "R2", round(1-((p_-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))
    return p_

p_hub = run("huber base", 'reg:pseudohubererror', 2500, .015, 7, 5)
p_q50 = run("quantile .5", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.5})
p_q45 = run("quantile .45", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.45})
p_q55 = run("quantile .55", 'reg:quantileerror', 2500, .015, 7, 5, {'quantile_alpha':0.55})
for w in [0.25,0.5,0.75]:
    bl = w*p_hub + (1-w)*p_q50
    print("blend hub/q50 w_hub=",w, "ES MAE", round(np.abs(bl-yes).mean(),3))
