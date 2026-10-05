import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
tr_days = [95,123,151,179,207,235,263,291,319,347]
es_days = [375,403,431]
Xtr, ytr = df[df.snapshot_day.isin(tr_days)][feat_cols], df[df.snapshot_day.isin(tr_days)].future_spend_4w.values
Xes, yes = df[df.snapshot_day.isin(es_days)][feat_cols], df[df.snapshot_day.isin(es_days)].future_spend_4w.values

def run(tag, n, lr):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective='reg:quantileerror',
        quantile_alpha=0.5, tree_method='hist', n_jobs=4)
    m.fit(Xtr, ytr, verbose=False)
    p_ = np.clip(m.predict(Xes), 0, None)
    print(tag, "ES MAE", round(np.abs(p_-yes).mean(),3), "R2", round(1-((p_-yes)**2).sum()/((yes-yes.mean())**2).sum(),4))

run("q50 n1200 lr.03 (E004-matched)", 1200, .03)
run("q50 n2500 lr.015", 2500, .015)
