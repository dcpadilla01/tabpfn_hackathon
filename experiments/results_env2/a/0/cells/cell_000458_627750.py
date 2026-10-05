import pandas as pd, numpy as np, agent_api, xgboost as xgb
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
# drift check
print('mean target by snapshot_day:')
print(df.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1))

def fit_q(Xtr, ytr, Xte, alpha=0.5, seed=7, n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                         max_depth=5, min_child_weight=40, learning_rate=0.08,
                         n_estimators=n, subsample=0.9, colsample_bytree=0.8,
                         n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr, ytr); return m.predict(Xte)

tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)

# A: baseline single q0.5
pa = fit_q(Xtr, ytr, Xte); print('A q0.5:', mae(pa))
# B: log-target median
pl = np.expm1(fit_q(Xtr, np.log1p(ytr), Xte)); print('B log-median:', mae(pl))
# C: avg of log-median and raw-median
print('C avg(raw,log):', mae((pa+pl)/2))
# D: seed bagging (5 seeds raw q0.5)
pb = np.mean([fit_q(Xtr, ytr, Xte, seed=s) for s in (1,2,3,4,5)], axis=0)
print('D seed-bag q0.5:', mae(pb))
# E: multi-quantile avg 0.45/0.5/0.55
pq = np.mean([fit_q(Xtr, ytr, Xte, alpha=a) for a in (0.45,0.5,0.55)], axis=0)
print('E multi-q avg:', mae(pq))
# F: recency-weighted training (weight 2^(day-95)/336)
w = 2.0**((df.snapshot_day.values[tr]-95)/336.0)
pw = fit_q(Xtr, ytr, Xte)
m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=5,
                     min_child_weight=40, learning_rate=0.08, n_estimators=400,
                     subsample=0.9, colsample_bytree=0.8, n_jobs=8, tree_method='hist')
m.fit(Xtr, ytr, sample_weight=w); print('F recency-wt:', mae(m.predict(Xte)))
# G: bag of (raw q0.5 + log) mixes
print('G avg of D and B:', mae((pb+pl)/2))
