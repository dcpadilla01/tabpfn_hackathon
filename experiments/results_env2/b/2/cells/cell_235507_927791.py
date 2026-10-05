import agent_api, numpy as np, pandas as pd
t = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('rows', len(m), 'any NA rows', int(m.drop(columns=['future_spend_4w']).isna().any(axis=1).sum()))
na_frac = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).isna().mean().sort_values(ascending=False)
print(na_frac.head(15).round(3))
# offline ridge benchmark
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
va_days = [459,487,515,543]
X = m[feat_cols].astype(float).copy()
y = m['future_spend_4w'].values
is_va = m['snapshot_day'].isin(va_days).values
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
Xtr, ytr = Xs[~is_va], y[~is_va]
Xva, yva = Xs[is_va], y[is_va]
def ridge_fit(Xt, yt, lam):
    A = Xt.T@Xt + lam*np.eye(Xt.shape[1])
    return np.linalg.solve(A, Xt.T@yt)
for lam in [1,10,100,1000]:
    w = ridge_fit(Xtr,ytr,lam)
    pred = Xva@w
    print('lam',lam,'val MAE', round(float(np.abs(pred-yva).mean()),3))
