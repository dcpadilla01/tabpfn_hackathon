import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values
m_tr = d<=403; m_va = d==431
mu = np.nanmean(Xv[m_tr],0)
print('mu nan count', np.isnan(mu).sum())
bad = [feat_cols[j] for j in range(len(feat_cols)) if np.isnan(mu[j])]
print('all-nan-in-train cols:', bad)
Xf = np.where(np.isnan(Xv), 0.0, Xv-mu)
sd = Xf[m_tr].std(0)+1e-9
print('sd nan', np.isnan(sd).sum(), 'sd zero', (sd<=1e-9).sum())
Xs = Xf/sd
print('Xs nan', np.isnan(Xs).sum(), 'inf', np.isinf(Xs).sum())
G = Xs[m_tr].T@Xs[m_tr]
print('G nan', np.isnan(G).sum(), 'G inf', np.isinf(G).sum())
print('max abs Xs', np.nanmax(np.abs(Xs[m_tr])))
big = [feat_cols[j] for j in range(len(feat_cols)) if np.nanmax(np.abs(Xs[m_tr][:,j]))>1e6]
print('huge cols:', big[:10])
