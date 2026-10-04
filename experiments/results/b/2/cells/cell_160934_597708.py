import agent_api, pandas as pd, numpy as np, datetime

t = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = df[feat_cols]
M = np.empty((len(df), len(feat_cols)), dtype=np.float64)
for j,c in enumerate(feat_cols):
    col = Xdf[c].values.astype(np.float64)
    col[~np.isfinite(col)] = np.nan
    M[:,j] = col
train_mask = df['future_spend_4w'].notna().values
y = df['future_spend_4w'].values.astype(np.float64)
hh = df['household_key'].values
uq = np.unique(hh); rng = np.random.RandomState(7); rng.shuffle(uq)
hh_hold = set(uq[:int(len(uq)*0.3)])
ho = np.array([h in hh_hold for h in hh]) & train_mask
tr = train_mask & ~ho
med = np.nanmedian(M[tr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
M = np.where(np.isnan(M), med[None,:], M)
q99 = np.quantile(np.abs(M[tr]), 0.999, axis=0)
M = np.clip(M, -q99[None,:], q99[None,:])
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
print('ols_all quantiles:', np.quantile(ols_all, [0,.01,.5,.99,1]).round(2))
resid_tr = y[tr]-ols_all[tr]
print('resid_tr quantiles:', np.quantile(resid_tr, [0,.01,.5,.99,1]).round(2))
print('y quantiles:', np.quantile(y[tr], [0,.5,.99,1]).round(2))
