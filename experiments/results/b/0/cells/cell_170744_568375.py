import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
bad = [c for c in feats if num[c].abs().max()>1e12 or num[c].isna().all()]
print('bad cols:', bad[:20], len(bad))
num = num[[c for c in feats if c not in bad]]
X = num.fillna(num.median())
is_tr = df.future_spend_4w.notna().values
y = df.future_spend_4w.values
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5).values
Ztr, Zva, ytr = Z[is_tr], Z[~is_tr], y[is_tr]

def ridge_fit(Zt, yv, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yv)
def mae(a,b): return np.abs(a-b).mean()

w = ridge_fit(Ztr, ytr, 5.0)
print('ridge raw-y val MAE:', round(mae(y[~is_tr], Zva@w),3))
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
print('ridge log-y val MAE:', round(mae(y[~is_tr], np.expm1(Zva@w2)),3))

ptr = Ztr@w
tr_df = df[is_tr].copy(); tr_df['pred']=ptr; tr_df['res']=ytr-ptr
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print('\ntrain MAE by spend28 quintile:')
print(tr_df.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nMAE by snapshot_day:')
print(tr_df.groupby('snapshot_day').agg(n=('res','size'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero-target share:', (ytr==0).mean())
z = tr_df[tr_df.future_spend_4w==0]
print('zero rows: pred mean', round(z.pred.mean(),2), 'MAE', round(z.res.abs().mean(),2), 'n', len(z))
print('pos rows MAE:', round(tr_df[tr_df.future_spend_4w>0].res.abs().mean(),2))