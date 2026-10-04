import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
allnan = [c for c in num.columns if num[c].notna().sum()==0]
print('all-nan cols:', allnan)
num = num.drop(columns=allnan)
X = num.fillna(num.median())
is_tr = df.future_spend_4w.notna().values
y = df.future_spend_4w.values
print('NaN in X:', X.isna().sum().sum())
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5)
print('NaN in Z:', Z.isna().sum().sum(), 'inf:', np.isinf(Z.values).sum())
Ztr, Zva, ytr = Z.values[is_tr], Z.values[~is_tr], y[is_tr]
def mae(a,b): return np.abs(a-b).mean()
def ridge_fit(Zt, yv, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yv)
w = ridge_fit(Ztr, ytr, 5.0)
pv = Zva@w
print('nan in pv:', np.isnan(pv).sum())
print('ridge raw-y val MAE:', round(mae(y[~is_tr], pv),3))
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
print('ridge log-y val MAE:', round(mae(y[~is_tr], np.expm1(Zva@w2)),3))
ptr = Ztr@w
tr_df = df[is_tr].copy(); tr_df['pred']=ptr; tr_df['res']=ytr-ptr
print('train MAE:', round(mae(ytr,ptr),3), 'bias:', round((ytr-ptr).mean(),3))
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print(tr_df.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nby snapshot_day:')
print(tr_df.groupby('snapshot_day').agg(n=('res','size'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero rows: pred mean', round(tr_df[tr_df.future_spend_4w==0].pred.mean(),2), 'MAE', round(tr_df[tr_df.future_spend_4w==0].res.abs().mean(),2))