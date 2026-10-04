import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
X = num.fillna(num.median())
y = df.future_spend_4w.values
is_tr = df.future_spend_4w.notna().values
mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
Z = ((X-mu)/sd).clip(-5,5)
Z['intercept']=1.0
Zv = Z.values

# pseudo-val: train on snapshot days <= 403, eval on 431
tr_mask = is_tr & (df.snapshot_day<=403).values
va_mask = is_tr & (df.snapshot_day==431).values
def fit(mask, lam=5.0, log=False):
    Zt, yt = Zv[mask], (np.log1p(y) if log else y)[mask]
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yt)
def mae(a,b): return np.abs(a-b).mean()
w = fit(tr_mask)
ptr, pva = Zv[tr_mask]@w, Zv[va_mask]@w
print('Pseudo-val (day 431) MAE:', round(mae(y[va_mask], pva),3))
print('train(<=403) MAE:', round(mae(y[tr_mask], ptr),3))
res = y[tr_mask]-ptr
t = df[tr_mask].copy(); t['pred']=ptr; t['res']=res
print('bias:', round(res.mean(),2))
t['b'] = pd.qcut(t.spend28, 5, duplicates='drop')
print(t.groupby('b', observed=True).agg(n=('res','size'), y=('future_spend_4w','mean'), p=('pred','mean'), mae=('res', lambda s: s.abs().mean()), bias=('res','mean')))
print('\nzero-target rows: n', (t.future_spend_4w==0).sum(), 'pred mean', round(t[t.future_spend_4w==0].pred.mean(),2))
v = df[va_mask].copy(); v['pred']=pva; v['res']=y[va_mask]-pva
print('\nday431: bias', round(v.res.mean(),2), 'MAE', round(v.res.abs().mean(),2))
# top ridge coefficients
coef = pd.Series(w[:-1], index=num.columns)
print('\nTop +coef:', coef.nlargest(12).round(2).to_dict())
print('Top -coef:', coef.nsmallest(12).round(2).to_dict())
# log-target variant with smearing
w2 = fit(tr_mask, log=True)
pva2 = np.expm1(Zv[va_mask]@w2)
ptr2 = np.expm1(Zv[tr_mask]@w2)
smear = (y[tr_mask]/np.clip(ptr2,1e-3,None)).mean()
print('\nlog-target pseudo-val MAE:', round(mae(y[va_mask], pva2),3), 'with smear:', round(mae(y[va_mask], pva2*smear),3))