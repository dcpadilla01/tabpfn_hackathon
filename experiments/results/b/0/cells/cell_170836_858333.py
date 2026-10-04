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
Z = ((X-mu)/sd).clip(-5,5); Z['intercept']=1.0
Zv = Z.values
def mae(a,b): return np.abs(a-b).mean()
def fit(mask, lam=5.0):
    Zt = Zv[mask]; yt = y[mask]
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@yt)

days = sorted(df.snapshot_day[is_tr].unique())
print('OOF bias/MAE by eval day (train on all earlier days):')
for d in days[3:]:
    trm = is_tr & (df.snapshot_day < d).values
    vam = is_tr & (df.snapshot_day == d).values
    w = fit(trm)
    p = Zv[vam]@w
    r = y[vam]-p
    print(f"day {d}: n={vam.sum():4d} y_mean={y[vam].mean():6.1f} p_mean={p.mean():6.1f} bias={r.mean():6.1f} MAE={np.abs(r).mean():6.1f} zero_share={(y[vam]==0).mean():.3f}")

# zero vs nonzero target feature comparison (day 431 OOF preds)
d = 431
trm = is_tr & (df.snapshot_day < d).values
vam = is_tr & (df.snapshot_day == d).values
w = fit(trm)
t = df[vam].copy(); t['pred']=Zv[vam]@w
z, nz = t[t.future_spend_4w==0], t[t.future_spend_4w>0]
print('\nzero rows:', len(z), 'mean pred', round(z.pred.mean(),1), '| nonzero rows:', len(nz), 'mean pred', round(nz.pred.mean(),1))
key = ['spend28','spend7','recency','lag_zero_streak','lag1_is_zero','zero28','lag_nonzero_cnt_1_13','trend_7_28','trend_28_56','dec_spend14','trips28','actdays28','tenure','ratio28_364','active_weeks112']
print('\nfeature means: zero-target vs nonzero-target (day 431):')
cmp = pd.DataFrame({'zero': z[key].mean(), 'nonzero': nz[key].mean()})
cmp['ratio'] = (cmp.zero/cmp.nonzero).round(2)
print(cmp.round(1))
# how many zero-target rows have recency <= 14?
print('\nzero-target rows by recency bucket:')
print(pd.cut(z.recency, [-1,7,14,21,28,56,1000]).value_counts().sort_index())
print('\nnonzero-target rows by recency bucket:')
print(pd.cut(nz.recency, [-1,7,14,21,28,56,1000]).value_counts().sort_index())