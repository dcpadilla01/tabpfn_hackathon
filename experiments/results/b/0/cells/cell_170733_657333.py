import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
train = df[df.future_spend_4w.notna()].copy()
val = df[df.future_spend_4w.isna()].copy()
print('train rows', len(train), 'val rows', len(val))

feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feats].apply(pd.to_numeric, errors='coerce')
med = num.median()
X = num.fillna(med).clip(-1e6,1e6)
mu, sd = X.iloc[train.index].mean(), X.iloc[train.index].std().replace(0,1)
# careful: use positional split
is_tr = df.future_spend_4w.notna().values
Z = ((X - X.mean()) / X.std().replace(0,1)).values
Ztr, ytr = Z[is_tr], df.future_spend_4w.values[is_tr]
Zva = Z[~is_tr]

def ridge_fit(Zt, y, lam=1.0):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1])
    return np.linalg.solve(A, Zt.T@y)
def mae(y, p): return np.abs(y-p).mean()

w = ridge_fit(Ztr, ytr, 5.0)
pv = Zva@w
print('ridge raw-y val MAE:', round(mae(df.future_spend_4w.values[~is_tr], pv),3))

# log-target variant
w2 = ridge_fit(Ztr, np.log1p(ytr), 5.0)
pv2 = np.expm1(Zva@w2)
print('ridge log-y val MAE:', round(mae(df.future_spend_4w.values[~is_tr], pv2),3))

ptr = Ztr@w
res = ytr - ptr
print('\ntarget describe:', pd.Series(ytr).describe())
print('train MAE:', round(mae(ytr,ptr),3))
tr_df = train.copy(); tr_df['pred']=ptr; tr_df['res']=res
tr_df['b'] = pd.qcut(tr_df.spend28, 5, duplicates='drop')
print('\ntrain MAE by spend28 quintile:')
print(tr_df.groupby('b', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g.future_spend_4w.mean(),'p':g.pred.mean(),'mae':g.res.abs().mean(),'bias':g.res.mean()})))
print('\nMAE by snapshot_day:')
print(tr_df.groupby('snapshot_day').apply(lambda g: pd.Series({'n':len(g),'mae':g.res.abs().mean(),'bias':g.res.mean(),'y':g.future_spend_4w.mean()})))
print('\nzero-target share:', (ytr==0).mean())
z = tr_df[tr_df.future_spend_4w==0]
print('zero-target rows: pred mean', round(z.pred.mean(),2), 'MAE', round(z.res.abs().mean(),2), 'n', len(z))
nz = tr_df[tr_df.future_spend_4w>0]
print('pos-target rows: MAE', round(nz.res.abs().mean(),2))