import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
t11 = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner')
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
for c in feats:
    if m[c].dtype==object: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
X=m[feats].astype(float)
inf_cols=[c for c in feats if np.isinf(X[c].values).any()]
print('inf cols:', inf_cols)
X=X.replace([np.inf,-np.inf], np.nan)
# nan counts
nac=X.isna().mean().sort_values(ascending=False)
print('top nan:', dict(nac.head(6).round(3)))
Xv=X.values; y=m['future_spend_4w'].values; sd=m['snapshot_day'].values

# ---- aggregate weekly seasonality (capped view 459) ----
snap=A.snapshot()
tx=snap.table('transactions')
wk=(tx['day']+8)//7
g=tx.groupby(wk)['sales_value'].sum()
idx=np.arange(1, g.index.max()+1)
ser=g.reindex(idx).fillna(0).values
print('\nweeks:', len(ser))
# mean spend by week mod 52
ph=np.arange(len(ser))%52
prof=pd.Series(ser).groupby(ph).mean()
print('seasonal profile (week-mod-52): min=%.0f max=%.0f mean=%.0f, peak weeks:'%(prof.min(),prof.max(),prof.mean()), list(prof.nlargest(6).index), 'trough:', list(prof.nsmallest(4).index))
# lag-52 autocorr of weekly totals
s=ser-ser.mean(); ac52=np.corrcoef(s[:-52],s[52:])[0,1]; ac1=np.corrcoef(s[:-1],s[1:])[0,1]
print('autocorr lag52=%.3f lag1=%.3f'%(ac52,ac1))
# trend: yearly totals
yr=np.arange(len(ser))//52
print('total spend yr1=%.1fM yr2=%.1fM'%(pd.Series(ser).groupby(yr).sum().values[0]/1e6, pd.Series(ser).groupby(yr).sum().values[1] if len(pd.Series(ser).groupby(yr).sum())>1 else -1))