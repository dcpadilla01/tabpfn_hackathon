import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days).values; iva = df.snapshot_day.isin(iva_days).values
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr],0)
bad = [num_cols[i] for i in np.where(np.isnan(mu))[0]]
print('all-NaN-on-inner-train cols:', bad)
for c in bad:
    print(c, 'nan frac overall %.2f' % df[c].isna().mean(), 'by day:', df.groupby('snapshot_day')[c].apply(lambda s: s.isna().mean().round(2)).to_dict())
