import agent_api as A
import pandas as pd, numpy as np

print(A.snapshot_days())
tt = A.train_targets()
print('targets shape', tt.shape)
y = tt.future_spend_4w
print(y.describe(percentiles=[.05,.25,.5,.75,.9,.95,.99]).round(1))
print('zero frac %.3f' % (y==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median']).round(1))

e5 = A.load_saved('e005_longrun.parquet')
print('e5', e5.shape)
for c in e5.columns: print(' ', c, e5[c].dtype)

e1 = A.load_saved('e001_recent_spend.parquet')
print('e1 cols:', list(e1.columns))
print('e1 cols missing from e5:', [c for c in e1.columns if c not in e5.columns])

v = A.snapshot(459)
print('transactions rows (<=459):', v.transactions.shape)

m = e5.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
numcols = [c for c in e5.columns if pd.api.types.is_numeric_dtype(e5[c])]
nonnum = [c for c in e5.columns if c not in numcols]
print('non-numeric cols:', nonnum)
cor = m[numcols].corrwith(m.future_spend_4w)
order = cor.abs().sort_values(ascending=False).index
print(cor.reindex(order).round(3).to_string())
print('top NaN fracs:')
print(e5[numcols].isna().mean().sort_values(ascending=False).head(10).round(3))