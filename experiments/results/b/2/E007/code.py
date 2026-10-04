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

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e5 = A.load_saved('e005_longrun.parquet')
key = ['household_key','snapshot_day']
feats = [c for c in e5.columns if c not in key]
to_log, keep = [], []
for c in feats:
    v = e5[c].dropna().astype(float)
    if len(v) == 0 or v.min() < 0:
        keep.append(c); continue
    p50, p95 = v.quantile(.5), v.quantile(.95)
    (to_log if (p50 > 0 and p95/max(p50,1e-12) > 4) else keep).append(c)
print('n log-transformed:', len(to_log))
print(to_log)
print('kept as-is:', keep)
out = e5.copy()
for c in to_log:
    out[c] = np.log1p(out[c].astype(float))

# quick diagnostics for later ideas
print('\nx_disc_share_84:', e5.x_disc_share_84.describe().round(3).to_dict())
v = A.snapshot(459)
t = v.transactions
print('discount cols: frac>0 | mean | p95')
for c in ['coupon_match_disc','coupon_disc','retail_disc']:
    s = t[c]
    print(' ', c, round((s>0).mean(),3), round(s.mean(),2), round(s.quantile(.95),2))
print('sales_value sum check:', round(t.sales_value.sum(),0))
p = A.save_table(out, 'e007_log.parquet')
print('saved:', p, out.shape)