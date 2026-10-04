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