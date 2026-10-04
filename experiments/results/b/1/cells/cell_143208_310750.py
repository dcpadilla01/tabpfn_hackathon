import pandas as pd, numpy as np
try:
    import agent_api as A
except Exception:
    A = None
df = A.load_saved('e001_history.parquet')
print('e001 shape', df.shape)
print(df.columns.tolist())
v = A.snapshot(459)
print('households type:', type(v.households))
hh = v.households
print(np.asarray(hh)[:5], 'n=', len(hh))
print('day', v.day, 'week', v.week)
p = v.products
print('products', p.shape)
print(p['department'].value_counts().head(15))
print(p['brand'].value_counts(dropna=False).head())
t = v.transactions
print('tx', t.shape, 'maxday', t.day.max())
print(t[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc','trans_time']].describe())
print(A.snapshot_days())
tt = A.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())