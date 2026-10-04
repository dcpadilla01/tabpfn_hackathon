import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
print('e3 shape', e3.shape)
print('e3 cols:', list(e3.columns))

tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape, 'missing target', int(df['future_spend_4w'].isna().sum()))
y = df['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share', float((y==0).mean()))

g = df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g)
print('std of snapshot means:', round(float(g['mean'].std()),2), 'overall mean:', round(float(y.mean()),2))

spend_cols = [c for c in e3.columns if 'spend' in c.lower()]
print('spend cols:', spend_cols)
def mae(p): return float((df[p]-y).abs().mean())
for c in spend_cols:
    print(c, 'MAE', round(mae(c),3), 'corr', round(float(df[c].corr(y)),3))
