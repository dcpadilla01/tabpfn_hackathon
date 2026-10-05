import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved('feats_v4.parquet')
print('feats', feats.shape)
print(list(feats.columns))
tt = agent_api.train_targets()
print('tt', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac train targets:', round((tt.future_spend_4w==0).mean(),4))
val = feats[feats.snapshot_day>=459]
print('val rows', len(val), 'hh', val.household_key.nunique())
tset = set(tt.household_key)
for s in sorted(val.snapshot_day.unique()):
    v = val[val.snapshot_day==s]
    print('day', s, 'rows', len(v), 'cov', round(v.household_key.isin(tset).mean(),3))
m = tt.groupby('household_key').future_spend_4w.agg(['mean','count'])
f2 = feats.merge(m, left_on='household_key', right_index=True, how='left')
cols = [c for c in feats.columns if feats[c].dtype.kind in 'fi']
cort = f2[cols+['mean']].corr()['mean'].drop('mean').sort_values(key=abs, ascending=False)
print('top corr with hh mean target:')
print(cort.head(15))
p = agent_api.load_saved('pred_e011.parquet')
print('pred_e011', p.shape, list(p.columns))
print(p.head(3))
