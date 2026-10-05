import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
p5 = load_saved('pred_e005.parquet')
print('pred_e005 shape', p5.shape, p5.columns.tolist())
print(p5.groupby('snapshot_day')['prediction'].agg(['mean','median','min','max']).round(1))
# train in-sample performance per day
pv = p5.merge(tt, on=KEYS)
pv['abs_err'] = (pv['prediction']-pv[TARGET]).abs()
print(pv.groupby('snapshot_day')[['abs_err']].mean().round(2))
# correlation of prediction with target
print('corr pred-target (train):', round(pv['prediction'].corr(pv[TARGET]),3))
# error decomposition: bias by target quantile
pv['tq'] = pd.qcut(pv[TARGET], 10, duplicates='drop')
print(pv.groupby('tq').agg(n=('abs_err','size'), mae=('abs_err','mean'), bias=('prediction', lambda s: None)).round(1))
pv['bias'] = pv['prediction'] - pv[TARGET]
print(pv.groupby('tq')['bias'].mean().round(1))
print(pv.groupby('tq')[TARGET].mean().round(1), pv.groupby('tq')['prediction'].mean().round(1))
