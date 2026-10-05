
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
tr = m[m.snapshot_day<=431].copy()
# candidate: week-of-year effect on target beyond household features?
# Fit quick surrogate: residual of a simple model vs fut_wk
import sklearn.ensemble as ske
feats = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w_F','future_spend_4w_T','t','fut_wk')]
X = tr[feats].fillna(-1).values
y = tr.future_spend_4w_T.values
# simple: use spend_28 as base predictor, residual analysis
r = y - tr.spend_28.values
tr['fut_wk'] = ((tr.snapshot_day+15+8)//7) % 52
print(tr.groupby('fut_wk')['r'].agg(['mean','count']).round(2).to_string())
# residual by snapshot day
print("\nresid mean by snapshot_day:\n", tr.groupby('snapshot_day')['r'].mean().round(2).to_string())
