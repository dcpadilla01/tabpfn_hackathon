
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
# per-day bias of e005 preds on train? we only have val preds for e005. Use repro_e5 (p13,p11) also val-only.
# Instead: train-side analysis. Correlation of features with target, and per-day target stats.
g = m.groupby("snapshot_day")['future_spend_4w'].agg(['mean','median','count'])
print(g.round(2))
# overall: how much of target is 0?
print("zero share train:", (m.future_spend_4w==0).mean().round(4))
# weekly seasonality: is target related to week_of_year?
m['wk'] = ((m.snapshot_day+8)//7) % 52
print(m.groupby('wk')['future_spend_4w'].mean().round(1).to_string())
