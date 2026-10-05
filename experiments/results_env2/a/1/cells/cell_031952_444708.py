
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
tr = m[m.snapshot_day<=431].copy()
tr['r'] = tr.future_spend_4w_T - tr.spend_28
tr['fut_wk'] = ((tr.snapshot_day+15+8)//7) % 52
print(tr.groupby('fut_wk')['r'].agg(['mean','count']).round(2).to_string())
print("\nresid mean by snapshot_day:\n", tr.groupby('snapshot_day')['r'].mean().round(2).to_string())
# trend in target over snapshot day? fit linear
d = tr.groupby('snapshot_day')['r'].mean()
print("\nslope of resid vs day:", np.polyfit(d.index, d.values, 1)[0].round(4))
