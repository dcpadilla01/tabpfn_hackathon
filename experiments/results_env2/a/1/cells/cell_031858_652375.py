
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"], suffixes=('_F','_T'))
m['t'] = m.future_spend_4w_T
m['fut_wk'] = ((m.snapshot_day + 15) + 8) // 7 % 52
g = m.groupby('fut_wk')['t'].agg(['mean','count'])
print(g.round(1).to_string())
# spend_28 mean by fut_wk (household mix constant-ish)
g2 = m.groupby('fut_wk')['spend_28'].mean()
print(g2.round(1).to_string())
# ratio target/spend_28 by fut_wk
print((g['mean']/g2).round(3).to_string())
