
import agent_api, numpy as np, pandas as pd
# Look at validation rows: what does the actual future spend look like for val snapshots?
# We can't see beyond 459 via snapshot/history caps, but build_features sees up to snapshot day.
# Key question: is there a systematic per-day effect (e.g. week-of-year seasonality) we can exploit?
# Check: at train snapshots, target mean by snapshot day vs week_of_year of the FUTURE window.
allF = agent_api.load_saved("allF.parquet")
tt = agent_api.train_targets()
m = allF.merge(tt, on=["household_key","snapshot_day"])
# future window week: weeks (day+1..day+28) -> mean week index
m['fut_wk'] = ((m.snapshot_day + 15) + 8) // 7 % 52
g = m.groupby('fut_wk')['future_spend_4w'].agg(['mean','count'])
print(g.round(1).to_string())
# Also check spend_28 mean by future week at train (to see if level shift is household-driven)
