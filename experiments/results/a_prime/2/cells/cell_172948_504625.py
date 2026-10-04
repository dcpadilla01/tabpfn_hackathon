import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print('n cols', len(cols))
import collections
pref = collections.Counter(c.split('_')[0] for c in cols)
print(dict(pref))

# check year-ago-style columns for train/val distribution mismatch
sus = [c for c in cols if 't13' in c or c.startswith('tlag_13') or 'lag_13' in c]
print('suspects:', sus)
for c in sus[:6]:
    g = df.groupby('snapshot_day')[c].agg(['mean','std'])
    print(c); print(g.tail(6).round(2)); print(g.head(3).round(2))

# fraction of rows with tlag_13==0 by period
for c in sus[:3]:
    z = (df[c]==0).groupby(df.snapshot_day<=347).mean()
    print(c,'zero-share train %.2f inner-val %.2f'% (z[True], z[False]))