import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]
print(len(cols),'cols:')
print(cols)

# target by snapshot (drift check)
print('\ntrain target mean/median by snapshot:')
print(d.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']).round(1))
print('\nspend_28 mean by snapshot (all rows):')
print(df.groupby('snapshot_day').spend_28.mean().round(1) if 'spend_28' in df.columns else 'no spend_28 col')
