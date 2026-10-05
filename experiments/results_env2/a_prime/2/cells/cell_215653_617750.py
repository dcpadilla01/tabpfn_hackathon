import pandas as pd, numpy as np, agent_api

for p in ['e005_decay_gapcv.parquet','e004_temporal.parquet','e001_recent_behavior.parquet']:
    df = agent_api.load_saved(p)
    print(p, df.shape)
    print(sorted([c for c in df.columns if c not in ('household_key','snapshot_day')]))
    print()

tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe())
print('zero frac train:', (tt.future_spend_4w==0).mean())
print(agent_api.snapshot_days())