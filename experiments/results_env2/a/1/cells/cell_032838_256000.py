import pandas as pd, numpy as np
for t in ['e004_new.parquet','e005_newfeats.parquet','lagfeats.parquet','lagfeats2.parquet','f_weekly.parquet','repro_e5.parquet']:
    df = agent_api.load_saved(t)
    print(t, df.shape)
    print(' ', [c for c in df.columns if c not in ('household_key','snapshot_day')][:40])
