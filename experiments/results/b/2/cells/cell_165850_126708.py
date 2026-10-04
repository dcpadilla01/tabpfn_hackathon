import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets()
print('targets:', tt.shape, list(tt.columns))
print(tt[TARGET].describe().round(2))
for name in ['e013_denoise','e014_base','e014_gbm_oob','e014_stack']:
    try:
        df = load_saved(name + '.parquet')
        print('==', name, df.shape)
        print('dtypes:', dict(df.dtypes.value_counts()))
        print('first cols:', list(df.columns)[:14])
        print('last cols:', list(df.columns)[-6:])
        print('snapshots:', sorted(df['snapshot_day'].unique()))
        print('nan%:', round(df.isna().mean().mean()*100, 2))
    except Exception as e:
        print('==', name, 'ERR', repr(e))