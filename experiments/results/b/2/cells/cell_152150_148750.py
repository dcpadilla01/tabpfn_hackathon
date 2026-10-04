import agent_api, pandas as pd, numpy as np
for name in ['e009_demo','e009_analog','e009_spline2p','e009_demo_mkt','e008_fwd_calendar']:
    try:
        df = agent_api.load_saved(name+'.parquet')
        cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
        print('==',name, df.shape)
        print(cols)
    except Exception as e:
        print(name, 'ERR', e)
tt = agent_api.train_targets()
print('\nTARGET', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac:', (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))
