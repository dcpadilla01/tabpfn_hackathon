import agent_api, numpy as np, pandas as pd
for name in ['timing_v1','lvl_v1','churn_vol_v1','selfcal_v1']:
    try:
        t = agent_api.load_saved(name+'.parquet')
        print('==',name, t.shape)
        print(list(t.columns)[:40])
    except Exception as e:
        print(name, 'ERR', e)
