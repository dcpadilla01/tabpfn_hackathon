import numpy as np, pandas as pd
tt = agent_api.train_targets()
print('mean/median y by snapshot:'); print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(1))
for p in ['stock_v1','rhythm_v1','mkt_v2','rank_block' ]:
    try:
        b=agent_api.load_saved(p+'.parquet'); print(p, b.shape, list(b.columns)[:40])
    except Exception as e: print(p,'ERR',e)
t=agent_api.load_saved('e013_stock.parquet')
cols=[c for c in t.columns if c not in ('household_key','snapshot_day')]
print('e013 n cols',len(cols))
print([c for c in cols if c.startswith('m_') or c.startswith('rk_') or c.startswith('stk_') or c.startswith('rs_') or c.startswith('rz_')])
