
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
print('e013 shape', e013.shape)
for name in ['stock_v1','rhythm_v1','mix_v1','season_v1']:
    t = A.load_saved(name+'.parquet')
    print(name, t.shape, list(t.columns)[:30])
tt = A.train_targets()
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median','std','max']))
print('zero rate overall', (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.apply(lambda s:(s==0).mean()))
