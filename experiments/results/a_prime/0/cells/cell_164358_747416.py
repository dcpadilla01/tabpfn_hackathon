import agent_api as A, pandas as pd, numpy as np
for name in ['e013_stock','rhythm_v1','stock_v1','mkt_v2','e011_rank']:
    t = A.load_saved(name+'.parquet')
    print('==', name, t.shape)
    print(list(t.columns))
tt = A.train_targets()
print(tt['future_spend_4w'].describe())
print('zero share', float((tt['future_spend_4w']==0).mean()))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))
