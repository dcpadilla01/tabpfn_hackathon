import agent_api as A
import pandas as pd, numpy as np

for name in ['mkt_v2','mix_v1','hist_v2']:
    t = A.load_saved(name + '.parquet')
    print(name, t.shape)
    print(list(t.columns))
    print()

tt = A.train_targets()
y = tt.future_spend_4w
print('n rows', len(tt))
print(y.describe())
print('zero share', (y == 0).mean())
print(y.quantile([.5, .75, .9, .95, .99]))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean', 'median', 'count']))
