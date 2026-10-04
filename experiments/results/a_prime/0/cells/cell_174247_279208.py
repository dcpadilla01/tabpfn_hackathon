
import numpy as np, pandas as pd
for name in ['e019_knn','e013_stock','e011_rank','e010_rhythm']:
    df = agent_api.load_saved(name + '.parquet')
    print('==', name, df.shape)
    print(sorted(map(str, df.columns)))
tt = agent_api.train_targets()
print(tt.future_spend_4w.describe(percentiles=[.25,.5,.75,.9,.95,.99]))
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean', 'median']))
