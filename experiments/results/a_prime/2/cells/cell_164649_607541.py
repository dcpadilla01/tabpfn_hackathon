import agent_api, pandas as pd, numpy as np, re
t = agent_api.load_saved('e009_ewma_longlags.parquet')
print('shape', t.shape)
print(t.dtypes.value_counts().to_string())
print('has index:', 'index' in t.columns)
num = t.select_dtypes(include=[np.number])
print('n numeric:', num.shape[1])
neg = [c for c in num.columns if num[c].min() < 0]
print('n neg cols:', len(neg)); print(neg[:40])
naf = num.isna().mean()
print('anyNaN:', int((naf>0).sum()), 'gt30%NaN:', int((naf>0.3).sum()))

tt = agent_api.train_targets()
print('tt shape', tt.shape)
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s:(s==0).mean())
print(g.to_string())
print(tt['future_spend_4w'].describe().to_string())
