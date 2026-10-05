
import pandas as pd, numpy as np

names = ['repro_e5','e005_preds','e005_newfeats','e004_features','e004_new','allF',
         'lagfeats','lagfeats2','f_weekly','e002_features',
         'e001_preds','e003_preds','e004_preds','e007_preds','e008_preds','e009_preds','e010_preds']
store = {}
for n in names:
    try:
        df = agent_api.load_saved(n + '.parquet')
        store[n] = df
        cols = list(df.columns)
        print('==', n, df.shape, 'ncols', len(cols))
        print('   ', cols[:18], '...' if len(cols) > 18 else '')
    except Exception as e:
        print('==', n, 'ERR', repr(e))

tt = agent_api.train_targets()
print('\ntrain_targets', tt.shape)
print(tt.future_spend_4w.describe())
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero_share'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda x: (x <= 0).mean())
print(g)

r, p = store.get('repro_e5'), store.get('e005_preds')
if r is not None and p is not None:
    print('\nrepro head:'); print(r.head(3))
    print('e005_preds head:'); print(p.head(3))
