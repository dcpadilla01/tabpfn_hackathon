import agent_api as A, pandas as pd, numpy as np
print(A.snapshot_days())
tt = A.train_targets()
print('train_targets', tt.shape, tt.columns.tolist())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(2))
print('zero share', round((tt.future_spend_4w==0).mean(),4))
print(tt.future_spend_4w.describe().round(2))
v = A.snapshot()
tr = v.table('transactions')
print('trans', tr.shape, 'hh', tr.household_key.nunique())
pr = v.table('products'); print('products', pr.shape, pr.department.nunique())
names = ['allF.parquet','e002_features.parquet','e004_features.parquet','e004_new.parquet','e005_newfeats.parquet','f_weekly.parquet','lagfeats.parquet','lagfeats2.parquet','e005_preds.parquet','e004_preds.parquet']
for nm in names:
    try:
        df = A.load_saved(nm)
        print('==',nm, df.shape)
        print(list(df.columns))
    except Exception as e:
        print('==',nm,'ERR',type(e).__name__, e)
