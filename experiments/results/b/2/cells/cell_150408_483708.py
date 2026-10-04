
import pandas as pd, numpy as np

paths = ['e001_recent_spend.parquet','e002_marketing.parquet','e002_marketing_v2.parquet',
         'e003_product_mix.parquet','e004_temporal.parquet','e005_longrun.parquet',
         'e006_fwd_profile.parquet','e007_log.parquet']
for p in paths:
    df = load_saved(p)
    print('==', p, df.shape)
    print(list(df.columns))
    print()

tt = train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero share:', (tt['future_spend_4w']==0).mean())
print(tt.head())
print(snapshot_days())
