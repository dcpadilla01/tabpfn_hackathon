
import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 200)

f = agent_api.load_saved('e004_features.parquet')
print('e004_features', f.shape)
print(f.dtypes.to_string())
print(f.head(3).to_string())

p = agent_api.load_saved('e004_preds.parquet')
print('\ne004_preds', p.shape, p.columns.tolist())
print(p.prediction.describe())
print('neg preds:', (p.prediction < 0).sum())

tt = agent_api.train_targets()
print('\ntrain_targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share:', (tt.future_spend_4w == 0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['size','mean','median']).to_string())
