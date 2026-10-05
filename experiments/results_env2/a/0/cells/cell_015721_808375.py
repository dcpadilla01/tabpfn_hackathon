
import agent_api, pandas as pd, numpy as np

P13 = agent_api.load_saved('pred_e013.parquet')
tt = agent_api.train_targets()
print(P13.dtypes)
print(tt.dtypes)
print(P13.head(3))
print(tt.head(3))
print(P13.household_key.unique()[:5], tt.household_key.unique()[:5])
