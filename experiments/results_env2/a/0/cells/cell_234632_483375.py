
import agent_api, pandas as pd, numpy as np
pred = agent_api.load_saved('pred_e011.parquet')
tt = agent_api.train_targets()
print(pred.dtypes)
print(pred.head(3))
print(tt.dtypes)
print(tt.head(3))
print("pred snap days:", sorted(pred.snapshot_day.unique()))
