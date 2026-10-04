import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved("e001_txhist.parquet")
print(t.shape)
print(t.columns.tolist())
print(t.head(3).T)
tt = agent_api.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
