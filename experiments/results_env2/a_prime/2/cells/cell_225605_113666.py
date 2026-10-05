import numpy as np, pandas as pd
base = agent_api.load_saved("e005_decay_gapcv.parquet")
print("base cols:", base.columns.tolist())
print("base shape:", base.shape)
tt = agent_api.train_targets()
print("targets:", tt.shape, "| zero frac:", (tt.future_spend_4w==0).mean())
print(tt.future_spend_4w.describe())
