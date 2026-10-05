import pandas as pd, numpy as np
from agent_api import load_saved, train_targets
T = train_targets()
print(T.columns.tolist(), T.shape)
print(T.head(3))
F = load_saved("allF.parquet")
tr = F.merge(T, on=["household_key","snapshot_day"], how="inner")
print(tr.columns[-5:].tolist(), len(tr))
