
import agent_api, pandas as pd, numpy as np
e1 = agent_api.load_saved('e001_history.parquet')
print(e1.shape)
print(list(e1.columns))
print(e1.head(3))
print(agent_api.snapshot_days())
