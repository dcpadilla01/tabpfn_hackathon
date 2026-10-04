import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e012_robust.parquet')
cols = t.columns.tolist()
# print all columns in chunks
for i in range(0, len(cols), 25):
    print(i, cols[i:i+25])
