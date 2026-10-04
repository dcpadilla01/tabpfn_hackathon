import agent_api, pandas as pd
t = agent_api.load_saved('e009_ewma_longlags.parquet')
cols = list(t.columns)
print(len(cols))
print(cols)
