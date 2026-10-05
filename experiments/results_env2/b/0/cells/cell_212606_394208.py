import pandas as pd, agent_api
df = agent_api.load_saved("e001_history.parquet")
print(df.columns.tolist())
print(df.shape)
print(df.head(3))
print(agent_api.snapshot_days())
