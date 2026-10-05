df = agent_api.load_saved("e001_recent_behavior.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.head())
