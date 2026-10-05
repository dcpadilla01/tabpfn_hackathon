df = agent_api.load_saved("e013_stationary.parquet")
print(df.shape)
print(sorted(df.columns.tolist()))
