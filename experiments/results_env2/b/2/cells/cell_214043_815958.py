df = agent_api.load_saved('rfm_cadence_v1.parquet')
print(df.shape)
print(list(df.columns))
print(df.head(3))