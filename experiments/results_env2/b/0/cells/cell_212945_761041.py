import agent_api
e002 = agent_api.load_saved("e002_mix.parquet")
print(e002.shape)
print(list(e002.columns))
print(agent_api.snapshot_days())
