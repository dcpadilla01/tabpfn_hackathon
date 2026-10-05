import agent_api
t = agent_api.load_saved("e006_zero_inflation.parquet")
print(t.shape)
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
print(len(feat))
print(feat)
