import agent_api
t = agent_api.load_saved('e011_price.parquet')
print(t.shape, len(t.columns)-2)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(', '.join(cols))
