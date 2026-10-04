import agent_api as api
e = api.load_saved('e018_basestab.parquet')
cols = [c for c in e.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for c in cols: print(c)