import agent_api as A
base = A.load_saved('e009_ewma_longlags.parquet')
print(base.shape)
cols = list(base.columns)
print("KEYS:", [c for c in cols if c in ('household_key','snapshot_day')])
for i in range(0, len(cols), 12):
    print(i, "|".join(cols[i:i+12]))