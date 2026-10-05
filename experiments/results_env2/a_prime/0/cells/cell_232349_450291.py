import agent_api as api
df = api.load_saved("e013_storeprod.parquet")
print(df.shape)
cols = list(df.columns)
print(len(cols))
for c in cols: print(c)