import agent_api as api
t = api.load_saved('e001_txhist.parquet')
print(t.shape)
print(t.columns.tolist())
print(t.head(3))