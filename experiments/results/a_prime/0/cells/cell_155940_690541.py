import agent_api, pandas as pd
tbl = agent_api.load_saved('stock_v1.parquet')
print(list(tbl.columns)); print(tbl.head(3).to_string())
base = agent_api.load_saved('e011_rank.parquet')
print('base cols sample:', list(base.columns)[:5])
