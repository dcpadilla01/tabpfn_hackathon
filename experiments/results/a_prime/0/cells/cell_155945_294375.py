import agent_api, pandas as pd
tbl = agent_api.load_saved('stock_v1.parquet')
base = agent_api.load_saved('e011_rank.parquet')
m = base.merge(tbl, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'base', base.shape)
assert len(m)==len(base)
assert m[[c for c in tbl.columns if c.startswith('stk_')]].isna().sum().sum()==0
agent_api.save_table(m, 'e013_stock.parquet')
print('saved e013_stock.parquet')
