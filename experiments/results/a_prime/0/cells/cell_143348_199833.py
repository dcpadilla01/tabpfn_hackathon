import pandas as pd
import agent_api as api
v2 = api.load_saved('hist_v2.parquet')
mk = api.load_saved('mkt_v1.parquet')
mg = v2.merge(mk.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True)
print(mg.shape, mg.columns[-6:].tolist())
path = api.save_table(mg, 'mkt_v2.parquet')
print(path)
