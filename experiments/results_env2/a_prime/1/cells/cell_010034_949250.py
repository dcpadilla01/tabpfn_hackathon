import agent_api as api
import pandas as pd

e15 = api.load_saved('e015_stack.parquet')
peer = api.load_saved('e019_peer.parquet')
m = e15.merge(peer, on=['household_key','snapshot_day'], how='inner')
print(m.shape, list(peer.columns))
p = api.save_table(m, 'e019_merged')
print('saved:', p)
