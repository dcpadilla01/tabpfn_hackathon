import agent_api as api
import pandas as pd
t = api.load_saved('cand_new.parquet')
e9 = api.load_saved('e009_macro.parquet')
cand = [c for c in t.columns if c not in ('household_key','snapshot_day')]
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
m = t.merge(e9.drop(columns=['spend_28']), on=['household_key','snapshot_day'], how='inner')
print(m.shape, len(cand)+len(e9c))
p = api.save_table(m, 'e013_union')
print(p)