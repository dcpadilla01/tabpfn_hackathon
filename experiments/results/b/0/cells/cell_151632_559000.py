import agent_api, pandas as pd
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
merged = e7.merge(cand, on=['household_key','snapshot_day'], how='left')
print(merged.shape)
agent_api.save_table(merged, 'e010_decay')
