import agent_api, pandas as pd
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
merged = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left')
print(merged.shape)
print([c for c in merged.columns if c in ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']])
agent_api.save_table(merged, 'e010_decay')
