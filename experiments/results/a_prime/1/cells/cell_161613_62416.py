import agent_api as A
import pandas as pd

e14 = A.load_saved("e014_demo_l13fix.parquet")
e13 = A.load_saved("e013_peers.parquet")

peer_cols = ['anchor_ly_pop','cohort_prior4w','has_ly4w','peer_p10','peer_p90','peer_ratio','peer_ratio2','peer_recent28','peer_recent28_med','pop_ratio','spend_ly4w']

# dedupe household_key+snapshot_day
e14u = e14.drop_duplicates(subset=['household_key','snapshot_day'])
e13u = e13.drop_duplicates(subset=['household_key','snapshot_day'])[['household_key','snapshot_day']+peer_cols]

m = e14u.merge(e13u, on=['household_key','snapshot_day'], how='left')
print(m.shape, "nulls in peer cols:", m[peer_cols].isna().mean().round(3).to_dict())

path = A.save_table(m, "e015_peers_demo_l13fix.parquet")
print(path)