import agent_api as A
import pandas as pd

for name in ["e013_peers", "e014_demo_l13fix", "micro", "nf_compact19", "nf_seasonal"]:
    try:
        df = A.load_saved(name + ".parquet")
        print(name, df.shape)
        print(list(df.columns)[:40])
        print("---")
    except Exception as e:
        print(name, "ERR", e)

# ---- cell ----
import agent_api as A
import pandas as pd

e12 = A.load_saved("e012_dorm.parquet")
e13 = A.load_saved("e013_peers.parquet")
e14 = A.load_saved("e014_demo_l13fix.parquet")

c12 = set(e12.columns); c13 = set(e13.columns); c14 = set(e14.columns)
print("e12 - e14:", sorted(c12 - c14))
print("e14 - e12:", sorted(c14 - c12))
print("e13 - e14:", sorted(c13 - c14))
print("e14 - e13:", sorted(c14 - c13))
print(e12.shape, e13.shape, e14.shape)

# ---- cell ----
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