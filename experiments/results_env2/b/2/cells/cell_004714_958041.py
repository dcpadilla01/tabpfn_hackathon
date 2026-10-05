
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved("e011_table.parquet")
print("e011:", t.shape)
print(list(t.columns))
print()

for name in ["hazard_v1","churn_vol_v1","timing_v1","deal_v1","selfcal_v1","union_all_v1","twin_v1"]:
    try:
        d = agent_api.load_saved(name + ".parquet")
        print(name, d.shape, list(d.columns)[:12])
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
print()
tt = agent_api.train_targets()
print("targets:", tt.shape, tt.columns.tolist())
k1 = set(map(tuple, t[["household_key","snapshot_day"]].values))
k2 = set(map(tuple, tt[["household_key","snapshot_day"]].values))
print("e011 keys == train keys?", k1==k2, len(k1), len(k2))
