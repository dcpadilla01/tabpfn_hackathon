
import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
print("e012 dtypes:", e012[['household_key','snapshot_day']].dtypes.to_dict())
print("rawrec dtypes:", r[['household_key','snapshot_day']].dtypes.to_dict())
print("e012 hh sample:", e012.household_key.head(3).tolist())
print("rawrec hh sample:", r.household_key.head(3).tolist())
print("e012 n unique hh:", e012.household_key.nunique(), "rawrec:", r.household_key.nunique())
m = set(zip(e012.household_key, e012.snapshot_day))
m2 = set(zip(r.household_key, r.snapshot_day))
print("e012 keys not in rawrec:", len(m-m2), "rawrec keys not in e012:", len(m2-m))
miss = list(m-m2)[:5]; print("examples:", miss)
# check those households in rawrec at other days
ex_hh = miss[0][0] if miss else None
if ex_hh is not None:
    print("rawrec rows for that hh:", r[r.household_key==ex_hh][['household_key','snapshot_day','sp28']].head(20))
