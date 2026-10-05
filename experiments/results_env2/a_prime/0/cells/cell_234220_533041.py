
import pandas as pd, numpy as np

names = ["e016_merged.parquet","e017_merged.parquet","e013_storeprod.parquet",
         "e017_phase.parquet","e016_batch2.parquet","e008_spendproc.parquet"]
tabs = {}
for nm in names:
    t = load_saved(nm)
    tabs[nm] = t
    dups = t.duplicated(["household_key","snapshot_day"]).sum()
    print(nm, t.shape, "dups:", dups)

t16, t17, t13 = tabs["e016_merged.parquet"], tabs["e017_merged.parquet"], tabs["e013_storeprod.parquet"]
k16 = set(map(tuple, t16[["household_key","snapshot_day"]].to_numpy()))
k17 = set(map(tuple, t17[["household_key","snapshot_day"]].to_numpy()))
print("key sets equal 16 vs 17:", k16==k17, "n_keys:", len(k16))

c13, c16, c17 = set(t13.columns), set(t16.columns), set(t17.columns)
print("c13<=c16:", c13<=c16, " c13<=c17:", c13<=c17)
incr16, incr17 = c16-c13, c17-c13
print("incr16:", len(incr16), "incr17:", len(incr17), "overlap:", len(incr16&incr17))
print("incr16 sample:", sorted(incr16)[:10])
print("incr17 sample:", sorted(incr17)[:10])
print("dtypes16:", dict(t16.dtypes.astype(str).value_counts()))
print("dtypes17:", dict(t17.dtypes.astype(str).value_counts()))
print("object cols16:", [c for c in t16.columns if t16[c].dtype==object][:12])
print("object cols17:", [c for c in t17.columns if t17[c].dtype==object][:12])
