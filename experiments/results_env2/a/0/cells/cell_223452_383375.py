import agent_api as A, pandas as pd, numpy as np
f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
m = tt.merge(f.drop(columns=["index"]), on=["household_key","snapshot_day"], how="inner")
print("inner merge:", m.shape)
# check which train rows are missing from feats_v3
key_t = set(zip(tt.household_key, tt.snapshot_day))
key_f = set(zip(f.household_key, f.snapshot_day))
missing = key_t - key_f
print("train keys missing in feats_v3:", len(missing))
import collections
c = collections.Counter(d for h,d in missing)
print(sorted(c.items()))
