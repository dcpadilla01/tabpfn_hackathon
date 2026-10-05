import agent_api as A, pandas as pd, numpy as np
f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
print(f.dtypes[["household_key","snapshot_day"]], tt.dtypes.values)
print("f hk unique sample:", f.household_key.unique()[:5], f.household_key.nunique())
print("tt hk unique sample:", tt.household_key.unique()[:5], tt.household_key.nunique())
print("f snap days:", sorted(f.snapshot_day.unique()))
print("tt snap days:", sorted(tt.snapshot_day.unique()))
# try casting
f2 = f.copy(); f2["household_key"]=f2["household_key"].astype(int); f2["snapshot_day"]=f2["snapshot_day"].astype(int)
m = tt.merge(f2.drop(columns=["index"]), on=["household_key","snapshot_day"], how="left")
print("merged after cast:", m.shape, "NaN rows:", m.drop(columns=["future_spend_4w"]).isna().any(axis=1).sum())
