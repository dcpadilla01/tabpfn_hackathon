import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
print("te rows:", len(te), "train rows:", len(tt), "merged:", len(df))
for c in ["te_hh_mean","te_hh_shrunk","te_bin","te_prior","te_home","te_kid","te_size","te_hh_n"]:
    print(c, "n_missing:", int(df[c].isna().sum()))
# check key match
print("keys equal:", set(map(tuple,te[["household_key","snapshot_day"]].values))==set(map(tuple,tt[["household_key","snapshot_day"]].values)))
# maybe te table has different snapshot_day dtype
print(te.dtypes.head(6))
print(te.head(3)[["household_key","snapshot_day","te_hh_mean","te_hh_shrunk"]])
print(tt.head(3))
