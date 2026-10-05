
import pandas as pd
t17 = load_saved("e017_merged.parquet")
dyn = load_saved("e018_dyn.parquet")
m = t17.merge(dyn, on=["household_key","snapshot_day"], how="left")
print(m.shape, "dups:", m.duplicated(["household_key","snapshot_day"]).sum())
print("NaN frac max:", round(m.isna().mean().max(),3))
save_table(m, "e018_merged.parquet")
