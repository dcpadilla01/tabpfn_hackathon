
import pandas as pd
m = load_saved("e018_merged.parquet")
nn = m.isna().mean().sort_values(ascending=False)
print(nn.head(10).round(3))
# check the same col in the dyn table itself
dyn = load_saved("e018_dyn.parquet")
print("dyn NaN:", dyn.isna().mean().sort_values(ascending=False).head(5).round(3))
print("t17 NaN max:", t17.isna().mean().max().round(3))
