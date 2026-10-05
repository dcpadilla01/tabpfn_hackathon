
import pandas as pd
dyn = load_saved("e018_dyn.parquet")
print(type(dyn), dyn.shape)
print(dyn.index.dtype, dyn.index.name)
print("cols:", dyn.columns.tolist()[:5])
print("is household_key a column?", "household_key" in dyn.columns)
print("index dups:", dyn.index.duplicated().sum())
t17 = load_saved("e017_merged.parquet")
print("t17 hk dtype:", t17.household_key.dtype, "dyn idx dtype:", dyn.index.dtype)
