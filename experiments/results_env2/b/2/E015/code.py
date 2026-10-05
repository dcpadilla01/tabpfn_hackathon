import pandas as pd, numpy as np
e11 = load_saved("e011_table.parquet")
e14 = load_saved("e014_table.parquet")
print("e011 shape", e11.shape)
print("e014 shape", e14.shape)
cols11 = list(e11.columns); cols14 = list(e14.columns)
print("e011 cols:")
print(cols11)
newc = [c for c in cols14 if c not in cols11]
print("new cols in e14 (%d):" % len(newc), newc)
print("missing from e11 in e14:", [c for c in cols11 if c not in cols14])
k11 = e11.set_index(["household_key","snapshot_day"]).index
k14 = e14.set_index(["household_key","snapshot_day"]).index
print("keys equal:", k11.equals(k14))
tt = train_targets()
print("train targets shape", tt.shape)
print(tt.future_spend_4w.describe())
print("rows per snapshot (e014):"); print(e14.groupby("snapshot_day").size().to_dict())
nan14 = e14[newc].isna().mean()
print("new col NaN means:\n", nan14.sort_values(ascending=False).head(10))
print(e14[newc].describe().T[["mean","std","min","max"]].head(30))
path = save_table(e14, "e014_retry.parquet")
print("saved path:", repr(path))