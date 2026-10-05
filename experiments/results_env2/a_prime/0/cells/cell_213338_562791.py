import numpy as np, pandas as pd
e003 = agent_api.load_saved("e003_union.parquet")
print("e003 shape:", e003.shape)
print("cols:", e003.columns.tolist())
tt = agent_api.train_targets()
m = e003.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
y = m["future_spend_4w"]
print("\ntarget describe:\n", y.describe())
print("\nzero share:", (y==0).mean())
num = m.select_dtypes(include=[np.number]).drop(columns=["snapshot_day","future_spend_4w"], errors="ignore")
corr = num.corrwith(y).sort_values()
print("\ntop positive corr:\n", corr.tail(15).to_string())
print("\ntop negative corr:\n", corr.head(8).to_string())
