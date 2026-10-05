import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
cols = [c for c in te.columns if c not in ("household_key","snapshot_day")]
print("E007 rows:", len(te), "feat cols:", len(cols))
for i in range(0, len(cols), 4):
    print(" | ".join(cols[i:i+4]))

tt = A.train_targets()
y = tt[A.TARGET]
print("\nTARGET describe:\n", y.describe().round(2))
print("zero share:", round((y==0).mean(),3))
print("\nper-snapshot target mean/median/count:")
print(tt.groupby("snapshot_day")[A.TARGET].agg(["mean","median","count"]).round(1))
