
import pandas as pd, numpy as np

tt = train_targets()
tt["zero"] = (tt.future_spend_4w == 0).astype(int)
print("zero fraction overall:", tt.zero.mean().round(4))
print(tt.groupby("snapshot_day").agg(n=("zero","size"), zero_frac=("zero","mean"),
      mean=("future_spend_4w","mean"), med=("future_spend_4w","median")).round(3))

f4 = load_saved("e004_features.parquet")
print("\ne004_features cols:", list(f4.columns))
f4n = load_saved("e004_new.parquet"); f5n = load_saved("e005_newfeats.parquet")
cols4n = [c for c in f4n.columns if c not in ("household_key","snapshot_day")]
cols5n = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in cols4n if c in f4.columns] + [c for c in cols5n if c in f4.columns]
print("dup cols:", dup)
F = f4.merge(f4n.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
dup2 = [c for c in cols5n if c in F.columns]
F = F.merge(f5n.drop(columns=dup2), on=["household_key","snapshot_day"], how="inner")
print("merged F:", F.shape)
print("snapshot_day counts:", F.snapshot_day.value_counts().sort_index().to_dict())

p5 = load_saved("e005_preds.parquet")
v = p5[predcol:=( "prediction" )]
print("\ne005 val preds low tail:", {t: float((p5.prediction < t).mean()) for t in [1,5,10,20,30]})
