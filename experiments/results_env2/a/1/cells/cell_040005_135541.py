import numpy as np, pandas as pd, xgboost as xgb, time
allF = load_saved("allF.parquet")
print("allF:", allF.shape)
print(allF.snapshot_day.value_counts().sort_index())
print("cols sample:", list(allF.columns)[:10], "...")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day")]
print("n_feats:", len(feats))
tr = allF.merge(oof, on=["household_key","snapshot_day"])
he = allF.merge(held, on=["household_key","snapshot_day"])
print("tr:", tr.shape, "he:", he.shape)
print("tr days:", sorted(tr.snapshot_day.unique()), "he days:", sorted(he.snapshot_day.unique()))
