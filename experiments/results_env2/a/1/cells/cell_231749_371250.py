import agent_api as A, pandas as pd, numpy as np
f4 = A.load_saved("e004_features.parquet")
f5 = A.load_saved("e005_newfeats.parquet")
tt = A.train_targets()
print("e004 cols:\n", [c for c in f4.columns])
print("e005 cols:\n", [c for c in f5.columns])
m = f4.merge(f5.drop(columns=[c for c in f5.columns if c in f4.columns or c in ("index",)]), on=["household_key","snapshot_day"], how="inner") if "index" in f4.columns else None
print("shapes", f4.shape, f5.shape)
