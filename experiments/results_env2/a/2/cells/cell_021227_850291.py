import agent_api as A, pandas as pd
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
print("feats_v3 columns:")
for c in f3.columns: print(" ", c)
print("\nfeats_seasonal columns:", [c for c in fs.columns if c not in ("household_key","snapshot_day")])
