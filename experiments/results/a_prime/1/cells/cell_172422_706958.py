import agent_api as A, pandas as pd, numpy as np
# 1) nf_seasonal + micro: which columns are NOT in base
base = A.load_saved("e018_union_full.parquet"); bc = set(base.columns)
for n in ["nf_seasonal","micro"]:
    df = A.load_saved(n+".parquet")
    print(n, [c for c in df.columns if c not in bc and c not in ("household_key","snapshot_day")])
# 2) e017_disc_seasonal: normalize _x/_y suffixes and find truly new cols
ds = A.load_saved("e017_disc_seasonal.parquet")
norm = set(c.rstrip("_x").rstrip("_y") if c.endswith(("_x","_y")) else c for c in ds.columns)
print("e017_disc_seasonal truly-new:", sorted(c for c in norm if c not in bc and c not in ("household_key","snapshot_day","index")))
# 3) train targets + target distribution
tt = A.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
print("zero share train:", (tt.future_spend_4w==0).mean())
# 4) display_mailer overview
v = A.snapshot()
dm = v.display_mailer
print("display_mailer", dm.shape); print(dm.head(3)); print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())