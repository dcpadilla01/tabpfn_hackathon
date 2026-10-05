
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")
df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day","y"], how="left")
prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float)
y = df["y"].values
days = df["snapshot_day"].values
print("y stats: nan", np.isnan(y).sum(), "min", np.nanmin(y), "max", np.nanmax(y))
print("rows per day:", df.groupby("snapshot_day").size().to_dict())
tr_mask = ~np.isnan(y); val_mask = np.isnan(y)
print("tr", tr_mask.sum(), "val", val_mask.sum())
print("val days:", np.unique(days[val_mask]))
print("tr days:", np.unique(days[tr_mask]))
