
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")
df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day"], how="left")
prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float)
print("X nan cols:", X.columns[X.isna().all()].tolist())
print("X inf count:", np.isinf(X.values).sum())
inf_cols = Xcolumns = [c for c in Xcols if np.isinf(X[c].values).any()]
print("inf cols:", inf_cols)
for c in inf_cols:
    print(c, df[c].describe())
