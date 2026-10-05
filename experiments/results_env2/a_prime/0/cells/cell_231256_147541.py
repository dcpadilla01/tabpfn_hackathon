import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
print("n num cols:", len(num), "rows:", len(df))
Xdf = df[num].astype(float)
print("Xdf shape:", Xdf.shape)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
print("X shape:", X.shape)
tr = (df["snapshot_day"] <= 403).values
va = (df["snapshot_day"] == 431).values
print("tr sum:", tr.sum(), "va sum:", va.sum(), "X[tr] shape:", X[tr].shape)
