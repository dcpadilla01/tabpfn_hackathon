import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# Ridge on standardized numeric features, fit on train snapshots <=403, validate on 431
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
yv = y.copy()

tr = df["snapshot_day"] <= 403
va = df["snapshot_day"] == 431
Xtr, ytr, Xva, yva = X[tr], yv[tr], X[va], yv[va]
def ridge(Xtr, ytr, Xva, lam):
    A_ = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1])
    b = np.linalg.solve(A_, Xtr.T@ytr)
    return Xva@b
print("ridge val MAE at 431 (lam):")
for lam in [300, 1000, 3000, 10000, 30000, 100000]:
    p = ridge(Xtr, ytr, Xva, lam)
    print(f"  lam={lam:>6}", round(float(np.mean(np.abs(yva-p))),2))
# global median baseline at 431
print("median pred MAE at 431:", round(float(np.mean(np.abs(yva-np.median(ytr)))),2))
# lag1-only
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
print("lag1 MAE at 431:", round(float(np.mean(np.abs(yva-lag1))),2))
