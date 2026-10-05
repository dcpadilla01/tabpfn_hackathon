import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
tr = df["snapshot_day"] <= 403
va = df["snapshot_day"] == 431
ytr, yva = y[tr], y[va]
ly = np.log1p(y)
Xz = np.c_[np.ones(tr.sum()), X[tr]]
z_tr = (ytr==0).astype(float)
beta = np.zeros(Xz.shape[1])
for _ in range(500):
    p = 1/(1+np.exp(-Xz@beta))
    beta -= 0.05/Xz.shape[0]*(Xz.T@(p-z_tr))
Xvz = np.c_[np.ones(va.sum()), X[va]]
pz = 1/(1+np.exp(-Xvz@beta))
mask = ytr>0
b = np.linalg.solve(X[tr][mask].T@X[tr][mask] + 1000*np.eye(X.shape[1]), X[tr][mask].T@ly[tr][mask])
pamt = np.expm1(np.clip(Xvz@b, 0, 8))
p2 = pz*pamt
print("two-part MAE@431:", round(float(np.mean(np.abs(yva-p2))),2))
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend two-part/lag1 w={w}:", round(float(np.mean(np.abs(yva-(w*p2+(1-w)*lag1)))),2))
