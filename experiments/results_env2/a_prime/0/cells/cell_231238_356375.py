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

def ridge_solve(Xtr, Ytr, lam):
    A_ = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1])
    return np.linalg.solve(A_, Xtr.T@Ytr)

# log-target ridge
ly = np.log1p(y)
for lam in [100, 300, 1000, 3000, 10000]:
    b = ridge_solve(X[tr], ly[tr], lam)
    p = np.expm1(np.clip(X[va]@b, 0, 8))
    print(f"log-target ridge lam={lam:>5} MAE@431:", round(float(np.mean(np.abs(yva-p))),2))

# two-part: logistic zero + ridge on log for nonzero
z_tr = (ytr==0).astype(float)
Xz = np.c_[np.ones(tr.sum()), X[tr]]
beta = np.zeros(Xz.shape[1])
for _ in range(500):
    p = 1/(1+np.exp(-Xz@beta))
    beta -= 0.05/Xz.shape[0]*(Xz.T@(p-z_tr))
Xvz = np.c_[np.ones(va.sum()), X[va]]
pz = 1/(1+np.exp(-Xvz@beta))
b = ridge_solve(X[tr][ytr>0], ly[tr][ytr>0], 1000)
pamt = np.expm1(np.clip(Xvz@b, 0, 8))
p2 = pz*pamt
print("two-part MAE@431:", round(float(np.mean(np.abs(yva-p2))),2))
# blend with lag1
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend two-part/lag1 w={w}:", round(float(np.mean(np.abs(yva-(w*p2+(1-w)*lag1)))),2))
