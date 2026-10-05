
import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()
print("train", train.shape, "val", val.shape)

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
Xtr = train[feat].astype(float).fillna(0).values
ytr = train.future_spend_4w.values
Xva = val[feat].astype(float).fillna(0).values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = (X-mu)/sd
    A = np.hstack([Z, np.ones((len(Z),1))])
    I = np.eye(A.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = (X-p[0])/p[1]
    pred = np.hstack([Z, np.ones((len(Z),1))]) @ p[2]
    return np.abs(pred-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

# correlation of target with key features
for c in ["spend_28","ew_28","spend_84","lr_mean28","spend_7","ew_7","days_since_last","active_28","spend_364"]:
    print(c, np.corrcoef(train[c].fillna(0), ytr)[0,1].round(3))
