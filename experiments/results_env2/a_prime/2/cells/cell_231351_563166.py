
import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]

def prep(d):
    out = {}
    for c in feat:
        s = pd.Series(list(d[c]), index=d.index)
        if s.dtype == object or str(s.dtype) == "category":
            s = pd.Series(pd.Categorical(s.fillna("NA")).codes, index=d.index)
        out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return pd.DataFrame(out, index=d.index)

Xtr_df, Xva_df = prep(train), prep(val)
Xtr, ytr = Xtr_df.values, train.future_spend_4w.values
Xva = Xva_df.values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = np.hstack([(X-mu)/sd, np.ones((len(X),1))])
    I = np.eye(Z.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(Z.T@Z + lam*I, Z.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = np.hstack([(X-p[0])/p[1], np.ones((len(X),1))])
    return np.abs(Z@p[2]-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

print("\n|corr|>0.35 with target:")
for c in feat:
    r = np.corrcoef(Xtr_df[c].values, ytr)[0,1]
    if abs(r) > 0.35:
        print(f"  {c}: {r:.3f}")
