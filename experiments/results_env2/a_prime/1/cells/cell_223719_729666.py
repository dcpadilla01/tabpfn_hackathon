
import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")
val = f[~f.snapshot_day.isin(t.snapshot_day.unique())]

# ---- drift check ----
print("mean future_spend_4w by train snapshot_day:")
print(m.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"]).round(1).to_string())
print("\nmean spend_28 by snapshot_day (train rows + val rows):")
a = m.groupby("snapshot_day").spend_28.mean().rename("train")
b = val.groupby("snapshot_day").spend_28.mean().rename("val")
print(pd.concat([a,b],axis=1).round(1).to_string())

# ---- local ridge proxy to identify model family ----
cat_cols = ["classification_1","classification_2","classification_3","classification_4",
            "classification_5","homeowner_desc","kid_category_desc"]
num_cols = [c for c in f.columns if c not in ("household_key","snapshot_day")+tuple(cat_cols)]

def design(df, num, cats, log=False, ref=None):
    Xn = df[num].copy().astype(float)
    if log:
        Xn = np.log1p(Xn.clip(lower=0))
    if ref is None:
        mu, sd = Xn.mean(), Xn.std().replace(0,1)
        Xn = (Xn-mu)/sd
    else:
        mu, sd = ref
        Xn = (Xn-mu)/sd
    Xn = Xn.fillna(0.0).values
    blocks=[Xn]
    for c in cats:
        d = pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float)
        blocks.append(d.values)
    return np.hstack(blocks)

tr = m.dropna(subset=["spend_28"])
ytr = tr.future_spend_4w.values
va = val
yva = None

def fit_ridge(X, y, alpha):
    A_ = X.T@X + alpha*np.eye(X.shape[1]); b = X.T@y
    return np.linalg.solve(A_, b)

def val_mae(Xtr, ytr, Xva, yva, alphas=(0.1,1,10,100)):
    best=(1e9,None)
    for al in alphas:
        w = fit_ridge(Xtr,ytr,al)
        p = Xva@w
        mae = np.mean(np.abs(yva-p))
        if mae<best[0]: best=(mae,al)
    return best

# need val targets to compare with harness MAE -> approximate with our own split instead:
# hold out last 2 train snapshots (403,431) as pseudo-val
ptr = m[m.snapshot_day<=375].dropna(subset=["spend_28"])
pv = m[m.snapshot_day.isin([403,431])].dropna(subset=["spend_28"])
ytr2, yva2 = ptr.future_spend_4w.values, pv.future_spend_4w.values

for name, log in [("raw",False),("log1p",True)]:
    Xtr = design(ptr, num_cols, cat_cols, log=log)
    Xva = design(pv, num_cols, cat_cols, log=log, ref=None)  # ref misuse; recompute properly below
    # recompute with consistent ref
    Xn_tr = np.log1p(ptr[num_cols].astype(float).clip(lower=0)) if log else ptr[num_cols].astype(float)
    mu,sd = Xn_tr.mean(), Xn_tr.std().replace(0,1)
    def dm(df):
        Xn = np.log1p(df[num_cols].astype(float).clip(lower=0)) if log else df[num_cols].astype(float)
        Xn = ((Xn-mu)/sd).fillna(0.0).values
        blocks=[Xn]
        for c in cat_cols:
            blocks.append(pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).reindex(columns=cols[c], fill_value=0).values)
        return np.hstack(blocks)
    cols = {c: pd.get_dummies(ptr[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).columns for c in cat_cols}
    Xtr, Xva = dm(ptr), dm(pv)
    mae, al = val_mae(Xtr, ytr2, Xva, yva2)
    print(f"\nridge pseudo-val (snap 403/431) [{name}]: MAE={mae:.3f} alpha={al}  n_feat={Xtr.shape[1]}")
