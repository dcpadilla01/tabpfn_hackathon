import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
names = ["e001_txhist","e002_channel","e003_catmix","e004_mkt","e004_mkt_full","e006_catmix_mkt","e007_logratio","nf_candidates","nf_p1","nf_transforms"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in names}

pool = tabs["e003_catmix"].copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates"]:
    df = tabs[nm]
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
print("pool features:", len(fcols))

m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object:
        Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values
y = m.future_spend_4w.values
days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]

mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd==0]=1
Z = (X-mu)/sd
Z = np.where(np.isnan(Z), 0.0, Z)
Z = np.clip(Z, -8, 8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])

def ridge_from_gram(Ztr, ytr, Zva, lam):
    G = Ztr.T@Ztr; b = Ztr.T@ytr
    G = G + lam*np.eye(G.shape[0]); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return Zva@w

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        tr = days != d
        va = days == d
        cols = [*sub, -1] if len(sub) else [-1]
        p = ridge_from_gram(Z1[tr][:, cols], y[tr], Z1[va][:, cols], lam)
        errs.append(np.abs(np.clip(p,0,None) - y[va]))
    return float(np.concatenate(errs).mean())

base_mae = loo_mae([])
print("predict-train-mean LOO MAE:", round(base_mae,3))

uni = []
for j in range(len(fcols)):
    uni.append((loo_mae([j]), fcols[j]))
uni.sort()
print("top 25 univariate:")
for v,c in uni[:25]: print(f"  {c:28s} {v:7.3f}")
print("worst 8:")
for v,c in uni[-8:]: print(f"  {c:28s} {v:7.3f}")

e3_idx = [fcols.index(c) for c in tabs["e003_catmix"].columns if c not in key]
print("E003 full LOO:", round(loo_mae(e3_idx),3))