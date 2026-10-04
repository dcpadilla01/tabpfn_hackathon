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
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]

# precompute per-snapshot Grams
Gd, bd, Zv, yv, nv = {}, {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]; nv[d] = va.sum()

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        s = [*sub, K-1]
        G = Gd[d][np.ix_(s,s)].copy(); b = bd[d][s].copy()
        G += lam*np.eye(len(s)); G[-1,-1] -= lam
        w = np.linalg.solve(G, b)
        p = np.clip(Zv[d][:,s]@w, 0, None)
        errs.append(np.abs(p - yv[d]))
    return float(np.concatenate(errs).mean())

# forward selection from empty
cur = []; best = loo_mae(cur); print("start (intercept only):", round(best,3))
hist = []
for step in range(18):
    cand_best = None
    for j in range(len(fcols)):
        if j in cur: continue
        v = loo_mae(cur+[j])
        if cand_best is None or v < cand_best[0]: cand_best = (v, j)
    v, j = cand_best
    hist.append((v, fcols[j]))
    print(f"step {step+1}: +{fcols[j]:28s} LOO={v:.3f} (delta {best-v:+.3f})")
    if v < best - 1e-4:
        cur = cur+[j]; best = v
    else:
        cur = cur+[j]; best = v  # keep adding anyway, monitor
print("\nselected:", [fcols[j] for j in cur])
print("final LOO:", round(best,3))

# compare: E003's 59 features
e3_idx = [fcols.index(c) for c in tabs["e003_catmix"].columns if c not in key]
print("E003 LOO:", round(loo_mae(e3_idx),3))
# E003 + top selected extras
extras = [fcols.index(c) for c in ["nf_pow90_ewm4","newm4","nf_pow90_ewm8","nf_pow90","h_act_pow90","nf_pow75_ewm","nf_pow75","newm8","nwmean12","h_act_s123","nf_pow90_ewm13","newm13"] if c in fcols]
print("E003+extras LOO:", round(loo_mae(e3_idx+extras),3))
print("extras-only LOO:", round(loo_mae(extras),3))