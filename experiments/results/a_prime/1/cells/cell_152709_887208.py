import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
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
Gd, bd, Zv, yv = {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        s = [*sub, K-1]
        G = Gd[d][np.ix_(s,s)].copy(); b = bd[d][s].copy()
        G += lam*np.eye(len(s)); G[-1,-1] -= lam
        w = np.linalg.solve(G, b)
        errs.append(np.abs(np.clip(Zv[d][:,s]@w,0,None) - yv[d]))
    return float(np.concatenate(errs).mean())

rob = [c for c in fcols if c.startswith(("r_","pct_","g_","k_","s_day"))]
print("robust features:", len(rob))
uni = sorted([(loo_mae([fcols.index(c)]), c) for c in rob])
for v,c in uni: print(f"  {c:14s} {v:7.3f}")

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
e3 = [c for c in A.load_saved("e003_catmix.parquet").columns if c not in key]
tests = {
 "prev18": sel_prev,
 "prev18+robust_all": sel_prev + rob,
 "E003+robust_all": e3 + rob,
 "robust_all_only": rob,
}
for nm, cols in tests.items():
    print(f"{nm:20s} {loo_mae([fcols.index(c) for c in cols]):.3f}  n={len(cols)}")

# greedy forward from prev18 with robust candidates
cur = [fcols.index(c) for c in sel_prev]; best = loo_mae(cur)
print("\nforward from prev18:")
for step in range(10):
    cb = None
    for c in rob:
        j = fcols.index(c)
        if j in cur: continue
        v = loo_mae(cur+[j])
        if cb is None or v < cb[0]: cb = (v, j)
    v, j = cb
    print(f"  +{fcols[j]:14s} LOO={v:.3f} delta {best-v:+.3f}")
    cur = cur+[j]; best = v