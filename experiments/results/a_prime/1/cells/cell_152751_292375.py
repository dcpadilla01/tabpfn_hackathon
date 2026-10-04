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

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
sub = [fcols.index(c) for c in sel_prev] + [K-1]

def fit_pred(d, lam=30.0):
    tr = days != d; va = days == d
    G = Z1[tr][:,sub].T@Z1[tr][:,sub]; b = Z1[tr][:,sub].T@y[tr]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return np.clip(Z1[va][:,sub]@w,0,None), y[va], va

# error decomposition for last snapshot as example
p, yy, va = fit_pred(431)
zmask = yy==0
print("MAE overall:", round(float(np.abs(p-yy).mean()),2))
print("MAE on zero-target rows:", round(float(np.abs(p[zmask]-0).mean()),2), " share:", round(float(zmask.mean()),3))
print("MAE on positive rows:", round(float(np.abs(p[~zmask]-yy[~zmask]).mean()),2))
print("mean pred on zero rows:", round(float(p[zmask].mean()),1))
# by target bucket
for lo,hi in [(0,0),(1,50),(50,100),(100,200),(200,400),(400,2200)]:
    mm = (yy>=lo)&(yy<=hi)
    print(f"  y in [{lo},{hi}]: n={mm.sum():5d} mean_pred={p[mm].mean():7.1f} mean_y={yy[mm].mean():7.1f} MAE={np.abs(p[mm]-yy[mm]).mean():7.1f}")

# how predictable is zero? univariate AUC-ish with simple features
zall = (y==0).astype(float)
from statistics import mean as _m
def auc_score(s, z):
    s = np.asarray(s, float); z = np.asarray(z, float)
    ok = ~np.isnan(s)
    s, z = s[ok], z[ok]
    r = pd.Series(s).rank().values
    n1, n0 = z.sum(), (1-z).sum()
    return (r[z==1].sum() - n1*(n1+1)/2) / (n1*n0)
cands = ["r_dsl","r_zerom13","r_zerow13","r_zerow4","r_gap_ratio","days_since_last","zero_recent","active_share_l1","nf_pow90_ewm4","spend_l1","tenure","r_gap_mean","h_inact_s123"]
for c in cands:
    if c in fcols:
        a = auc_score(X[:, fcols.index(c)], zall)
        print(f"AUC({c:16s}) = {a:.3f}")

# two-part model test (LOO on last few snapshots): p0 linear prob + level ridge on positives
def two_part(d, lam=30.0):
    tr = days != d; va = days == d
    # p0
    G = Z1[tr][:,sub].T@Z1[tr][:,sub]; b = Z1[tr][:,sub].T@zall[tr]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w0 = np.linalg.solve(G, b)
    p0 = np.clip(Z1[va][:,sub]@w0, 0, 1)
    # level on positives
    trp = tr & (y>0)
    G = Z1[trp][:,sub].T@Z1[trp][:,sub]; b = Z1[trp][:,sub].T@y[trp]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    wl = np.linalg.solve(G, b)
    pl = np.clip(Z1[va][:,sub]@wl, 0, None)
    return (1-p0)*pl, y[va]

errs1, errs2 = [], []
for d in TRAIN:
    p1, yy1 = fit_pred(d)[:2]
    p2, yy2 = two_part(d)
    errs1.append(np.abs(p1-yy1)); errs2.append(np.abs(p2-yy2))
print("\nLOO MAE single ridge:", round(float(np.concatenate(errs1).mean()),3))
print("LOO MAE two-part    :", round(float(np.concatenate(errs2).mean()),3))