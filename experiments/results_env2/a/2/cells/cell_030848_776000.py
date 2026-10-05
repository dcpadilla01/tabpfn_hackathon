import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oh = A.load_saved("oof_harness.parquet")
op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
oe = A.load_saved("oof_e008.parquet")
m2 = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left").merge(oe[["household_key","snapshot_day","oof_log"]], on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_med","oof_sq","oof_log","oof_med_w112","oof_med_w224","oof_pt"]
mm = m2.dropna(subset=L)
y = mm["y"].values; P = mm[L].values

# per-snapshot optimal weights (simplex, coordinate refinement)
def refine(y, P, w0, steps=(0.1,0.05,0.02,0.01,0.005)):
    K = len(w0); w = w0.copy(); bm = np.abs(y - P@w).mean()
    for step in steps:
        improved=True
        while improved:
            improved=False
            for i in range(K):
                for j in range(K):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y-P@w2).mean()
                    if m2<bm: w,bm=w2,m2; improved=True
    return w, bm

for d in sorted(mm.snapshot_day.unique()):
    s = mm[mm.snapshot_day==d]
    ys, Ps = s["y"].values, s[L].values
    rng = np.random.default_rng(int(d))
    bm, bw = 1e9, None
    for it in range(15):
        W = rng.dirichlet(np.ones(len(L))*0.5, size=3000)
        maes = np.abs(ys[:,None]-(Ps@W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b]<bm: bm,bw = maes[b],W[b].copy()
    w,bm = refine(ys,Ps,bw)
    print(d, "n",len(s), "MAE", round(bm,3), dict(zip(L,np.round(w,2))))

# global simplex optimal on the subset (again, for reference)
rng = np.random.default_rng(3); bm, bw = 1e9, None
for it in range(40):
    W = rng.dirichlet(np.ones(len(L))*0.5, size=4000)
    maes = np.abs(y[:,None]-(P@W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b]<bm: bm,bw = maes[b],W[b].copy()
wg,bmg = refine(y,P,bw)
print("\nglobal subset optimal:", dict(zip(L,np.round(wg,3))), round(bmg,4))

# LAD stacking with unconstrained (nonneg) weights via scipy
K = len(L)
def obj(w): return np.abs(y - P@w).mean()
res = minimize(obj, np.ones(K)/K, method="Nelder-Mead", options={"maxiter":4000,"xatol":1e-4,"fatol":1e-6})
wnn = np.clip(res.x, 0, None); wnn = wnn/wnn.sum()
print("unconstrained-NM (nonneg renorm):", dict(zip(L,np.round(wnn,3))), round(obj(wnn),4))
# allow negatives
res2 = minimize(obj, wnn, method="Nelder-Mead", options={"maxiter":6000,"xatol":1e-4,"fatol":1e-6})
print("neg-allowed NM:", dict(zip(L,np.round(res2.x,3))), round(obj(res2.x),4))
