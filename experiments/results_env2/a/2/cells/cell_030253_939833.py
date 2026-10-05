import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
print("oof_e016_cv", oof.shape, oof.columns.tolist())
print(oof.groupby("snapshot_day").size())

y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
print("\nindividual OOF MAE / corr-with-y / pairwise corr:")
for k,l in enumerate(L):
    print(f"{l:9s} MAE {np.abs(y-P[:,k]).mean():.4f}  corr {np.corrcoef(y,P[:,k])[0,1]:.4f}")
C = np.corrcoef(P.T)
print("pairwise corr matrix:\n", np.round(C,3))

def mae(w, y, P):
    return np.abs(y - P @ w).mean()

# E016-style hand weights guesses
for w in [[0.4,0.3,0.3,0.0],[0.4,0.0,0.3,0.3],[0.34,0.33,0.33,0.0],[0.25,0.25,0.25,0.25]]:
    print("w",w,"OOF MAE", round(mae(np.array(w),y,P),4))

# random Dirichlet search over simplex
rng = np.random.default_rng(0)
W = rng.dirichlet(np.ones(4)*0.5, size=300000)
maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
best = maes.argmin()
print("\nbest random w:", np.round(W[best],3), "OOF MAE", round(maes[best],4))

# local refinement around best
bw = W[best].copy()
for step in [0.2,0.1,0.05,0.02,0.01,0.005]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for j in range(4):
                if i==j or bw[j]<step: continue
                w2 = bw.copy(); w2[i]+=step; w2[j]-=step
                if mae(w2,y,P) < mae(bw,y,P):
                    bw = w2; improved=True
print("refined w:", np.round(bw,4), "OOF MAE", round(mae(bw,y,P),4))

# also check other OOF tables for extra learners
oh = A.load_saved("oof_harness.parquet"); op = A.load_saved("oof_pt.parquet"); oe = A.load_saved("oof_e008.parquet")
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left", suffixes=("","_pt"))
print("\nmerged harness:", m[["oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]].isna().mean().round(3).to_dict())
L2 = L + ["oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
P2 = m[L2].values; ok = ~np.isnan(P2).any(axis=1)
y2, P2 = y[ok], P2[ok]
print("rows with all:", ok.sum(), "of", len(y))
for k,l in enumerate(L2):
    print(f"{l:12s} MAE {np.abs(y2-P2[:,k]).mean():.4f}")
W2 = rng.dirichlet(np.ones(len(L2))*0.5, size=300000)
maes2 = np.abs(y2[:,None] - (P2 @ W2.T)).mean(axis=0)
b2 = maes2.argmin()
print("best random (9 learners):", dict(zip(L2, np.round(W2[b2],3))), "OOF MAE", round(maes2[b2],4))
