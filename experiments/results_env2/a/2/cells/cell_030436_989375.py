import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
print("individual OOF MAE / corr:")
for k,l in enumerate(L):
    print(f"{l:9s} MAE {np.abs(y-P[:,k]).mean():.4f}  corr {np.corrcoef(y,P[:,k])[0,1]:.4f}")
print("pairwise corr:\n", np.round(np.corrcoef(P.T),3))

def mae_w(w): return np.abs(y - P @ w).mean()
for w in [[0.4,0.3,0.3,0.0],[0.4,0.0,0.3,0.3],[0.34,0.33,0.33,0.0],[0.25,0.25,0.25,0.25],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]:
    print("w",w,"OOF MAE", round(mae_w(np.array(w)),4))

# chunked random Dirichlet search
rng = np.random.default_rng(0)
best_mae, best_w = 1e9, None
for it in range(20):
    W = rng.dirichlet(np.ones(4)*0.5, size=5000)          # 5000x4
    PW = P @ W.T                                          # 26437 x 5000  (~1e8 ok)
    maes = np.abs(y[:,None] - PW).mean(axis=0)
    b = maes.argmin()
    if maes[b] < best_mae: best_mae, best_w = maes[b], W[b]
print("best random w:", np.round(best_w,4), "OOF MAE", round(best_mae,4))

# local refinement (coordinate exchange on simplex)
bw = best_w.copy(); bm = mae_w(bw)
for step in [0.2,0.1,0.05,0.02,0.01,0.005,0.002]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for j in range(4):
                if i==j or bw[j]<step: continue
                w2 = bw.copy(); w2[i]+=step; w2[j]-=step
                m2 = mae_w(w2)
                if m2 < bm: bw, bm = w2, m2; improved=True
print("refined w:", np.round(bw,4), "OOF MAE", round(bm,4))
